from time import monotonic
from collections import OrderedDict

from ..models import CaseDocumentTextResponse, CourtCase, PreferredDocumentTextResponse
from ..providers.base import CourtProvider
from .pdf import extract_pdf_text_async
from .cases import CaseAggregationService
from .search_runtime import active_diagnostics


_text_cache = OrderedDict()
_CACHE_TTL = 3600
_CACHE_CHARS = 5_000_000


async def _document_text(document, provider, *, label='PDF'):
    # A PDF URL identifies an immutable court act. Separate providers (and
    # credentials) never share cache entries. Failed or blank extractions are
    # not cached, and legal/model judgments are always recomputed.
    source = getattr(provider, 'provider', provider)
    key = (source, document.document_id, document.file_url)
    now = monotonic()
    for old_key, (saved, _) in list(_text_cache.items()):
        if now - saved > _CACHE_TTL:
            del _text_cache[old_key]
    if key in _text_cache:
        _, text = _text_cache[key]
        _text_cache.move_to_end(key)
        diagnostics = active_diagnostics.get()
        if diagnostics is not None:
            diagnostics.pdf_cache_hits += 1
        return text
    pdf_bytes = await provider.download_pdf(document.file_url)
    if pdf_bytes is None:
        raise PreferredDocumentNotFoundError(f'{label} document was not found by the source')
    text = await _extract_with_timing(pdf_bytes)
    if text.strip():
        _text_cache[key] = (now, text)
        while sum(len(value[1]) for value in _text_cache.values()) > _CACHE_CHARS:
            _text_cache.popitem(last=False)
    return text


async def _extract_with_timing(pdf_bytes):
    start = monotonic()
    try:
        return await extract_pdf_text_async(pdf_bytes)
    finally:
        diagnostics = active_diagnostics.get()
        if diagnostics is not None:
            diagnostics.stage_seconds['pdf_extraction'] = round(
                diagnostics.stage_seconds.get('pdf_extraction', 0) + monotonic() - start, 3)


class PreferredDocumentNotFoundError(RuntimeError):
    pass


def document_for_role(case: CourtCase, role: str):
    if role == "search_evidence":
        substantive = [document for document in case.documents
                       if CaseAggregationService.document_status(document) == 'substantive']
        first_instance = [document for document in substantive if document.instance_level == 1]
        candidates = first_instance or substantive or [
            document for document in case.documents
            if CaseAggregationService.document_status(document) == 'unknown'
        ]
        return max(candidates, key=CaseAggregationService._factual_base_sort_key) if candidates else None
    if role == "factual_base":
        return (
            case.factual_base_document
            or case.first_instance_document
            or case.latest_substantive_document
            or (case.first_instance_documents[0] if case.first_instance_documents else None)
            or case.preferred_document
        )
    if role == "latest_substantive":
        return (
            case.latest_substantive_document
            or case.preferred_document
            or case.factual_base_document
            or case.first_instance_document
        )
    raise ValueError(f"Unsupported case document role: {role}")


async def extract_case_document_text(
    case: CourtCase,
    provider: CourtProvider,
    *,
    role: str,
) -> CaseDocumentTextResponse:
    document = document_for_role(case, role)
    if document is None or not document.file_url:
        raise PreferredDocumentNotFoundError(
            f"{role.replace('_', ' ').capitalize()} PDF document was not found in the case"
        )

    if case.case_id and document.case_id != case.case_id:
        raise PreferredDocumentNotFoundError("Selected document belongs to another case")
    text = await _document_text(document, provider, label=f"{role.replace('_', ' ').capitalize()} PDF")
    return CaseDocumentTextResponse(
        case_id=case.case_id,
        case_number=case.case_number,
        role=role,
        document_id=document.document_id,
        document=document,
        text=text,
        char_count=len(text),
    )


async def extract_preferred_document_text(
    case: CourtCase,
    provider: CourtProvider,
) -> PreferredDocumentTextResponse:
    document = (
        case.preferred_document
        or case.latest_substantive_document
        or case.factual_base_document
        or case.first_instance_document
    )
    if document is None or not document.file_url:
        raise PreferredDocumentNotFoundError(
            "Preferred PDF document was not found in the case"
        )
    if case.case_id and document.case_id != case.case_id:
        raise PreferredDocumentNotFoundError("Selected document belongs to another case")
    text = await _document_text(document, provider, label='Preferred PDF')
    return PreferredDocumentTextResponse(
        case_id=case.case_id,
        case_number=case.case_number,
        document_id=document.document_id,
        document=document,
        text=text,
        char_count=len(text),
    )
