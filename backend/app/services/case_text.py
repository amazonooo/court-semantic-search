from time import monotonic

from ..models import CaseDocumentTextResponse, CourtCase, PreferredDocumentTextResponse
from ..providers.base import CourtProvider
from .pdf import extract_pdf_text_async
from .search_runtime import active_diagnostics


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
    pdf_bytes = await provider.download_pdf(document.file_url)
    if pdf_bytes is None:
        raise PreferredDocumentNotFoundError(
            f"{role.replace('_', ' ').capitalize()} PDF document was not found by the source"
        )

    text = await _extract_with_timing(pdf_bytes)
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
    pdf_bytes = await provider.download_pdf(document.file_url)
    if pdf_bytes is None:
        raise PreferredDocumentNotFoundError(
            "Preferred PDF document was not found by the source"
        )

    text = await _extract_with_timing(pdf_bytes)
    return PreferredDocumentTextResponse(
        case_id=case.case_id,
        case_number=case.case_number,
        document_id=document.document_id,
        document=document,
        text=text,
        char_count=len(text),
    )
