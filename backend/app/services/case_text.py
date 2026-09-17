from ..models import CaseDocumentTextResponse, CourtCase, PreferredDocumentTextResponse
from ..providers.base import CourtProvider
from .pdf import extract_pdf_text


class PreferredDocumentNotFoundError(RuntimeError):
    pass


def _document_for_role(case: CourtCase, role: str):
    if role == "factual_base":
        return (
            case.factual_base_document
            or case.first_instance_document
            or (case.first_instance_documents[0] if case.first_instance_documents else None)
            or case.latest_substantive_document
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
    document = _document_for_role(case, role)
    if document is None:
        raise PreferredDocumentNotFoundError(
            f"{role.replace('_', ' ').capitalize()} PDF document was not found in the case"
        )

    pdf_bytes = await provider.download_pdf(document.file_url)
    if pdf_bytes is None:
        raise PreferredDocumentNotFoundError(
            f"{role.replace('_', ' ').capitalize()} PDF document was not found by the source"
        )

    text = extract_pdf_text(pdf_bytes)
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
    if document is None:
        raise PreferredDocumentNotFoundError(
            "Preferred PDF document was not found in the case"
        )
    pdf_bytes = await provider.download_pdf(document.file_url)
    if pdf_bytes is None:
        raise PreferredDocumentNotFoundError(
            "Preferred PDF document was not found by the source"
        )

    text = extract_pdf_text(pdf_bytes)
    return PreferredDocumentTextResponse(
        case_id=case.case_id,
        case_number=case.case_number,
        document_id=document.document_id,
        document=document,
        text=text,
        char_count=len(text),
    )
