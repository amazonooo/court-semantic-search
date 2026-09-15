from ..models import CourtCase, PreferredDocumentTextResponse
from ..providers.base import CourtProvider
from .pdf import extract_pdf_text


class PreferredDocumentNotFoundError(RuntimeError):
    pass


async def extract_preferred_document_text(
    case: CourtCase,
    provider: CourtProvider,
) -> PreferredDocumentTextResponse:
    document = case.preferred_document
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
