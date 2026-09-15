from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...models import (
    DocumentSearchParams,
    DocumentSearchResult,
    PdfTextRequest,
    PdfTextResponse,
)
from ...providers.base import CourtProvider
from ...providers.factory import get_court_provider
from ...services.pdf import PdfExtractionError, extract_pdf_text

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("/search", response_model=DocumentSearchResult)
async def search_documents(
    params: Annotated[DocumentSearchParams, Query()],
    provider: Annotated[CourtProvider, Depends(get_court_provider)],
) -> DocumentSearchResult:
    return await provider.search_documents(params)


@router.post("/extract-text", response_model=PdfTextResponse)
async def extract_document_text(
    request: PdfTextRequest,
    provider: Annotated[CourtProvider, Depends(get_court_provider)],
) -> PdfTextResponse:
    pdf_bytes = await provider.download_pdf(request.file_url)
    if pdf_bytes is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PDF document was not found by the source",
        )

    try:
        text = extract_pdf_text(pdf_bytes)
    except PdfExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return PdfTextResponse(
        file_url=request.file_url,
        text=text,
        char_count=len(text),
    )
