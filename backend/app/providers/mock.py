from ..models import DocumentSearchParams, DocumentSearchResult
from .base import CourtProvider


class MockCourtProvider(CourtProvider):
    async def search_documents(self, params: DocumentSearchParams) -> DocumentSearchResult:
        return DocumentSearchResult(count=0, pages=0, page=params.page, items=[])

    async def download_pdf(self, file_url: str) -> bytes | None:
        return None
