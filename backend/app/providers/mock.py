from .base import CourtProvider


class MockCourtProvider(CourtProvider):
    async def search_documents(self, query: str, filters=None):
        return []

    async def get_document(self, document_id: str):
        return None

    async def get_case(self, case_number: str):
        return None
