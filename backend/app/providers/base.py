from abc import ABC, abstractmethod
from typing import Any


class CourtProvider(ABC):
    @abstractmethod
    async def search_documents(self, query: str, filters: dict[str, Any] | None = None):
        raise NotImplementedError

    @abstractmethod
    async def get_document(self, document_id: str):
        raise NotImplementedError

    @abstractmethod
    async def get_case(self, case_number: str):
        raise NotImplementedError
