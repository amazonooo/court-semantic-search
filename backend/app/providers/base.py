from abc import ABC, abstractmethod

from ..models import DocumentSearchParams, DocumentSearchResult


class CourtProviderError(RuntimeError):
    def __init__(self, message: str, *, error_code: str | int | None = None):
        super().__init__(message)
        self.error_code = error_code


class CourtProviderConfigurationError(CourtProviderError):
    pass


class CourtProviderValidationError(CourtProviderError):
    pass


class CourtProviderAccessError(CourtProviderError):
    pass


class CourtProviderTemporaryError(CourtProviderError):
    pass


class CourtProvider(ABC):
    @abstractmethod
    async def search_documents(self, params: DocumentSearchParams) -> DocumentSearchResult:
        raise NotImplementedError

    @abstractmethod
    async def download_pdf(self, file_url: str) -> bytes | None:
        raise NotImplementedError
