from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from .api.routes.cases import router as cases_router
from .api.routes.documents import router as documents_router
from .api.routes.demo import router as demo_router
from .providers.base import (
    CourtProviderAccessError,
    CourtProviderConfigurationError,
    CourtProviderError,
    CourtProviderTemporaryError,
    CourtProviderValidationError,
)

app = FastAPI(title="Court Semantic Search API", version="0.4.0")
app.include_router(documents_router)
app.include_router(cases_router)
app.include_router(demo_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.exception_handler(CourtProviderValidationError)
async def handle_provider_validation_error(
    request: Request, exc: CourtProviderValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_code": exc.error_code},
    )


@app.exception_handler(CourtProviderAccessError)
async def handle_provider_access_error(
    request: Request, exc: CourtProviderAccessError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc), "error_code": exc.error_code},
    )


@app.exception_handler(CourtProviderTemporaryError)
async def handle_provider_temporary_error(
    request: Request, exc: CourtProviderTemporaryError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc), "error_code": exc.error_code},
    )


@app.exception_handler(CourtProviderConfigurationError)
async def handle_provider_configuration_error(
    request: Request, exc: CourtProviderConfigurationError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "error_code": exc.error_code},
    )


@app.exception_handler(CourtProviderError)
async def handle_provider_error(
    request: Request, exc: CourtProviderError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc), "error_code": exc.error_code},
    )
