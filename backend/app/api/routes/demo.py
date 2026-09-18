from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from ...config import get_settings
from ...providers.mock import MockCourtProvider


router = APIRouter(tags=["demo"])


@router.get("/demo", include_in_schema=False)
async def demo_page() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parents[4] / "frontend" / "demo.html")


@router.get("/api/demo/status")
async def demo_status() -> dict[str, str | bool]:
    settings = get_settings()
    court_provider = settings.court_provider.strip().lower()
    llm_provider = settings.llm_provider.strip().lower()
    court_ready = court_provider == "mock" or bool(settings.parser_api_key)

    if llm_provider == "yandex":
        llm_ready = bool(settings.yandex_api_key and settings.yandex_folder_id)
        return {
            "court_provider": court_provider,
            "llm_provider": llm_provider,
            "model": settings.yandex_model,
            "provider_ready": court_ready,
            "llm_ready": llm_ready,
            "ready": court_ready and llm_ready,
            "ollama_ready": False,
            "model_available": False,
        }

    if llm_provider != "ollama":
        return {
            "court_provider": court_provider,
            "llm_provider": llm_provider,
            "model": "",
            "provider_ready": court_ready,
            "llm_ready": False,
            "ready": False,
            "ollama_ready": False,
            "model_available": False,
        }

    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        available = any(
            item.get("name") == settings.ollama_model
            or item.get("model") == settings.ollama_model
            for item in models
        )
        ready = True
    except (httpx.HTTPError, ValueError, TypeError):
        available = False
        ready = False
    return {
        "court_provider": court_provider,
        "llm_provider": settings.llm_provider,
        "model": settings.ollama_model,
        "provider_ready": court_ready,
        "llm_ready": ready and available,
        "ready": court_ready and ready and available,
        "ollama_ready": ready,
        "model_available": available,
    }


@router.get("/api/demo/documents/{document_id}/pdf")
async def demo_pdf(document_id: str) -> Response:
    provider = MockCourtProvider()
    pdf = await provider.download_pdf(f"/api/demo/documents/{document_id}/pdf")
    if pdf is None:
        raise HTTPException(status_code=404, detail="Demo document not found")
    return Response(content=pdf, media_type="application/pdf")
