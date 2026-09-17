import httpx

from ..models import SearchPlan
from .base import LlmError, QueryPlanner
from .prompt import SYSTEM_PROMPT, parse_search_plan


class OllamaQueryPlanner(QueryPlanner):
    def __init__(
        self,
        *,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = client

    async def plan(self, description: str) -> SearchPlan:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": description},
            ],
            "format": SearchPlan.model_json_schema(),
            "stream": False,
            "think": False,
            "options": {"temperature": 0.2},
        }
        try:
            if self._client is None:
                async with httpx.AsyncClient(timeout=120) as client:
                    response = await client.post(f"{self._base_url}/api/chat", json=payload)
            else:
                response = await self._client.post(f"{self._base_url}/api/chat", json=payload)
        except httpx.RequestError as exc:
            raise LlmError("Ollama is unreachable; start Ollama and pull the configured model") from exc
        if response.status_code != 200:
            raise LlmError(f"Ollama returned HTTP {response.status_code}")
        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LlmError("Ollama returned an invalid response") from exc
        return parse_search_plan(content)
