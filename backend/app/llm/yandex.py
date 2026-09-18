import httpx

from ..models import SearchPlan
from .base import LlmError, QueryPlanner
from .prompt import SYSTEM_PROMPT, parse_search_plan


class YandexQueryPlanner(QueryPlanner):
    def __init__(
        self,
        api_key: str,
        folder_id: str,
        *,
        model: str = "yandexgpt-lite",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._folder_id = folder_id
        self._model = model
        self._client = client

    async def plan(self, description: str) -> SearchPlan:
        payload = {
            "modelUri": f"gpt://{self._folder_id}/{self._model}",
            "completionOptions": {"stream": False, "temperature": 0.2, "maxTokens": 1000},
            "messages": [
                {"role": "system", "text": SYSTEM_PROMPT},
                {"role": "user", "text": description},
            ],
            "jsonSchema": {"schema": SearchPlan.model_json_schema()},
        }
        url = "https://ai.api.cloud.yandex.net/foundationModels/v1/completion"
        headers = {"Authorization": f"Api-Key {self._api_key}"}
        try:
            if self._client is None:
                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.post(url, headers=headers, json=payload)
            else:
                response = await self._client.post(url, headers=headers, json=payload)
        except httpx.RequestError as exc:
            raise LlmError("Yandex AI Studio is unreachable") from exc
        if response.status_code != 200:
            raise LlmError(f"Yandex AI Studio returned HTTP {response.status_code}")
        try:
            content = response.json()["result"]["alternatives"][0]["message"]["text"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LlmError("Yandex AI Studio returned an invalid response") from exc
        return parse_search_plan(content)
