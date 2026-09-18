import httpx

from ..models import SearchPlan
from .base import LlmError, QueryPlanner
from .prompt import SYSTEM_PROMPT, parse_search_plan


YANDEX_COMPLETION_URL = (
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
)


def _search_plan_json_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
            },
            "must_have": {
                "type": "array",
                "items": {"type": "string"},
            },
            "exclude": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["queries", "must_have", "exclude"],
    }


class YandexQueryPlanner(QueryPlanner):
    def __init__(
        self,
        api_key: str,
        folder_id: str,
        *,
        model: str = "yandexgpt-5-lite",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._folder_id = folder_id
        self._model = model
        self._client = client

    async def plan(self, description: str) -> SearchPlan:
        payload = {
            "modelUri": f"gpt://{self._folder_id}/{self._model}",
            "completionOptions": {
                "stream": False,
                "temperature": 0.2,
                "maxTokens": "1000",
                "reasoningOptions": {"mode": "DISABLED"},
            },
            "messages": [
                {"role": "system", "text": SYSTEM_PROMPT},
                {"role": "user", "text": description},
            ],
            "jsonSchema": {
                "schema": _search_plan_json_schema(),
            },
        }

        headers = {
            "Authorization": f"Api-Key {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            if self._client is None:
                async with httpx.AsyncClient(timeout=60) as client:
                    response = await client.post(
                        YANDEX_COMPLETION_URL,
                        headers=headers,
                        json=payload,
                    )
            else:
                response = await self._client.post(
                    YANDEX_COMPLETION_URL,
                    headers=headers,
                    json=payload,
                )
        except httpx.RequestError as exc:
            raise LlmError("Yandex AI Studio is unreachable") from exc

        if response.status_code != 200:
            detail = response.text.strip().replace("\n", " ")
            raise LlmError(
                f"Yandex AI Studio returned HTTP "
                f"{response.status_code}: {detail}"
            )

        try:
            body = response.json()
            result = body.get("result", body)
            content = result["alternatives"][0]["message"]["text"]
        except (AttributeError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise LlmError(
                "Yandex AI Studio returned an invalid response"
            ) from exc

        return parse_search_plan(content)
