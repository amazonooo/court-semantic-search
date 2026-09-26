import ssl
from datetime import date

import httpx
import truststore

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
            "filters": {
                "type": "object",
                "properties": {key: {"type": ["string", "null"]} for key in (
                    "case_number", "inn", "court", "date_from", "date_to",
                    "dispute_type", "dispute_category",
                )},
                "required": ["case_number", "inn", "court", "date_from", "date_to",
                             "dispute_type", "dispute_category"],
                "additionalProperties": False,
            },
            "exclude": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["queries", "must_have", "exclude", "filters"],
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
                {"role": "system", "text": SYSTEM_PROMPT + f" Сегодня {date.today().isoformat()}."},
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
                ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                async with httpx.AsyncClient(
                    timeout=60,
                    verify=ssl_context,
                ) as client:
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
            raise LlmError(
                f"Yandex AI Studio is unreachable ({type(exc).__name__})"
            ) from exc

        if response.status_code != 200:
            if response.status_code == 403:
                raise LlmError("Yandex AI Studio denied access (HTTP 403); check the service account, API key scope and folder permissions")
            detail = response.text.strip().replace("\n", " ").replace(self._api_key, "[REDACTED]")[:500]
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

        return parse_search_plan(content, description)
