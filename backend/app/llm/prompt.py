import json

from pydantic import ValidationError

from ..models import SearchPlan
from .base import LlmError


SYSTEM_PROMPT = (
    "Ты составляешь поисковые запросы по судебным актам арбитражных судов РФ. "
    "Верни JSON с 2-5 разными короткими русскоязычными запросами, сохраняющими "
    "конкретную схему фактов и юридический вопрос. Не придумывай факты. "
    "must_have — ключевые признаки, exclude — явно нежелательные темы. "
    "Не делай каждый запрос слишком узким."
)


def parse_search_plan(content: str) -> SearchPlan:
    try:
        return SearchPlan.model_validate(json.loads(content))
    except (ValueError, ValidationError) as exc:
        raise LlmError("LLM returned an invalid search plan") from exc
