from abc import ABC, abstractmethod

from ..models import SearchPlan
from ..services.relevance import RelevanceCandidate, RelevanceJudgment


class LlmError(RuntimeError):
    pass


class QueryPlanner(ABC):
    @abstractmethod
    async def plan(self, description: str) -> SearchPlan:
        raise NotImplementedError


class RelevanceReranker(ABC):
    @abstractmethod
    async def judge(self, description: str, candidates: list[RelevanceCandidate]) -> list[RelevanceJudgment]:
        raise NotImplementedError
