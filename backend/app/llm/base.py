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
    async def judge_with_budget(self, description: str, candidates: list[RelevanceCandidate],
                                timeout_seconds: float) -> list[RelevanceJudgment]:
        return await self.judge(description, candidates)

    @abstractmethod
    async def judge(self, description: str, candidates: list[RelevanceCandidate]) -> list[RelevanceJudgment]:
        raise NotImplementedError
