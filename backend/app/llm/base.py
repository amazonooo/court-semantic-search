from abc import ABC, abstractmethod

from ..models import SearchPlan


class LlmError(RuntimeError):
    pass


class QueryPlanner(ABC):
    @abstractmethod
    async def plan(self, description: str) -> SearchPlan:
        raise NotImplementedError
