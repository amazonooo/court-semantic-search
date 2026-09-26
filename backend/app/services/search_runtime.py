"""Per-request budgets and safe telemetry; no global mutable search state."""
import asyncio
from contextvars import ContextVar
import logging
from time import monotonic
from uuid import uuid4

from ..config import Settings
from ..models import DocumentSearchParams, SearchDiagnostics, SearchEvent
from ..providers.base import CourtProvider

logger = logging.getLogger(__name__)
active_diagnostics: ContextVar[SearchDiagnostics | None] = ContextVar(
    "search_diagnostics", default=None
)


class SearchBudgetExceeded(TimeoutError):
    pass


class SearchRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.started = monotonic()
        self.deadline = self.started + settings.search_timeout_seconds
        self.phase = "retrieval"
        self.phase_deadline = self.deadline
        self.diagnostics = SearchDiagnostics(
            request_id=uuid4().hex,
            limits={
                "total_seconds": settings.search_timeout_seconds,
                "retrieval_seconds": settings.retrieval_timeout_seconds,
                "operation_seconds": settings.parser_api_timeout_seconds,
                "relevance_seconds": settings.relevance_timeout_seconds,
                "max_queries": settings.search_max_queries,
                "max_pages_per_query": settings.search_max_pages_per_query,
                "max_cases": settings.search_max_cases,
                "max_pdf_downloads": settings.search_max_pdf_downloads,
                "max_attempts_per_call": settings.parser_api_max_retries,
            },
        )

    def begin_retrieval(self):
        self.phase = "retrieval"
        self.phase_deadline = min(
            self.deadline, monotonic() + self.settings.retrieval_timeout_seconds
        )

    def begin_evidence(self, *, reserve_seconds: float = 0):
        self.phase = "evidence"
        # Preserve time for semantic reranking after PDF work, even if the
        # source consumes its full per-operation budget.
        self.phase_deadline = max(monotonic(), self.deadline - reserve_seconds)

    def begin_relevance(self):
        self.phase = "relevance"
        self.phase_deadline = self.deadline

    def remaining(self):
        return max(0.0, min(self.deadline, self.phase_deadline) - monotonic())

    def finish(self):
        self.diagnostics.total_seconds = round(monotonic() - self.started, 3)
        logger.info(
            "search_complete request_id=%s seconds=%.3f search_calls=%s pdf_calls=%s http_attempts=%s",
            self.diagnostics.request_id, self.diagnostics.total_seconds,
            self.diagnostics.search_calls, self.diagnostics.pdf_calls,
            self.diagnostics.http_attempts,
        )


class BoundedProvider(CourtProvider):
    def __init__(self, provider: CourtProvider, runtime: SearchRuntime):
        self.provider = provider
        self.runtime = runtime

    async def _call(self, operation, factory):
        remaining = self.runtime.remaining()
        if remaining <= 0:
            raise SearchBudgetExceeded("Search time budget exhausted")
        diagnostics = self.runtime.diagnostics
        if operation == "search":
            diagnostics.search_calls += 1
        else:
            if diagnostics.pdf_calls >= self.runtime.settings.search_max_pdf_downloads:
                raise SearchBudgetExceeded("PDF download budget exhausted")
            diagnostics.pdf_calls += 1
        started = monotonic()
        outcome = "ok"
        try:
            return await asyncio.wait_for(
                factory(), min(remaining, self.runtime.settings.parser_api_timeout_seconds)
            )
        except TimeoutError as exc:
            outcome = "timeout"
            raise SearchBudgetExceeded("Source operation exceeded its time budget") from exc
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:
            outcome = type(exc).__name__
            raise
        finally:
            elapsed = round(monotonic() - started, 3)
            diagnostics.events.append(SearchEvent(
                stage=self.runtime.phase, operation=operation,
                seconds=elapsed, outcome=outcome,
            ))
            diagnostics.stage_seconds[self.runtime.phase] = round(
                diagnostics.stage_seconds.get(self.runtime.phase, 0) + elapsed, 3
            )
            logger.info("search_operation request_id=%s operation=%s seconds=%.3f outcome=%s",
                        diagnostics.request_id, operation, elapsed, outcome)

    async def search_documents(self, params: DocumentSearchParams):
        return await self._call("search", lambda: self.provider.search_documents(params))

    async def download_pdf(self, file_url: str):
        data = await self._call("pdf_download", lambda: self.provider.download_pdf(file_url))
        if data:
            self.runtime.diagnostics.pdf_downloaded += 1
            self.runtime.diagnostics.pdf_bytes += len(data)
        return data
