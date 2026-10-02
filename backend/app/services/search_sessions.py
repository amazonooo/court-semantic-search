"""Bounded, resumable traversal of a provider's dated full-text result sets.

Tokens are opaque capabilities held in process memory. No search description,
credentials or PDF text is serialized into them. Restarting the service expires
them; a token is bound to its original provider, description, filters and plan.
"""
import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from hashlib import sha256
import json
from secrets import token_urlsafe
from time import monotonic

from ..models import CourtDocument, SearchPlan, QueryProgress, SearchRangeProgress
from ..providers.base import CourtProviderValidationError

SESSION_TTL = 3600
MAX_SESSIONS = 32
MAX_DOCUMENTS = 20000


@dataclass
class RangeTask:
    query: str
    progress: SearchRangeProgress
    page: int = 1
    received: set[str] = field(default_factory=set)
    page_size: int = 0


@dataclass
class SearchSession:
    fingerprint: str
    provider: object
    plan: SearchPlan
    token: str = field(default_factory=lambda: token_urlsafe(32))
    touched: float = field(default_factory=monotonic)
    pending: deque = field(default_factory=deque)
    documents: dict[str, CourtDocument] = field(default_factory=dict)
    matched: dict[str, set[str]] = field(default_factory=dict)
    checked: dict[str, str] = field(default_factory=dict)
    progress: dict[str, QueryProgress] = field(default_factory=dict)
    reference_loaded: bool = False
    reference_case_ids: set[str] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def retrieval_complete(self):
        return not self.pending and all(state.complete for state in self.progress.values())


_sessions: OrderedDict[str, SearchSession] = OrderedDict()


def fingerprint(request, plan):
    value = {'description': request.description, 'plan': plan.model_dump(mode='json'),
             'coverage_mode': request.coverage_mode, 'reference_case_number': request.reference_case_number,
             'semantic_reranking': getattr(request, 'semantic_reranking', False)}
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def get_session(request, plan, provider):
    now = monotonic()
    for token, state in list(_sessions.items()):
        if now - state.touched > SESSION_TTL and not state.lock.locked():
            del _sessions[token]
    if request.continuation_token:
        state = _sessions.get(request.continuation_token)
        if state is None:
            raise CourtProviderValidationError('Срок продолжения истёк или сервис перезапущен. Начните новый поиск.')
        if state.fingerprint != fingerprint(request, plan) or state.provider is not provider:
            raise CourtProviderValidationError('Продолжение относится к другому описанию, плану или источнику.')
        if state.lock.locked():
            raise CourtProviderValidationError('Этот поиск уже продолжается. Дождитесь результата.')
        state.touched = now
        _sessions.move_to_end(state.token)
        return state
    for token, state in list(_sessions.items()):
        if len(_sessions) < MAX_SESSIONS:
            break
        if not state.lock.locked():
            del _sessions[token]
    if len(_sessions) >= MAX_SESSIONS:
        raise CourtProviderValidationError('Все места продолжения заняты. Повторите позже.')
    state = SearchSession(fingerprint(request, plan), provider, plan.model_copy(deep=True))
    # The explicit traversal period is shown to users. It is never labelled
    # "the entire database": provider coverage and text semantics are external.
    lower = plan.filters.date_from or date(1900, 1, 1)
    upper = plan.filters.date_to or date.today()
    for query in plan.queries:
        progress = QueryProgress(query=query)
        period = SearchRangeProgress(date_from=lower, date_to=upper)
        progress.ranges.append(period)
        state.progress[query] = progress
        state.pending.append(RangeTask(query, period))
    _sessions[state.token] = state
    return state


def split_task(state, task):
    lower, upper = task.progress.date_from, task.progress.date_to
    middle = lower + (upper - lower) // 2
    task.progress.status = 'split'
    # Older half first: deep traversal must not inherit a newest-first bias.
    for start, end in ((lower, middle), (middle + timedelta(days=1), upper)):
        period = SearchRangeProgress(date_from=start, date_to=end)
        state.progress[task.query].ranges.append(period)
        state.pending.append(RangeTask(task.query, period))
