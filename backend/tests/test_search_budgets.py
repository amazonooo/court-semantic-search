import asyncio
from datetime import date
from time import monotonic

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings, get_settings
from backend.app.llm.factory import get_query_planner, UnavailablePlanner
from backend.app.main import app
from backend.app.models import DocumentSearchResult, EvidenceSearchRequest, SearchPlan
from backend.app.providers.base import CourtProviderAccessError
from backend.app.providers.factory import get_court_provider
from backend.app.providers.parser_api import ParserApiProvider
from backend.app.services.evidence import find_evidence
from backend.app.services.semantic_search import SemanticSearchService
from backend.app.services.search_runtime import active_diagnostics
from backend.tests.test_semantic_search import FakeProvider


def settings(**kw):
    return Settings(_env_file=None, **kw)


def request(**kw):
    return EvidenceSearchRequest(
        description='Найти судебные дела по указанным обстоятельствам',
        plan=SearchPlan(queries=['налоговые расходы', 'расходы по займу'], must_have=['налог']),
        **kw,
    )


@pytest.mark.asyncio
async def test_stalled_retrieval_returns_partial_and_cancels_network_work():
    class Slow(FakeProvider):
        stopped = False
        async def search_documents(self, params):
            try:
                await asyncio.sleep(5)
            finally:
                self.stopped = True
    source = Slow()
    start = monotonic()
    result = await SemanticSearchService(source, None, settings(
        search_timeout_seconds=.1, retrieval_timeout_seconds=.04,
        parser_api_timeout_seconds=.1,
    )).search_with_evidence(request())
    assert monotonic() - start < .5
    assert source.stopped
    assert result.partial and result.warnings
    assert result.case_count == result.cases_checked == 0
    assert result.diagnostics.search_calls == 1
    assert result.diagnostics.pdf_calls == 0
    assert result.diagnostics.total_seconds > 0
    assert active_diagnostics.get() is None


@pytest.mark.asyncio
async def test_pdf_timeout_is_not_counted_as_checked():
    class SlowPdf(FakeProvider):
        stopped = False
        async def download_pdf(self, file_url):
            try:
                await asyncio.sleep(5)
            finally:
                self.stopped = True
    source = SlowPdf()
    result = await SemanticSearchService(source, None, settings(
        search_timeout_seconds=.15, parser_api_timeout_seconds=.03,
    )).search_with_evidence(request())
    assert source.stopped
    assert result.cases_attempted == 1
    assert result.cases_checked == result.diagnostics.pdf_downloaded == 0
    assert result.items[0].verification_status == 'unverified'
    assert result.items[0].coverage is None
    assert result.partial


@pytest.mark.asyncio
async def test_server_caps_client_work_and_keeps_all_user_plan_terms():
    class Many(FakeProvider):
        async def search_documents(self, params):
            self.calls.append(params)
            return DocumentSearchResult(count=100, pages=20, page=params.page,
                items=[self.document.model_copy(update={
                    'document_id': f'doc-{i}', 'case_id': f'case-{i}',
                    'file_url': f'https://example.org/{i}.pdf'}) for i in range(10)])
    source = Many()
    query = request(max_pages_per_query=5, max_cases=20)
    query.plan = SearchPlan(queries=['a query', 'b query', 'c query', 'd query', 'e query'],
                           exclude=['явное исключение'])
    before = query.plan.model_dump()
    result = await SemanticSearchService(source, None, settings()).search_with_evidence(query)
    assert len(source.calls) == 3
    assert all(p.page == 1 for p in source.calls)
    assert result.diagnostics.pdf_calls == result.cases_attempted == 6
    assert result.partial
    assert query.plan.model_dump() == before == result.plan.model_dump()


@pytest.mark.asyncio
async def test_semantic_flow_applies_explicit_dates_and_rechecks_source_dates():
    class DateSource(FakeProvider):
        async def search_documents(self, params):
            assert params.date_from == date(2022, 1, 1)
            assert params.date_to == date(2024, 12, 31)
            assert params.inn == '7700000000'
            return DocumentSearchResult(count=3, pages=1, page=1, items=[
                self.document,
                self.document.model_copy(update={'document_id': 'old', 'registration_date': date(2020, 1, 1)}),
                self.document.model_copy(update={'document_id': 'unknown', 'registration_date': None}),
            ])
    q = request(filters={'dateFrom': '2022-01-01', 'dateTo': '2024-12-31', 'inn': '7700000000'})
    result = await SemanticSearchService(DateSource(), None, settings()).search_with_evidence(q)
    assert result.document_count == 1
    assert result.diagnostics.filtered_by_date == 4  # two invalid source rows in each query


@pytest.mark.parametrize(('text', 'term', 'found'), [
    ('Налоговым органом исключены расходы.', 'налоговый орган', True),
    ('Налоговый орган проверил расчёт.', 'налоговый орган', True),
    ('Заказчик требует неустойку за просрочку.', 'неустойка', True),
    ('Кредитор включен в реестр требований.', 'реестр требований', True),
    ('Оспаривание решения собрания участников.', 'собрание участников', True),
    ('Применяется статья 265 НК РФ.', 'статья 265 НК РФ', True),
    ('Применяется статья 266 НК РФ.', 'статья 265 НК РФ', False),
    ('Конкурс на выполнение работ.', 'конкурсное производство', False),
    ('Проводился международный конгресс.', 'конкурсный', False),
    ('Налоговый ' + 'вопрос ' * 80 + 'орган', 'налоговый орган', False),
    ('Налогообложение и организация.', 'налоговый орган', False),
])
def test_generic_lexical_evidence_has_word_boundaries_proximity_and_exact_quotes(text, term, found):
    matches = find_evidence(text, [term])
    assert bool(matches) is found
    for match in matches:
        assert match.quote in text
        assert match.term == term


@pytest.mark.asyncio
async def test_auth_error_is_not_retried_or_disguised_as_no_results():
    calls = 0
    def handler(req):
        nonlocal calls
        calls += 1
        return httpx.Response(403, json={'error':'Subscription expired', 'error_code':40302})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        source = ParserApiProvider(api_key='secret-test', base_url='https://parser-api.com/test', client=client, max_retries=3)
        with pytest.raises(CourtProviderAccessError) as exc:
            await SemanticSearchService(source, None, settings()).search_with_evidence(request())
    assert exc.value.error_code == 40302
    assert calls == 1


def test_route_preserves_provider_error_code_without_requiring_llm_credentials():
    class Denied(FakeProvider):
        async def search_documents(self, params):
            raise CourtProviderAccessError('Access denied', error_code=40301)
    app.dependency_overrides[get_court_provider] = Denied
    app.dependency_overrides[get_query_planner] = lambda: UnavailablePlanner('No LLM key')
    app.dependency_overrides[get_settings] = lambda: settings()
    try:
        with TestClient(app) as client:
            response = client.post('/api/cases/search-with-evidence', json=request().model_dump(mode='json'))
        assert response.status_code == 503
        assert response.json()['error_code'] == 40301
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_concurrent_searches_have_independent_diagnostics():
    async def search_one():
        return await SemanticSearchService(FakeProvider(), None, settings()).search_with_evidence(request())
    a, b = await asyncio.gather(search_one(), search_one())
    assert a.diagnostics.request_id != b.diagnostics.request_id
    assert a.diagnostics.search_calls == b.diagnostics.search_calls == 2
    assert a.diagnostics.pdf_calls == b.diagnostics.pdf_calls == 1
    assert active_diagnostics.get() is None


@pytest.mark.asyncio
async def test_evidence_uses_first_instance_and_preserves_exact_document(monkeypatch):
    from types import SimpleNamespace
    from backend.app.services.case_text import extract_case_document_text
    class History(FakeProvider):
        def __init__(self):
            super().__init__()
            self.downloaded = []
        async def search_documents(self, params):
            first = self.document.model_copy(update={'instance_level': 1})
            appeal = self.document.model_copy(update={
                'document_id': 'appeal', 'instance_level': 2,
                'document_type': 'Постановление', 'file_url': 'https://example.org/appeal.pdf'})
            return DocumentSearchResult(count=2, pages=1, page=1, items=[first, appeal])
        async def download_pdf(self, url):
            self.downloaded.append(url)
            return b'%PDF-test'
    async def text(_):
        return 'Налоговый орган проверил налог на прибыль.'
    monkeypatch.setattr('backend.app.services.case_text.extract_pdf_text_async', text)
    source = History()
    result = await SemanticSearchService(source, None, settings()).search_with_evidence(request())
    item = result.items[0]
    assert item.evidence_document.document_id == 'doc-1'
    assert item.case.preferred_document.document_id == 'appeal'
    assert source.downloaded == [item.evidence_document.file_url]
    assert item.evidence[0].quote in await text(None)
    assert result.diagnostics.pdf_downloaded == result.cases_checked == 1
    assert 'pdf_extraction' in result.diagnostics.stage_seconds


@pytest.mark.asyncio
async def test_pdf_worker_is_terminated_when_request_is_cancelled(monkeypatch):
    from backend.app.services.pdf import extract_pdf_text_async
    class Worker:
        returncode = None
        killed = False
        async def communicate(self, data):
            await asyncio.sleep(5)
        def kill(self):
            self.killed = True
            self.returncode = -9
        async def wait(self):
            return self.returncode
    worker = Worker()
    async def spawn(*args, **kwargs):
        return worker
    monkeypatch.setattr('backend.app.services.pdf.asyncio.create_subprocess_exec', spawn)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(extract_pdf_text_async(b'%PDF-test'), .03)
    assert worker.killed
