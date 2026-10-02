"""Provider simulations: no paid APIs and no invented real court PDF text."""
from datetime import date
import asyncio
from math import ceil
from types import SimpleNamespace

import pytest

from backend.app.config import Settings
from backend.app.llm.base import LlmError
from backend.app.llm.gigachat import GigaChatQueryPlanner
from backend.app.llm.prompt import parse_search_plan
from backend.app.models import DocumentSearchResult, EvidenceSearchRequest, SearchPlan, SemanticSearchRequest
from backend.app.providers.base import CourtProviderTemporaryError, CourtProviderValidationError
from backend.app.services.case_text import extract_case_document_text
from backend.app.services.cases import CaseAggregationService
from backend.app.services.evidence import find_evidence, term_alternatives
from backend.app.services.semantic_search import SemanticSearchService
from backend.app.services.case_text import document_for_role
from backend.app.services.search_sessions import get_session
from backend.tests.test_semantic_search import FakeProvider
import json


def settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


def request(**kwargs):
    return EvidenceSearchRequest(description='Налоговый орган оспаривает расходы на выкуп имущества',
        plan=SearchPlan(queries=['аренда право выкупа', 'выкупная стоимость имущества'],
                        must_have=['выкуп ранее арендовавшегося имущества']), **kwargs)


class DatedSource(FakeProvider):
    def __init__(self):
        super().__init__()
        self.rows = [self.document.model_copy(update={
            'document_id': f'doc-{year}', 'case_id': f'case-{year}',
            'case_number': f'А40-{year}/2010', 'registration_date': date(year, 6, 1),
            'file_url': f'https://example.org/{year}.pdf', 'instance_level': 1,
        }) for year in range(2010, 2026)]
        self.rows[-2] = self.rows[-2].model_copy(update={'case_number': 'А68-10855/2018'})
    async def search_documents(self, params):
        self.calls.append(params)
        rows = [d for d in self.rows if
                (not params.date_from or d.registration_date >= params.date_from) and
                (not params.date_to or d.registration_date <= params.date_to)]
        rows.sort(key=lambda d: d.registration_date, reverse=True)
        # Simulate a source page cap even when count exceeds exposed pages.
        return DocumentSearchResult(count=len(rows), pages=min(2, ceil(len(rows) / 2)),
            page=params.page, items=rows[(params.page - 1) * 2:params.page * 2])


@pytest.mark.asyncio
async def test_full_traversal_reaches_old_cases_and_all_rows_without_restarting_pages():
    source = DatedSource()
    service = SemanticSearchService(source, None, settings())
    q = request(coverage_mode='full', filters={'dateFrom': '2010-01-01', 'dateTo': '2025-12-31'},
                max_search_calls=2, max_pages_per_query=2)
    seen = set()
    for _ in range(60):
        result = await service.search(q)
        seen.update(item.case.case_number for item in result.items)
        assert result.diagnostics.search_calls <= 2
        if result.retrieval_complete:
            break
        q.continuation_token = result.continuation_token
    else:
        pytest.fail('Traversal never completed')
    assert len(seen) == 16 and 'А68-10855/2018' in seen
    assert all(progress.complete for progress in result.diagnostics.query_progress)
    calls = [(p.text, p.date_from, p.date_to, p.page) for p in source.calls]
    assert len(calls) == len(set(calls))
    assert all(p.date_from >= date(2010, 1, 1) and p.date_to <= date(2025, 12, 31) for p in source.calls)


@pytest.mark.asyncio
async def test_failed_range_is_retained_for_retry_and_never_reported_empty():
    class Failing(FakeProvider):
        async def search_documents(self, params):
            self.calls.append(params)
            if len(self.calls) == 1:
                raise CourtProviderTemporaryError('unstable')
            return DocumentSearchResult(count=1, pages=1, page=1, items=[self.document])
    source = Failing()
    service = SemanticSearchService(source, None, settings())
    q = request(coverage_mode='full', max_search_calls=1)
    first = await service.search(q)
    assert first.diagnostics.query_progress[0].error and not first.retrieval_complete
    q.continuation_token = first.continuation_token
    for _ in range(3):
        result = await service.search(q)
        if result.retrieval_complete:
            break
    assert result.retrieval_complete


@pytest.mark.asyncio
async def test_interrupted_page_is_resumed_without_losing_its_range():
    class Interrupted(FakeProvider):
        async def search_documents(self, params):
            self.calls.append(params)
            if len(self.calls) == 1:
                raise asyncio.CancelledError()
            return DocumentSearchResult(count=1, pages=1, page=1, items=[self.document])
    source = Interrupted()
    q = request(coverage_mode='full', max_search_calls=1)
    session = get_session(q, q.plan, source)
    q.continuation_token = session.token
    service = SemanticSearchService(source, None, settings())
    with pytest.raises(asyncio.CancelledError):
        await service.search(q)
    assert not session.lock.locked() and len(session.pending) == 2
    result = await service.search(q)
    assert source.calls[0] == source.calls[1]
    assert result.document_count == 1 and not result.retrieval_complete
    result = await service.search(q)
    assert result.retrieval_complete


@pytest.mark.asyncio
async def test_single_day_capped_source_is_explicitly_incomplete():
    class Capped(FakeProvider):
        async def search_documents(self, params):
            return DocumentSearchResult(count=5000, pages=1, page=1, items=[self.document])
    result = await SemanticSearchService(Capped(), None, settings()).search(request(
        coverage_mode='full', filters={'dateFrom': '2023-01-01', 'dateTo': '2023-01-01'}))
    assert not result.retrieval_complete
    assert all(p.ranges[0].status == 'blocked' for p in result.diagnostics.query_progress)
    assert any('ограничил выдачу' in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_token_cannot_resume_a_changed_description_or_plan():
    source = DatedSource()
    service = SemanticSearchService(source, None, settings())
    q = request(coverage_mode='full', max_search_calls=1)
    result = await service.search(q)
    q.continuation_token = result.continuation_token
    q.description = 'Другой запрос о расходах на строительство объекта'
    with pytest.raises(CourtProviderValidationError, match='другому описанию'):
        await service.search(q)


@pytest.mark.asyncio
async def test_rejected_cases_do_not_consume_pdf_calls_or_expand_history():
    source = FakeProvider()
    result = await SemanticSearchService(source, None, settings()).search_with_evidence(
        request(excluded_case_ids=['case-1'], max_case_expansions=2))
    assert result.case_count == 1 and result.items == []
    assert result.diagnostics.feedback_cases_skipped == 1
    assert result.diagnostics.pdf_calls == result.diagnostics.cases_expanded == 0


@pytest.mark.asyncio
async def test_continuation_checks_new_candidates_without_rechecking_the_same_pdf(monkeypatch):
    class Source(FakeProvider):
        async def search_documents(self, params):
            rows = [self.document.model_copy(update={'document_id': f'doc-{i}', 'case_id': f'case-{i}',
                'case_number': f'А40-{i}/2023', 'file_url': f'https://example.org/{i}.pdf'}) for i in range(3)]
            return DocumentSearchResult(count=3, pages=1, page=1, items=rows)
    read = []
    async def extract(case, provider, *, role):
        document = case.documents[0]
        read.append(document.document_id)
        return SimpleNamespace(document=document, text='Суд рассмотрел обстоятельства выкупа ранее арендовавшегося имущества.')
    monkeypatch.setattr('backend.app.services.semantic_search.extract_case_document_text', extract)
    source = Source()
    service = SemanticSearchService(source, None, settings())
    q = request(coverage_mode='full', max_cases=1)
    for index in range(3):
        result = await service.search_with_evidence(q)
        assert result.cases_checked == 1
        assert result.diagnostics.previously_checked_cases_skipped == index
        q.continuation_token = result.continuation_token
    assert len(read) == len(set(read)) == 3
    assert result.retrieval_complete and result.verification_complete and not result.continuation_token


@pytest.mark.asyncio
async def test_expanded_merits_act_is_retained_across_pdf_batches(monkeypatch):
    class Source(FakeProvider):
        async def search_documents(self, params):
            self.calls.append(params)
            if params.case_number:
                rows = [self.document.model_copy(update={'document_id': 'merits-1',
                    'instance_level': 1, 'file_url': 'https://example.org/merits.pdf'})]
            else:
                rows = [self.document.model_copy(update={'document_type': 'Определение',
                            'content_types': ['Об отложении судебного разбирательства']}),
                        self.document.model_copy(update={'document_id': 'doc-2',
                            'case_id': 'case-2', 'case_number': 'А40-2/2023'})]
            return DocumentSearchResult(count=len(rows), pages=1, page=1, items=rows)
    read = []
    async def extract(case, provider, *, role):
        document = document_for_role(case, role)
        read.append(document.document_id)
        return SimpleNamespace(document=document, text='Синтетический текст: выкуп ранее арендовавшегося имущества.')
    monkeypatch.setattr('backend.app.services.semantic_search.extract_case_document_text', extract)
    source = Source()
    service = SemanticSearchService(source, None, settings())
    q = request(coverage_mode='full', max_cases=1, max_case_expansions=2)
    first = await service.search_with_evidence(q)
    assert first.diagnostics.cases_expanded == 1
    q.continuation_token = first.continuation_token
    second = await service.search_with_evidence(q)
    assert second.diagnostics.previously_checked_cases_skipped == 1
    assert second.diagnostics.search_calls == 0 and not second.continuation_token
    assert set(read) == {'merits-1', 'doc-2'} and len(read) == 2


@pytest.mark.asyncio
async def test_successful_text_is_cached_but_failed_downloads_are_not(monkeypatch):
    class Source(FakeProvider):
        downloads = 0
        async def download_pdf(self, url):
            self.downloads += 1
            return b'pdf'
    async def extract(data):
        return 'Синтетический текст судебного акта'
    monkeypatch.setattr('backend.app.services.case_text._extract_with_timing', extract)
    source = Source()
    case = CaseAggregationService(source)._build_case([source.document])
    for _ in range(2):
        await extract_case_document_text(case, source, role='search_evidence')
    assert source.downloads == 1
    unavailable = FakeProvider()
    with pytest.raises(Exception, match='not found'):
        await extract_case_document_text(case, unavailable, role='search_evidence')


@pytest.mark.parametrize('term', ['обязательство/убытки вследствие займа', 'переход долга/убытков по займу'])
def test_slash_alternative_keeps_common_qualifiers(term):
    assert not find_evidence('Суд установил исполнение обязательства. Также имеются убытки.', [term])
    assert all('займ' in alternative or 'займа' in alternative for alternative in term_alternatives(term))


def test_loan_alternative_can_match_credit_without_loan():
    assert find_evidence('Компания получила кредит на покупку компании.',
                         ['Компания получила заем/кредит на покупку компании'])


def test_tax_subject_omission_is_corrected_before_source_calls():
    plan = parse_search_plan(json.dumps({'queries': ['аренда право выкупа', 'выкупная стоимость'],
                                         'must_have': ['выкуп имущества']}), request().description)
    assert plan.must_have[0] == 'Налоговый предмет спора по описанной операции'


def test_reversed_borrower_role_is_rejected():
    with pytest.raises(LlmError, match='перепутала'):
        parse_search_plan(json.dumps({'queries': ['компания купила компанию-заемщика', 'передача долга'],
            'must_have': ['присоединение к приобретенной компании']}),
            'Компания-покупатель привлекла заем на покупку третьей компании и присоединилась к ней')


@pytest.mark.asyncio
async def test_planner_repairs_reversed_borrower_once_before_search():
    class Model:
        calls = []
        async def complete(self, **kwargs):
            self.calls.append(kwargs)
            query = ('компания купила компанию-заемщика' if len(self.calls) == 1
                     else 'покупка компании кредит')
            return json.dumps({'queries': [query, 'передача долга'],
                'must_have': ['Покупатель присоединился к приобретенной компании']})
    model = Model()
    plan = await GigaChatQueryPlanner(model).plan(
        'Компания-покупатель привлекла заем на покупку третьей компании и присоединилась к ней')
    assert len(model.calls) == 2
    assert 'перепутала' in model.calls[-1]['system']
    assert plan.queries[0] == 'покупка компании кредит'


@pytest.mark.asyncio
async def test_reference_case_uses_number_only_and_is_labelled_as_calibration(monkeypatch):
    class Source(FakeProvider):
        async def search_documents(self, params):
            self.calls.append(params)
            rows = [self.document.model_copy(update={'case_number': 'А68-10855/2018'})] if params.case_number else []
            return DocumentSearchResult(count=len(rows), pages=int(bool(rows)), page=1, items=rows)
    async def extract(case, provider, *, role):
        return SimpleNamespace(document=case.documents[0], text='Синтетический текст: выкуп ранее арендовавшегося имущества.')
    monkeypatch.setattr('backend.app.services.semantic_search.extract_case_document_text', extract)
    source = Source()
    result = await SemanticSearchService(source, None, settings()).search_with_evidence(
        request(reference_case_number='А68-10855/2018', max_search_calls=3))
    assert result.diagnostics.search_calls == 3
    assert source.calls[-1].case_number == 'А68-10855/2018' and source.calls[-1].text is None
    assert result.items[0].is_reference and result.items[0].matched_queries == []
    assert result.confirmed_count == 0


@pytest.mark.asyncio
async def test_bankruptcy_metadata_gate_is_scoped_to_tax_subject_and_preserves_other_searches():
    source = FakeProvider()
    source.document = source.document.model_copy(update={'document_type': 'Определение',
        'content_types': ['О включении требований в реестр требований кредиторов']})
    tax = await SemanticSearchService(source, None, settings()).search_with_evidence(request())
    assert tax.diagnostics.wrong_subject_cases_skipped == 1 and tax.diagnostics.pdf_calls == 0
    bankruptcy = await SemanticSearchService(source, None, settings()).search(request().model_copy(
        update={'description': 'Найти дела о банкротстве и включении требований ФНС в реестр'}))
    assert len(bankruptcy.items) == 1
