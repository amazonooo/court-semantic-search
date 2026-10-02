import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.llm.base import LlmError, RelevanceReranker
from backend.app.llm.gigachat import GigaChatRelevanceReranker
from backend.app.llm.factory import get_query_planner
from backend.app.main import app
from backend.app.models import EvidenceSearchRequest, SearchPlan
from backend.app.providers.factory import get_court_provider
from backend.app.services.relevance import (
    RelevanceCandidate, RelevanceJudgment, CriterionJudgment, SourceCitation,
    full_document_passages, lexical_query_overlap,
)
from backend.app.services.semantic_search import SemanticSearchService
from backend.tests.test_semantic_search import FakeProvider, FakePlanner

DESCRIPTION = 'Покупатель приобрёл компанию, затем присоединился к приобретённой компании.'
PLAN = SearchPlan(queries=['приобретение компании', 'присоединение покупателя'],
                  must_have=['приобретение компании', 'присоединение к приобретенной компании'])
PURCHASE = 'ООО Альфа приобрело доли ООО Бета.'
MERGER = 'ООО Альфа прекратило деятельность путём присоединения к ООО Бета.'


@pytest.fixture
def api():
    provider = FakeProvider()
    app.dependency_overrides[get_court_provider] = lambda: provider
    app.dependency_overrides[get_query_planner] = FakePlanner
    with TestClient(app) as client:
        yield client, provider
    app.dependency_overrides.clear()


@pytest.mark.parametrize('route', ['/api/cases/semantic-search', '/api/cases/search-with-evidence'])
def test_unapproved_plan_never_searches_the_source(api, route):
    client, provider = api
    response = client.post(route, json={'description': DESCRIPTION, 'plan': PLAN.model_dump()})
    assert response.status_code == 409
    assert not provider.calls


def test_edit_requires_new_approval_and_approved_text_reaches_provider(api):
    client, provider = api
    original = PLAN.model_dump(mode='json')
    approval = {'queries': original['queries'], 'must_have': original['must_have']}
    edited = {**original, 'queries': ['покупка долей', original['queries'][1]]}
    payload = {'description': DESCRIPTION, 'plan': edited, 'plan_approval': approval}
    stale = client.post('/api/cases/semantic-search', json=payload)
    assert stale.status_code == 422 and not provider.calls
    payload['plan_approval'] = {key: edited[key] for key in ('queries', 'must_have')}
    result = client.post('/api/cases/semantic-search', json=payload)
    assert result.status_code == 200
    assert provider.calls == [('покупка долей', 1), ('присоединение покупателя', 1)]
    assert result.json()['plan']['queries'] == edited['queries']


def test_lexical_selection_uses_query_words_across_complete_text():
    words, total = lexical_query_overlap(
        'При приобретении компании. ' + 'Текст решения. ' * 4000 + 'Присоединением покупателя.',
        PLAN.queries,
    )
    assert set(words) == {'приобретение', 'компании', 'присоединение', 'покупателя'}
    assert len(words) == total == 4


def test_transport_covers_every_character_without_normalizing_quotes():
    text = '\n\n'.join(f'Страница {i}: ' + 'Обстоятельства. ' * 800 for i in range(1, 21))
    passages = full_document_passages(text)
    for marker in [f'Страница {i}:' for i in range(1, 21)]:
        assert any(marker in part for part in passages.values())
    assert next(iter(passages.values())).startswith('Страница 1:')
    assert list(passages.values())[-1].endswith('Обстоятельства. ')
    assert all(part in text for part in passages.values())


class FullDocumentAPI:
    def __init__(self):
        self.read = []
        self.final = []

    async def complete(self, **kwargs):
        data = json.loads(kwargs['user'])
        if 'facts' in kwargs['schema']['properties']:
            self.read.extend(data['passages'].values())
            facts = []
            for source, text in data['passages'].items():
                for quote in [PURCHASE, MERGER]:
                    if quote in text:
                        facts.append({'statement': quote, 'entities': 'ООО Альфа; ООО Бета',
                                      'passage_id': source, 'quote': quote})
            return json.dumps({'facts': facts})
        self.final.append(data)
        case = data['cases'][0]
        citations = []
        for quote in [PURCHASE, MERGER]:
            source = next(key for key, text in case['passages'].items() if quote in text)
            citations.append({'passage_id': source, 'quote': quote})
        assert citations[0]['passage_id'] != citations[1]['passage_id']
        return json.dumps({'items': [{'key': case['key'], 'score': 5,
            'reason': 'ООО Бета — приобретённая компания и правопреемник ООО Альфа.',
            'passage_id': citations[0]['passage_id'], 'criteria': {
                'C1': {'status': 'supported', **citations[0], 'citations': citations[:1], 'reason': 'Компания приобретена.'},
                'C2': {'status': 'supported', **citations[1], 'citations': citations,
                       'reason': 'В обоих эпизодах приобретённая компания — ООО Бета.'},
            }}]})


@pytest.mark.asyncio
async def test_long_act_joins_cross_page_facts_and_validates_both_citations(monkeypatch):
    # Two related facts far beyond the old 5 x 1000 character sample, including
    # late pages and all intervening pages, are passed through the real adapter.
    text = '\n'.join(f'Страница {i}. ' + (PURCHASE if i == 3 else MERGER if i == 20 else '')
                     + 'Иные обстоятельства решения. ' * 150 for i in range(1, 21))
    async def extract(case, provider, *, role):
        return SimpleNamespace(document=case.documents[0], text=text)
    monkeypatch.setattr('backend.app.services.semantic_search.extract_case_document_text', extract)
    provider = FakeProvider()
    api = FullDocumentAPI()
    result = await SemanticSearchService(provider, None,
        Settings(_env_file=None), reranker=GigaChatRelevanceReranker(api)).search_with_evidence(
            EvidenceSearchRequest(description=DESCRIPTION, plan=PLAN, semantic_reranking=True))
    item = result.items[0]
    assert item.analysis_complete and item.analysis_scope == 'full_text'
    assert result.confirmed_count == 1
    assert len(item.semantic_criteria[1].citations) == 2
    assert {citation.quote[citation.highlights[0].start:citation.highlights[0].end]
            for citation in item.semantic_criteria[1].citations} == {PURCHASE, MERGER}
    read = '\n'.join(api.read)
    assert all(f'Страница {i}.' in read for i in range(1, 21))
    assert len(api.final) == 1


@pytest.mark.asyncio
async def test_one_fabricated_supporting_citation_invalidates_composite_criterion():
    class API:
        async def complete(self, **kwargs):
            return json.dumps({'items': [{'key': '0', 'score': 5, 'reason': 'Связь', 'passage_id': 'P1',
                'criteria': {'C1': {'status': 'supported', 'passage_id': 'P1', 'quote': PURCHASE,
                    'reason': 'Связь компаний', 'citations': [
                        {'passage_id': 'P1', 'quote': PURCHASE},
                        {'passage_id': 'P2', 'quote': 'ООО Альфа присоединилось к ООО Бета.'}]}}}]})
    rows = await GigaChatRelevanceReranker(API()).judge(DESCRIPTION, [
        RelevanceCandidate('0', 'А1', {'P1': PURCHASE, 'P2': 'Присоединилось ООО Гамма.'},
                           ('присоединение к приобретенной компании',), True)])
    assert rows[0].criteria[0].status == 'unclear'
    assert rows[0].criteria[0].citations == ()


@pytest.mark.asyncio
async def test_partial_long_act_read_never_produces_final_verdict():
    class Failing(FullDocumentAPI):
        async def complete(self, **kwargs):
            if self.read:
                raise LlmError('Недоступна следующая часть')
            return await super().complete(**kwargs)
    api = Failing()
    text = PURCHASE + ' Продолжение. ' * 6000 + MERGER
    rows = await GigaChatRelevanceReranker(api).judge(DESCRIPTION, [
        RelevanceCandidate('0', 'А1', full_document_passages(text), tuple(PLAN.must_have), True)])
    assert not rows and api.read and not api.final


@pytest.mark.asyncio
async def test_budget_preserves_completed_act_when_another_times_out():
    class Reranker:
        async def _judge(self, description, candidates):
            candidate = candidates[0]
            if candidate.key == 'slow':
                await asyncio.sleep(1)
            return [RelevanceJudgment(candidate.key, 3, 'Оценка', 'P1')]
    from backend.app.llm.full_document import judge_full_documents
    rows = await judge_full_documents(Reranker(), DESCRIPTION, [
        RelevanceCandidate('fast', 'А1', {'P1': PURCHASE}, (), True),
        RelevanceCandidate('slow', 'А2', {'P1': MERGER}, (), True),
    ], timeout_seconds=.15)
    assert [row.key for row in rows] == ['fast']


@pytest.mark.asyncio
async def test_incomplete_semantic_analysis_remains_available_for_continuation(monkeypatch):
    async def extract(case, provider, *, role):
        return SimpleNamespace(document=case.documents[0], text=PURCHASE + ' ' + MERGER)
    class EmptyJudge(RelevanceReranker):
        calls = 0
        async def judge(self, description, candidates):
            self.calls += 1
            return []
    monkeypatch.setattr('backend.app.services.semantic_search.extract_case_document_text', extract)
    provider, judge = FakeProvider(), EmptyJudge()
    request = EvidenceSearchRequest(description=DESCRIPTION, plan=PLAN,
        coverage_mode='full', max_search_calls=10, semantic_reranking=True)
    first = await SemanticSearchService(provider, None, Settings(_env_file=None), reranker=judge).search_with_evidence(request)
    assert first.retrieval_complete and not first.verification_complete and first.continuation_token
    assert not first.items[0].analysis_complete and first.confirmed_count == 0
    second = await SemanticSearchService(provider, None, Settings(_env_file=None), reranker=judge).search_with_evidence(
        request.model_copy(update={'continuation_token': first.continuation_token}))
    assert judge.calls == 2 and second.cases_attempted == 1
    assert second.diagnostics.previously_checked_cases_skipped == 0


@pytest.mark.asyncio
async def test_semantic_analysis_receives_acts_ordered_by_query_words(monkeypatch):
    from backend.app.models import DocumentSearchResult
    provider = FakeProvider()
    weak = provider.document.model_copy(update={'case_id': 'weak', 'case_number': 'А1', 'document_id': 'weak-doc'})
    strong = provider.document.model_copy(update={'case_id': 'strong', 'case_number': 'А2', 'document_id': 'strong-doc'})
    class Source(FakeProvider):
        async def search_documents(self, params):
            return DocumentSearchResult(count=2, pages=1, page=1, items=[weak, strong])
    async def extract(case, provider, *, role):
        text = 'Компания предъявила иск.' if case.case_id == 'weak' else 'Приобретение компании и присоединение покупателя.'
        return SimpleNamespace(document=case.documents[0], text=text)
    class Judge(RelevanceReranker):
        order = []
        async def judge(self, description, candidates):
            self.order = [item.case_number for item in candidates]
            assert all(item.full_text for item in candidates)
            return []
    judge = Judge()
    monkeypatch.setattr('backend.app.services.semantic_search.extract_case_document_text', extract)
    result = await SemanticSearchService(Source(), None, Settings(_env_file=None), reranker=judge).search_with_evidence(
        EvidenceSearchRequest(description=DESCRIPTION, plan=PLAN, semantic_reranking=True))
    assert judge.order == ['А2', 'А1']
    by_id = {item.case.case_id: item for item in result.items}
    assert by_id['strong'].lexical_word_count == 4 > by_id['weak'].lexical_word_count
