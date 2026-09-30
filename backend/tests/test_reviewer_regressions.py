"""Synthetic scenarios derived from the reviewers' errors, not real PDF contents."""
import json
from types import SimpleNamespace

import pytest

from backend.app.config import Settings
from backend.app.llm.base import LlmError, RelevanceReranker
from backend.app.llm.gigachat import GigaChatRelevanceReranker
from backend.app.llm.prompt import parse_search_plan
from backend.app.models import (
    CriterionAssessment, DocumentSearchResult, EvidenceCase, EvidenceSearchRequest,
    SearchPlan, SemanticSearchRequest, TextHighlight,
)
from backend.app.services.case_text import document_for_role
from backend.app.services.cases import CaseAggregationService
from backend.app.services.evidence import find_evidence
from backend.app.services.recommendations import assess_recommendation
from backend.app.services.relevance import (
    CriterionJudgment, RelevanceCandidate, RelevanceJudgment, model_passages,
)
from backend.app.services.semantic_search import SemanticSearchService, _select_queries
from backend.tests.test_semantic_search import FakeProvider


def settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


def request(**kwargs):
    return EvidenceSearchRequest(description='Налоговый орган оспаривает расходы на выкуп имущества',
        plan=SearchPlan(queries=['аренда право выкупа', 'оспаривание расходов выкуп'],
                        must_have=['налоговый предмет спора', 'выкуп ранее арендовавшегося имущества']),
        **kwargs)


def test_land_query_keeps_vri_and_does_not_spend_a_slot_on_a_norm_alone():
    queries = ['применение статьи 285 ГК РФ земельные участки нецелевое использование',
               'изъятие земельного участка несоответствие вида разрешенного использования',
               'расширительное толкование статьи 285 ГК РФ нецелевое использование земли',
               'статья 285 ГК РФ земельный участок не соответствует виду разрешенного использования',
               'основание статья 285 ГК РФ']
    selected = _select_queries(queries, 3, ['статья 285 ГК РФ',
        'нецелевое использование земельного участка', 'вид разрешенного использования'])
    assert queries[3] in selected
    assert queries[4] not in selected


@pytest.mark.parametrize('bad', ['дивidend included in customs value', 'dividends customs value'])
def test_generated_english_criterion_is_rejected_before_search(bad):
    content = json.dumps({'queries': ['дивиденды таможенная стоимость', 'включение дивидендов'],
                          'must_have': [bad]})
    with pytest.raises(LlmError, match='английским'):
        parse_search_plan(content, 'Обжалование включения дивидендов в таможенную стоимость')


def test_latin_party_name_supplied_by_the_user_is_preserved():
    plan = parse_search_plan(json.dumps({'queries': ['аренда Samsung', 'выкуп Samsung'],
                                         'must_have': ['аренда имущества Samsung']}),
                             'Компания Samsung выкупила ранее арендовавшееся имущество')
    assert 'Samsung' in plan.must_have[0]


@pytest.mark.parametrize(('kind', 'content', 'expected'), [
    ('Определение', ['Об отложении рассмотрения заявления/жалобы'], 'procedural'),
    ('Определение', ['Прекратить производство по делу, Принять отказ от иска'], 'procedural'),
    ('Определение', ['О признании сделки должника недействительной'], 'substantive'),
    ('Определение', [], 'unknown'),
])
def test_document_kind_alone_does_not_reject_a_substantive_definition(kind, content, expected):
    document = FakeProvider().document.model_copy(update={'document_type': kind, 'content_types': content})
    assert CaseAggregationService.document_status(document) == expected


@pytest.mark.asyncio
async def test_pagination_visits_all_queries_and_reports_exact_depth_under_call_cap():
    class Source(FakeProvider):
        async def search_documents(self, params):
            self.calls.append((params.text, params.page))
            return DocumentSearchResult(count=90, pages=3, page=params.page, items=[self.document])
    source = Source()
    result = await SemanticSearchService(source, None, settings(search_max_pages_per_query=3)).search(
        SemanticSearchRequest(description='Найти налоговые споры о выкупе имущества',
            plan=request().plan, max_queries=2, max_pages_per_query=3, max_search_calls=4))
    assert source.calls == [(query, page) for page in (1, 2) for query in request().plan.queries]
    assert result.diagnostics.search_calls == 4
    assert all(entry.pages_fetched == [1, 2] and entry.source_pages == 3 and
               entry.source_document_count == 90 and not entry.complete
               for entry in result.diagnostics.query_progress)
    assert result.partial and any('2 из 3' in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_expansion_replaces_adjournment_with_merits_and_rejects_foreign_case(monkeypatch):
    class Source(FakeProvider):
        def __init__(self):
            super().__init__()
            self.document = self.document.model_copy(update={'document_type': 'Определение',
                'content_types': ['Об отложении рассмотрения заявления/жалобы'], 'instance_level': 1})
            self.decision = self.document.model_copy(update={'document_id': 'merits',
                'document_type': 'Решение', 'content_types': [], 'file_url': 'https://example.org/merits.pdf'})
        async def search_documents(self, params):
            self.calls.append(params)
            rows = [self.document] if not params.case_number else [
                self.decision, self.decision.model_copy(update={'document_id': 'foreign', 'case_id': 'foreign'})]
            return DocumentSearchResult(count=len(rows), pages=1, page=1, items=rows)
    source = Source()
    extracted = []
    async def extract(case, provider, *, role):
        document = document_for_role(case, role)
        extracted.append(document.document_id)
        return SimpleNamespace(document=document, text='Суд рассмотрел спор о налоговых расходах и выкупе имущества.')
    monkeypatch.setattr('backend.app.services.semantic_search.extract_case_document_text', extract)
    result = await SemanticSearchService(source, None, settings()).search_with_evidence(
        request(max_case_expansions=1, max_search_calls=3))
    assert len(source.calls) == 3 and source.calls[-1].case_number == source.document.case_number
    assert result.diagnostics.cases_expanded == result.diagnostics.expansion_documents == 1
    assert extracted == ['merits']
    assert result.items[0].evidence_document.document_id == 'merits'
    assert {doc.document_id for doc in result.items[0].case.documents} == {'doc-1', 'merits'}


@pytest.mark.asyncio
async def test_exhausted_search_budget_does_not_download_a_procedural_pdf():
    source = FakeProvider()
    source.document = source.document.model_copy(update={'document_type': 'Определение',
        'content_types': ['Об отложении рассмотрения заявления/жалобы']})
    result = await SemanticSearchService(source, None, settings()).search_with_evidence(
        request(max_case_expansions=1, max_search_calls=2))
    assert result.diagnostics.search_calls == 2
    assert result.diagnostics.pdf_calls == result.cases_checked == 0
    assert result.confirmed_count == 0 and result.rejected_count == 1
    assert result.items[0].document_status == 'procedural'
    assert not any('Не удалось прочитать' in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_procedural_only_case_does_not_take_a_merits_pdf_slot(monkeypatch):
    class Source(FakeProvider):
        async def search_documents(self, params):
            procedural = self.document.model_copy(update={'document_id': 'proc', 'case_id': 'proc',
                'document_type': 'Определение', 'content_types': ['Об отложении рассмотрения заявления/жалобы']})
            other = self.document.model_copy(update={'document_id': 'other', 'case_id': 'other'})
            rows = [self.document, other] if params.text == request().plan.queries[0] else [procedural]
            return DocumentSearchResult(count=len(rows), pages=1, page=1, items=rows)
    checked = []
    async def extract(case, provider, *, role):
        document = document_for_role(case, role)
        checked.append(document.document_id)
        return SimpleNamespace(document=document, text='Расходы на выкуп ранее арендовавшегося имущества.')
    monkeypatch.setattr('backend.app.services.semantic_search.extract_case_document_text', extract)
    result = await SemanticSearchService(Source(), None, settings()).search_with_evidence(request())
    assert result.case_count == 3 and result.cases_checked == 2
    assert set(checked) == {'doc-1', 'other'}
    assert result.diagnostics.procedural_cases_skipped == 1


def candidate():
    document = FakeProvider().document
    case = CaseAggregationService(FakeProvider())._build_case([document])
    return EvidenceCase(case=case, matched_queries=[], evidence_document=document,
        document_status='substantive', relevance_status='semantic', relevance_score=5)


@pytest.mark.parametrize(('status', 'highlights', 'expected'), [
    ('supported', [TextHighlight(start=0, end=10)], 'confirmed'),
    ('supported', [], 'unverified'),
    ('not_shown', [], 'unverified'),
    ('contradicted', [TextHighlight(start=0, end=10)], 'not_recommended'),
])
def test_high_score_cannot_bypass_required_fact_and_exact_quote(status, highlights, expected):
    item = candidate()
    item.semantic_criteria = [CriterionAssessment(term='приобретение компании', status=status,
        quote='Приобретались товары' if status == 'contradicted' else 'Приобретена компания', highlights=highlights)]
    assess_recommendation(item, ['приобретение компании'])
    assert item.recommendation_status == expected


def test_lexical_coverage_never_confirms_legal_situation():
    item = candidate()
    item.relevance_status = 'textual'
    item.coverage = 1
    item.matched_terms = ['налоговый спор']
    assess_recommendation(item, ['налоговый спор'])
    assert item.recommendation_status == 'unverified'


@pytest.mark.asyncio
async def test_service_separates_confirmed_missing_and_contradicted_cases(monkeypatch):
    class Source(FakeProvider):
        async def search_documents(self, params):
            documents = [self.document.model_copy(update={'case_id': key, 'case_number': key,
                'document_id': key}) for key in ('confirmed', 'missing', 'contradicted')]
            return DocumentSearchResult(count=3, pages=1, page=1, items=documents)

    class Judge(RelevanceReranker):
        async def judge(self, description, candidates):
            result = []
            for entry in candidates:
                passage_id = next(iter(entry.passages))
                criteria = [CriterionJudgment(entry.must_have[0], 'supported', passage_id,
                    'Налоговый орган оспаривает расходы')]
                if entry.case_number == 'confirmed':
                    criteria.append(CriterionJudgment(entry.must_have[1], 'supported', passage_id,
                        'выкуп ранее арендовавшегося имущества'))
                elif entry.case_number == 'contradicted':
                    criteria.append(CriterionJudgment(entry.must_have[1], 'contradicted', passage_id,
                        'Выкуп имущества не рассматривался.'))
                # Even an erroneously high overall score cannot bypass admission.
                result.append(RelevanceJudgment(entry.key, 5, 'Оценка модели', passage_id, tuple(criteria)))
            return result

    async def extract(case, provider, *, role):
        text = ('Налоговый орган оспаривает расходы на выкуп ранее арендовавшегося имущества.'
            if case.case_id == 'confirmed' else
            'Налоговый орган оспаривает расходы. Выкуп имущества не рассматривался.')
        return SimpleNamespace(document=document_for_role(case, role), text=text)

    monkeypatch.setattr('backend.app.services.semantic_search.extract_case_document_text', extract)
    result = await SemanticSearchService(Source(), None, settings(), reranker=Judge()).search_with_evidence(
        request(max_cases=3, semantic_reranking=True))
    assert (result.confirmed_count, result.related_count, result.unverified_count, result.rejected_count) == (1, 0, 1, 1)
    assert [item.recommendation_status for item in result.items] == ['confirmed', 'unverified', 'not_recommended']
    assert result.cases_checked == result.cases_attempted == 3
    assert result.items[1].relevance_score == result.items[2].relevance_score == 5


@pytest.mark.asyncio
async def test_hallucinated_criterion_quote_is_downgraded_by_model_adapter():
    class API:
        async def complete(self, **kwargs):
            return json.dumps({'items': [{'key': '0', 'score': 5, 'reason': 'Совпадает', 'passage_id': 'P1',
                'criteria': {'C1': {'status': 'supported', 'passage_id': 'P1', 'quote': 'Приобретена компания'}}}]})
    rows = await GigaChatRelevanceReranker(API()).judge('Покупка компании и переход долга', [
        RelevanceCandidate('0', 'А1', {'P1': 'Приобретались товары.'}, ('приобретение компании',))])
    assert rows[0].criteria[0].status == 'unclear'
    assert rows[0].criteria[0].quote is None


def test_short_pdf_body_is_not_lost_when_its_header_fills_first_excerpt():
    text = 'Служебная информация суда. ' * 60 + 'Налоговый орган оспаривал расходы на выкуп имущества.'
    passages = model_passages(text, 'Спор о выкупе имущества', ['выкуп имущества'], find_evidence(text, ['выкуп имущества']))
    assert 'Налоговый орган оспаривал расходы на выкуп имущества.' in ' '.join(passages.values())
    assert len(passages) <= 5 and all(len(part) <= 1000 for part in passages.values())


def test_long_pdf_sampling_keeps_rare_criterion_and_stays_within_consent_budget():
    text = ('Арбитражный суд рассмотрел заявление. ' * 100 +
        'Общество выкупило ранее арендовавшееся имущество. ' +
        'В материалах отражены многочисленные документы. ' * 100 +
        'Покупатель присоединился к приобретённой компании, к ней перешёл долг. ' +
        'Суд оценил доказательства. ' * 100)
    terms = ['выкуп имущества', 'покупатель присоединился к приобретённой компании']
    passages = model_passages(text, 'Переход долга после присоединения', terms, find_evidence(text, terms))
    assert any('к ней перешёл долг' in part for part in passages.values())
    assert len(passages) <= 5 and all(len(part) <= 1000 for part in passages.values())
