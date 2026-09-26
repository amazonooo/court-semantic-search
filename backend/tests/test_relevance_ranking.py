from types import SimpleNamespace

import pytest

from backend.app.config import Settings
from backend.app.llm.base import RelevanceReranker
from backend.app.models import DocumentSearchResult, EvidenceSearchRequest, SearchPlan
from backend.app.services.evidence import find_evidence
from backend.app.services.relevance import grounded_model_quote, score_textual_relevance
from backend.app.services.relevance import CriterionJudgment, RelevanceJudgment
from backend.app.services.semantic_search import SemanticSearchService
from backend.tests.test_semantic_search import FakeProvider


SUPPLY_DESCRIPTION = (
    "Поставщик передал покупателю товар по договору поставки. Покупатель не оплатил "
    "товар, поставщик требует взыскать задолженность."
)
SUPPLY_TERMS = ["договор поставки", "передача товара", "неоплата товара",
                "взыскание задолженности"]
SUPPLY_MATCH = (
    "Суд рассмотрел иск поставщика к покупателю о взыскании задолженности по "
    "договору поставки. Истец доказал передачу товара по накладным, покупатель "
    "товар принял, но не оплатил. Из-за неоплаты товара заявлено взыскание задолженности."
)
SUPPLY_FALSE = (
    "Конкурсный управляющий оспорил продажу жилого дома должника в деле о банкротстве. "
    "В материалах также упомянут договор поставки тепловой энергии. В цитате статьи "
    "Гражданского кодекса говорится о передаче товара покупателю. Требование касается "
    "возврата жилого дома в конкурсную массу."
)
TAX_DESCRIPTION = (
    "После присоединения компании-заемщика правопреемник учел проценты по займу "
    "и убытки при расчете налога на прибыль. Налоговая оспаривает налоговую выгоду."
)
TAX_TERMS = ["присоединение компании", "проценты по займу", "налог на прибыль",
             "налоговая выгода"]
TAX_MATCH = (
    "Налоговый орган оспаривает налоговую выгоду правопреемника: после "
    "присоединения компании-заемщика тот учел проценты по займу и убытки, "
    "уменьшив налог на прибыль. Предмет спора — деловая цель реорганизации."
)
TAX_FALSE = (
    "Суд рассмотрел привлечение руководителя банкрота к субсидиарной ответственности "
    "за непередачу документов. В качестве обстоятельства упомянуто присоединение "
    "компании, но налоговые расходы не оспаривались."
)


@pytest.mark.parametrize(("description", "terms", "good", "bad"), [
    (SUPPLY_DESCRIPTION, SUPPLY_TERMS, SUPPLY_MATCH, SUPPLY_FALSE),
    (TAX_DESCRIPTION, TAX_TERMS, TAX_MATCH, TAX_FALSE),
])
def test_unrelated_mentions_rank_below_matching_legal_situation(description, terms, good, bad):
    positive = score_textual_relevance(good, description, terms, [], find_evidence(good, terms))
    negative = score_textual_relevance(bad, description, terms, [], find_evidence(bad, terms))
    assert positive.score >= 3
    assert negative.score <= 1
    assert positive.score > negative.score
    assert positive.quote in good


def test_bankruptcy_request_is_not_excluded_by_topic():
    description = "При банкротстве конкурсный управляющий требует привлечь директора к субсидиарной ответственности."
    terms = ["банкротство", "конкурсный управляющий", "субсидиарная ответственность"]
    document = (
        "В деле о банкротстве конкурсный управляющий заявил требование о привлечении "
        "директора к субсидиарной ответственности за непередачу документов должника."
    )
    result = score_textual_relevance(document, description, terms, [], find_evidence(document, terms))
    assert result.score >= 3


def test_model_quote_highlights_only_verbatim_source_text():
    source = "Общество присоединило заемщика и затем учло проценты по займу при расчете налога."
    quote, highlights = grounded_model_quote(source, "учло проценты по займу")
    assert quote in source
    assert [quote[span.start:span.end] for span in highlights] == ["учло проценты по займу"]
    unsupported, spans = grounded_model_quote(source, "учет процентов по займу")
    assert unsupported == source
    assert spans == []


@pytest.mark.asyncio
async def test_search_samples_across_queries_and_sorts_by_pdf_relevance(monkeypatch):
    class Judge(RelevanceReranker):
        async def judge(self, description, candidates):
            assert description == SUPPLY_DESCRIPTION
            assert all(item.must_have == tuple(SUPPLY_TERMS) for item in candidates)
            return [RelevanceJudgment(
                key=item.key,
                score=5 if item.case_number == "А40-4/2025" else 1,
                reason="Совпадает предмет требования" if item.case_number == "А40-4/2025" else "Другой предмет",
                passage_id=next(iter(item.passages)),
                criteria=tuple(CriterionJudgment(
                    term=term,
                    status="supported" if item.case_number == "А40-4/2025" else "not_shown",
                    passage_id=next(iter(item.passages)) if item.case_number == "А40-4/2025" else None,
                    quote="поставщика" if item.case_number == "А40-4/2025" else None,
                ) for term in SUPPLY_TERMS),
            ) for item in candidates]

    class Source(FakeProvider):
        def __init__(self):
            super().__init__()
            self.documents = [self.document.model_copy(update={
                "document_id": f"doc-{i}", "case_id": f"case-{i}",
                "case_number": f"А40-{i}/2025", "file_url": f"https://example.org/{i}.pdf",
            }) for i in range(5)]

        async def search_documents(self, params):
            rows = self.documents[:4] if params.text == "широкая поставка" else [self.documents[4]]
            return DocumentSearchResult(count=len(rows), pages=1, page=1, items=rows)

    async def extract(case, provider, *, role):
        document = case.documents[0]
        return SimpleNamespace(document=document,
                               text=SUPPLY_MATCH if case.case_id == "case-4" else SUPPLY_FALSE)

    monkeypatch.setattr("backend.app.services.semantic_search.extract_case_document_text", extract)
    request = EvidenceSearchRequest(description=SUPPLY_DESCRIPTION, max_cases=3,
        semantic_reranking=True,
        plan=SearchPlan(queries=["широкая поставка", "неоплата переданного товара"],
                        must_have=SUPPLY_TERMS))
    result = await SemanticSearchService(Source(), None, Settings(_env_file=None,
        search_max_cases=3, search_max_pdf_downloads=3), reranker=Judge()).search_with_evidence(request)
    assert result.case_count == 5
    assert result.cases_checked == 3
    assert result.items[0].case.case_id == "case-4"
    assert result.items[0].relevance_score == 5
    assert result.items[0].relevance_status == "semantic"
    assert result.items[0].relevance_quote
    assert [entry.term for entry in result.items[0].semantic_criteria] == SUPPLY_TERMS
    assert all(entry.status == "supported" and entry.quote
               for entry in result.items[0].semantic_criteria)
    assert result.items[0].semantic_criteria[0].highlights
    assert result.items[-1].relevance_score <= 1
