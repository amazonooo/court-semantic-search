"""Admission to the recommended results requires sourced legal facts, not words."""
from ..models import EvidenceCase


def assess_recommendation(item: EvidenceCase, required_terms: list[str]) -> None:
    item.recommendation_status = 'unverified'
    if item.document_status == 'procedural':
        item.recommendation_status = 'not_recommended'
        item.recommendation_reason = 'Получен процессуальный акт; нужный вопрос по существу не проверен.'
        return
    if item.text_error or not item.evidence_document:
        item.recommendation_reason = 'PDF не прочитан или проверка текста не завершена.'
        return
    if item.relevance_score is not None and item.relevance_score <= 2:
        item.recommendation_status = 'not_recommended'
        item.recommendation_reason = ('Модель оценила только общую тему или другой предмет спора.'
            if item.relevance_status == 'semantic' else
            'Слабое словесное совпадение; юридическое сходство не подтверждено.')
        return
    if item.relevance_status != 'semantic':
        item.recommendation_reason = 'Смысловая проверка не выполнена; совпадение слов не подтверждает условия.'
        return
    by_term = {entry.term: entry for entry in item.semantic_criteria}
    contradicted = [term for term in required_terms if term in by_term and
                   by_term[term].status == 'contradicted' and by_term[term].highlights]
    if contradicted:
        item.recommendation_status = 'not_recommended'
        item.recommendation_reason = 'В тексте показано несоответствие условий: ' + '; '.join(contradicted)
        return
    missing = [term for term in required_terms if term not in by_term or
               by_term[term].status != 'supported' or not by_term[term].highlights]
    if not required_terms or missing:
        item.recommendation_reason = ('Не все обязательные условия подтверждены цитатами: ' + '; '.join(missing)
            if missing else 'В плане нет обязательных условий для подтверждения сходства.')
        return
    if item.excluded_terms:
        item.recommendation_reason = 'Найдены слова исключения; их контекст требует отдельной проверки.'
        return
    if item.document_status != 'substantive':
        item.recommendation_reason = 'Не установлено, что выбранный акт разрешает вопрос по существу.'
        return
    if item.relevance_score is not None and item.relevance_score >= 4:
        item.recommendation_status = 'confirmed'
        item.recommendation_reason = 'По проанализированному тексту модель подтвердила все обязательные условия цитатами.'
    elif item.relevance_score == 3:
        item.recommendation_status = 'related'
        item.recommendation_reason = 'Условия подтверждены в тексте, но модель оценила сходство ситуации как частичное.'
    else:
        item.recommendation_reason = 'Модель не смогла оценить сходство по полученному тексту.'
