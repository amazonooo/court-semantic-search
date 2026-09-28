"""Evidence-grounded relevance inputs; lexical sampling never decides relevance."""
from dataclasses import dataclass
import re
import unicodedata

from ..models import EvidenceMatch, TextHighlight
from .evidence import find_evidence

_WORD = re.compile(r"[а-яёa-z]{4,}", re.I)
_SPACE = re.compile(r"\s+")
_COMMON = frozenset(
    "дело дела суд суда судебный судебного истец ответчик требование требования "
    "решение постановление российской федерации право правовой спор спора"
    .split()
)


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold().replace("ё", "е")


def _search_words(description: str, terms: list[str]) -> set[str]:
    words = _WORD.findall(_normalize(" ".join([description, *terms])))
    return {word[:6] for word in words if word not in _COMMON}


def _whole_words(source: str, start: int, end: int) -> str:
    while start > 0 and not source[start - 1].isspace():
        start -= 1
    while end < len(source) and not source[end].isspace():
        end += 1
    return source[start:end].strip()


def _short_quote(value: str, length: int = 700) -> str:
    if len(value) <= length:
        return value
    return value[:length].rsplit(" ", 1)[0].strip()


def sample_passages(text: str, description: str, terms: list[str],
                    matches: list[EvidenceMatch] | None = None) -> dict[str, str]:
    """Keep opening, disposition and diverse topical windows within a bounded prompt."""
    source = _SPACE.sub(" ", text).strip()
    if not source:
        return {}
    passages: list[str] = [_whole_words(source, 0, min(1500, len(source)))]
    words = _search_words(description, terms)
    windows = []
    for start in range(0, len(source), 600):
        window = _whole_words(source, start, min(start + 1100, len(source)))
        if len(window) < 120:
            continue
        hits = {word[:6] for word in _WORD.findall(_normalize(window))} & words
        windows.append((len(hits), start, window))
    used_starts: list[int] = []
    for _score, start, window in sorted(windows, key=lambda row: (-row[0], row[1])):
        if len(used_starts) >= 4:
            break
        if start < 1000 or any(abs(start - earlier) < 1000 for earlier in used_starts):
            continue
        used_starts.append(start)
        passages.append(window)
    for match in matches or []:
        quote = _SPACE.sub(" ", match.quote).strip()
        if quote and not any(quote in passage for passage in passages):
            passages.append(quote[:600])
        if len(passages) >= 7:
            break
    ending = _whole_words(source, max(0, len(source) - 1000), len(source))
    if ending not in passages and len(source) > 1500:
        passages.append(ending)
    return {f"P{index}": passage for index, passage in enumerate(passages[:8], 1)}


@dataclass(frozen=True)
class TextualRelevance:
    score: int
    reason: str
    quote: str


@dataclass(frozen=True)
class RelevanceCandidate:
    key: str
    case_number: str
    passages: dict[str, str]
    must_have: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelevanceJudgment:
    key: str
    score: int | None
    reason: str
    passage_id: str | None
    criteria: tuple['CriterionJudgment', ...] = ()


@dataclass(frozen=True)
class CriterionJudgment:
    term: str
    status: str
    passage_id: str | None
    quote: str | None = None


def grounded_model_quote(passage: str, selected: str | None) -> tuple[str, list[TextHighlight]]:
    """Show source context and mark only a model quote copied verbatim from it."""
    if not selected or selected != selected.strip() or len(selected) > 240 or selected not in passage:
        return passage, []
    position = passage.find(selected)
    left = max(0, position - 90)
    right = min(len(passage), position + len(selected) + 90)
    while left > 0 and passage[left - 1].isalnum() and passage[left].isalnum():
        left -= 1
    while right < len(passage) and passage[right - 1].isalnum() and passage[right].isalnum():
        right += 1
    context = passage[left:right]
    leading = len(context) - len(context.lstrip())
    quote = context.strip()
    return quote, [TextHighlight(start=position - left - leading,
                                 end=position - left - leading + len(selected))]


def model_passages(text: str, description: str, terms: list[str],
                   matches: list[EvidenceMatch]) -> dict[str, str]:
    """At most five short source excerpts per PDF leave the user's machine."""
    all_passages = list(sample_passages(text, description, terms, matches).values())
    selected = [_short_quote(part, 1000) for part in all_passages[:4]]
    source = _SPACE.sub(" ", text).strip()
    ending = _whole_words(source, max(0, len(source) - 1000), len(source))
    if ending and len(source) > 1500 and ending not in selected:
        selected.append(_short_quote(ending, 1000))
    return {f"P{index}": passage for index, passage in enumerate(selected, 1)}


def score_textual_relevance(text: str, description: str, must_have: list[str],
                            exclude: list[str], matches: list[EvidenceMatch]) -> TextualRelevance:
    """Conservative local ranking based on independent facts and their context.

    A match in a legal citation or a distant paragraph cannot by itself receive a
    high score. This is intentionally labeled textual, not a semantic verdict.
    """
    passages = sample_passages(text, description, must_have, matches)
    if not passages:
        return TextualRelevance(0, "Из PDF не удалось получить контекст для сравнения.", "")
    if not must_have:
        words = _search_words(description, [])
        ratios = [(len({word[:6] for word in _WORD.findall(_normalize(part))} & words)
                   / max(1, len(words)), part) for part in passages.values()]
        ratio, passage = max(ratios, key=lambda pair: pair[0])
        score = min(2, round(3 * ratio))
        return TextualRelevance(score,
            "План не содержит отдельных обязательных признаков; оценено только совпадение контекста описания.",
            _short_quote(passage))
    weights = {term: min(3, max(1, len(_WORD.findall(term)))) for term in must_have}
    total_weight = sum(weights.values())
    found = {match.term for match in matches if match.term in weights}
    overall = sum(weights[term] for term in found) / total_weight
    per_passage = []
    description_words = _search_words(description, [])
    for index, passage in enumerate(passages.values()):
        contextual = {match.term for match in find_evidence(passage, must_have)}
        ratio = sum(weights[term] for term in contextual) / total_weight
        topical_words = {word[:6] for word in _WORD.findall(_normalize(passage))}
        topical_overlap = len(topical_words & description_words)
        # The beginning usually states the claim; a citation may merely repeat
        # a legal formula. When no full criterion matches, show the passage
        # closest to the description instead of an uninformative document header.
        per_passage.append((ratio, topical_overlap, -index, passage, contextual))
    best, _, _, quote, joined_terms = max(per_passage)
    opening = per_passage[0][0]
    raw = 4 * (0.45 * overall + 0.40 * best + 0.15 * opening)
    score = min(4, max(0, round(raw)))
    if overall <= 0.5:
        score = min(score, 1)
    elif best < 0.5 and opening < 0.25:
        score = min(score, 2)
    explicit_exclusions = {match.term for match in find_evidence(passages["P1"], exclude)}
    if explicit_exclusions:
        score = max(0, score - 2)
    absent = [term for term in must_have if term not in found]
    reason = (f"Словесные признаки: {len(found)} из {len(must_have)}; "
              f"в одном контексте: {len(joined_terms)} из {len(must_have)}.")
    if absent:
        reason += " Не найдены: " + "; ".join(absent[:4]) + "."
    if explicit_exclusions:
        reason += " В предмете акта встречается явное исключение из запроса."
    reason += " Совпадение текста ещё не подтверждает юридическое сходство."
    return TextualRelevance(score, reason, _short_quote(quote))
