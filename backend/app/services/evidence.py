"""Lexical evidence with source quotes, never a legal relevance verdict."""
from functools import lru_cache
import re
import unicodedata
from threading import Lock

import pymorphy3

from ..models import EvidenceMatch, TextHighlight

# Grammatical function words, never subjects, outcomes, or legal categories.
_STOP_WORDS = frozenset('в во на по к ко с со из за от до для при о об и или либо а но что как это'.split())
_ANALYZER_LOCK = Lock()
_TOKEN = re.compile(r'[а-яёa-z0-9][а-яёa-z0-9\u0300-\u036f]*', re.I)
_SENTENCE_BREAK = re.compile(r'(?<=[.!?])\s+(?=[А-ЯЁA-Z«])|\n\s*\n')
_ALTERNATIVE = re.compile(r'\s*/\s*|\s+(?:или|либо)\s+', re.I)
_WORD_ALTERNATIVE = re.compile(r'([а-яёa-z0-9-]+)\s*/\s*([а-яёa-z0-9-]+)', re.I)


def term_alternatives(term: str) -> list[str]:
    # "обязательство/убытки по займу перешли" is two facts with shared
    # qualifiers, not a bare "обязательство" OR the remaining sentence.
    # Expand slash alternatives in place. For full clauses
    # joined with или/либо retain each complete branch instead.
    match = _WORD_ALTERNATIVE.search(term)
    if match:
        left, right = term.split('/', 1)
        left_words, right_words = _TOKEN.findall(normalize(left)), _TOKEN.findall(normalize(right))
        if len(left_words) > 1 and len(right_words) > 1 and word_forms(left_words[0]) & word_forms(right_words[0]):
            return [*term_alternatives(left), *term_alternatives(right)]
        prefix, suffix = term[:match.start()], term[match.end():]
        return list(dict.fromkeys(value for word in match.groups()
            for value in term_alternatives(prefix + word + suffix)))
    return _ALTERNATIVE.split(term)


def normalize(value: str) -> str:
    return unicodedata.normalize('NFC', value).casefold().replace('ё', 'е')


@lru_cache(maxsize=1)
def analyzer():
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=32768)
def word_forms(word: str) -> frozenset[str]:
    if re.fullmatch('[а-я]+', word):
        with _ANALYZER_LOCK:
            return frozenset(parse.normal_form for parse in analyzer().parse(word))
    return frozenset([word])


def _is_word_character(character: str) -> bool:
    return character.isalnum() or unicodedata.category(character).startswith('M')


def _source_quote(source: str, start: int, end: int) -> tuple[str, int]:
    """Prefer a complete sentence; otherwise keep exact, whole-word context."""
    preceding = list(_SENTENCE_BREAK.finditer(source, max(0, start - 300), start))
    following = _SENTENCE_BREAK.search(source, end, min(len(source), end + 350))
    left = preceding[-1].end() if preceding else max(0, start - 120)
    right = following.start() if following else min(len(source), end + 160)
    if right - left > 600:
        left = max(0, start - 140)
        right = min(len(source), end + 220)
    # The fallback can land inside a word. Expand rather than drop source text.
    while left > 0 and left < len(source) and _is_word_character(source[left - 1]) and _is_word_character(source[left]):
        left -= 1
    while right < len(source) and right > 0 and _is_word_character(source[right - 1]) and _is_word_character(source[right]):
        right += 1
    raw = source[left:right]
    leading = len(raw) - len(raw.lstrip())
    return raw.strip(), left + leading


def find_evidence(text: str, terms: list[str]) -> list[EvidenceMatch]:
    # Preserve original character offsets and quotes, normalizing tokens only.
    source = text
    tokens = [(word_forms(normalize(m.group())), m.start(), m.end())
              for m in _TOKEN.finditer(source)]
    matches = []
    for term in terms:
        alternatives = [
            {word_forms(word) for word in _TOKEN.findall(normalize(part))
             if word not in _STOP_WORDS}
            for part in term_alternatives(term)
        ]
        # A slash or "или" joins alternative facts. Only one branch has to be
        # supported, while words within that branch still share a short passage.
        for required in alternatives:
            if not required:
                continue
            # All words in one alternative must occur together in a short
            # passage, not across pages or inside unrelated longer words.
            for i, (word, start, _) in enumerate(tokens):
                if not any(word & forms for forms in required):
                    continue
                window = tokens[i:i + max(18, len(required) * 3)]
                seen = set()
                for index, (word, _, end) in enumerate(window):
                    if end - start > 300:
                        break
                    seen.update(word)
                    if all(forms & seen for forms in required):
                        quote, quote_start = _source_quote(source, start, end)
                        highlights = [TextHighlight(start=token_start - quote_start,
                                                    end=token_end - quote_start)
                                      for token_forms, token_start, token_end in window[:index + 1]
                                      if any(token_forms & forms for forms in required)]
                        matches.append(EvidenceMatch(term=term, quote=quote,
                                                     highlights=highlights))
                        break
                if matches and matches[-1].term == term:
                    break
            if matches and matches[-1].term == term:
                break
    return matches
