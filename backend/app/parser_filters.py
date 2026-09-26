"""Parser API's documented filter vocabulary and plan-boundary helpers."""

import re


DISPUTE_TYPES = (
    "дисциплинарные споры",
    "о несостоятельности (банкротстве) организаций и граждан",
    "о признании и приведении в исполнение решений иностранных судов и иностранных арбитражных решений",
    "об административных правонарушениях",
    "об оспаривании решений трет. судов и о выдаче исп. листов на принудительное исполнение решений трет. судов",
    "об установлении фактов, имеющих юридическое значение",
    "экономические споры по административным правоотношениям",
    "экономические споры по гражданским правоотношениям",
)


def _fold(value: str) -> str:
    return " ".join(value.casefold().split())


_TYPES_BY_NAME = {_fold(value): value for value in DISPUTE_TYPES}


def canonical_dispute_type(value: str) -> str | None:
    """Return the API spelling only for a documented dispute type."""
    return _TYPES_BY_NAME.get(_fold(value))


def valid_dispute_category(value: str) -> bool:
    """The API accepts category numbers such as 7.1 or 20.2.6.1."""
    return re.fullmatch(r"\d+(?:\.\d+)*", value.strip()) is not None


def _take(filters: dict, name: str, alias: str) -> object:
    value = filters.pop(name, None)
    aliased = filters.pop(alias, None)
    return value if value is not None else aliased


def generated_filters(raw: dict, description: str) -> dict:
    """Keep model-proposed filters only when supported by the user's text.

    The LLM may invent a plausible English slug, category number, court, or
    year. These must not become API filters that reject or narrow every query.
    Structured filters supplied by API callers are validated separately.
    """
    filters = dict(raw)
    source = _fold(description)
    for name, alias in (("case_number", "caseNumber"), ("inn", "inn"), ("court", "court")):
        proposed = _take(filters, name, alias)
        if isinstance(proposed, str) and proposed.strip() and _fold(proposed) in source:
            filters[name] = proposed.strip()

    mentioned_years = {int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", description)}
    # A year in the facts (for example, the merger year) is not a request to
    # limit publication dates. Require a stated search period as well.
    explicit_period = bool(
        re.search(r"\b(?:найд\w*|ищ\w*|поиск\w*|дела|акты|решения|практик\w*)\b", source)
        and re.search(r"\b(?:с|после|до|по|за|в период)\s+(?:\d{1,2}[^\d\s]+\s*)?(?:19|20)\d{2}\b", source)
    )
    for name, alias in (("date_from", "dateFrom"), ("date_to", "dateTo")):
        proposed = _take(filters, name, alias)
        if not isinstance(proposed, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", proposed):
            continue
        year = int(proposed[:4])
        after_previous_year = name == "date_from" and (year - 1) in mentioned_years and bool(
            re.search(rf"\bпосле\s+{year - 1}\b", source)
        )
        if explicit_period and (year in mentioned_years or after_previous_year or proposed in description):
            filters[name] = proposed

    proposed_type = _take(filters, "dispute_type", "disputeType")
    proposed_category = _take(filters, "dispute_category", "disputeCategory")
    if isinstance(proposed_type, str):
        canonical = canonical_dispute_type(proposed_type)
        if canonical and _fold(canonical) in source:
            filters["dispute_type"] = canonical
    if isinstance(proposed_category, str):
        category = proposed_category.strip()
        category_mentioned = re.search(
            rf"\bкатегори\w*(?:\s+спора)?\s*(?:№\s*)?{re.escape(category)}(?![\d.])",
            source,
        )
        if valid_dispute_category(category) and category_mentioned:
            filters["dispute_category"] = category
    return filters
