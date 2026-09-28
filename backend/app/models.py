from datetime import date
from hashlib import sha256
from typing import Annotated, Literal
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .parser_filters import canonical_dispute_type, valid_dispute_category


def _validated_dispute_type(value: str | None) -> str | None:
    if value is None:
        return None
    canonical = canonical_dispute_type(value)
    if canonical is None:
        raise ValueError("disputeType must be an official Parser API dispute type")
    return canonical


def _validated_dispute_category(value: str | None) -> str | None:
    if value is None:
        return None
    if not valid_dispute_category(value):
        raise ValueError("disputeCategory must be a category number, for example 7.1")
    return value.strip()


class DocumentSearchParams(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    case_number: str | None = Field(default=None, alias="caseNumber")
    inn: str | None = None
    text: str | None = None
    court: str | None = None
    date_from: date | None = Field(default=None, alias="dateFrom")
    date_to: date | None = Field(default=None, alias="dateTo")
    page: int = Field(default=1, ge=1)
    dispute_type: str | None = Field(default=None, alias="disputeType")
    dispute_category: str | None = Field(default=None, alias="disputeCategory")

    @field_validator("dispute_type")
    @classmethod
    def validate_dispute_type(cls, value: str | None) -> str | None:
        return _validated_dispute_type(value)

    @field_validator("dispute_category")
    @classmethod
    def validate_dispute_category(cls, value: str | None) -> str | None:
        return _validated_dispute_category(value)

    @model_validator(mode="after")
    def validate_search_criteria(self) -> "DocumentSearchParams":
        criteria = (
            self.case_number,
            self.inn,
            self.text,
            self.court,
            self.dispute_type,
            self.dispute_category,
        )
        if not any(value and value.strip() for value in criteria):
            raise ValueError(
                "At least one of caseNumber, inn, text, court, disputeType or disputeCategory is required"
            )
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("dateFrom must be earlier than or equal to dateTo")
        return self


class CaseSearchParams(DocumentSearchParams):
    max_pages: int = Field(default=3, alias="maxPages", ge=1, le=20)
    expand_cases: bool = Field(default=True, alias="expandCases")
    max_cases_to_expand: int = Field(default=10, alias="maxCasesToExpand", ge=1, le=50)
    max_case_pages: int = Field(default=3, alias="maxCasePages", ge=1, le=20)

    def to_document_search_params(self) -> DocumentSearchParams:
        return DocumentSearchParams.model_validate(
            self.model_dump(
                exclude={
                    "max_pages",
                    "expand_cases",
                    "max_cases_to_expand",
                    "max_case_pages",
                },
                by_alias=True,
            )
        )


class CaseCollectionParams(BaseModel):
    """High-level filters for reusable court-case collections.

    `participant` is mapped to Parser API's `inn` field, which the provider
    documentation also accepts for a participant name/FIO. Region resolution
    and OGRN are intentionally not guessed here; they remain provider/KAD work.
    """

    model_config = ConfigDict(populate_by_name=True)

    participant: str | None = None
    text: str | None = None
    court: str | None = None
    date_from: date | None = Field(default=None, alias="dateFrom")
    date_to: date | None = Field(default=None, alias="dateTo")
    dispute_type: str | None = Field(default=None, alias="disputeType")
    dispute_category: str | None = Field(default=None, alias="disputeCategory")

    @field_validator("dispute_type")
    @classmethod
    def validate_dispute_type(cls, value: str | None) -> str | None:
        return _validated_dispute_type(value)

    @field_validator("dispute_category")
    @classmethod
    def validate_dispute_category(cls, value: str | None) -> str | None:
        return _validated_dispute_category(value)
    max_pages: int = Field(default=3, alias="maxPages", ge=1, le=20)
    expand_cases: bool = Field(default=True, alias="expandCases")
    max_cases_to_expand: int = Field(default=10, alias="maxCasesToExpand", ge=1, le=50)
    max_case_pages: int = Field(default=3, alias="maxCasePages", ge=1, le=20)

    @model_validator(mode="after")
    def validate_collection(self) -> "CaseCollectionParams":
        criteria = (
            self.participant,
            self.text,
            self.court,
            self.dispute_type,
            self.dispute_category,
        )
        if not any(value and value.strip() for value in criteria):
            raise ValueError(
                "At least one of participant, text, court, disputeType or disputeCategory is required"
            )
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("dateFrom must be earlier than or equal to dateTo")
        return self

    def to_case_search_params(self) -> CaseSearchParams:
        return CaseSearchParams(
            inn=self.participant,
            text=self.text,
            court=self.court,
            dateFrom=self.date_from,
            dateTo=self.date_to,
            disputeType=self.dispute_type,
            disputeCategory=self.dispute_category,
            maxPages=self.max_pages,
            expandCases=self.expand_cases,
            maxCasesToExpand=self.max_cases_to_expand,
            maxCasePages=self.max_case_pages,
        )


class CourtDocument(BaseModel):
    document_id: str
    case_id: str
    case_number: str
    case_url: str | None = None
    registration_date: date | None = None
    instance_number: str | None = None
    instance_level: int | None = None
    court: str | None = None
    document_type: str | None = None
    content_types: list[str] = Field(default_factory=list)
    file_name: str | None = None
    file_url: str

    @staticmethod
    def build_document_id(file_url: str, fallback: str = "") -> str:
        stable_value = file_url or fallback
        return sha256(stable_value.encode("utf-8")).hexdigest()


class DocumentSearchResult(BaseModel):
    count: int
    pages: int
    page: int
    items: list[CourtDocument]


class CourtCase(BaseModel):
    case_id: str
    case_number: str
    case_url: str | None = None
    document_count: int
    latest_document_date: date | None = None
    highest_instance_level: int | None = None
    documents: list[CourtDocument]
    first_instance_documents: list[CourtDocument] = Field(default_factory=list)
    first_instance_document: CourtDocument | None = None
    factual_base_document: CourtDocument | None = None
    appellate_documents: list[CourtDocument] = Field(default_factory=list)
    cassation_documents: list[CourtDocument] = Field(default_factory=list)
    procedural_documents: list[CourtDocument] = Field(default_factory=list)
    latest_substantive_document: CourtDocument | None = None

    # Deprecated compatibility fields. New consumers should use the explicit
    # document roles above; these remain so older clients can migrate gradually.
    preferred_document_id: str | None = Field(
        default=None,
        json_schema_extra={"deprecated": True},
    )
    preferred_document: CourtDocument | None = Field(
        default=None,
        json_schema_extra={"deprecated": True},
    )


class CaseSearchResult(BaseModel):
    source_document_count: int
    source_pages: int
    pages_fetched: int
    candidate_unique_document_count: int
    filtered_out_by_date: int
    case_expansion_pages_fetched: int
    expanded_case_count: int
    unique_document_count: int
    case_count: int
    items: list[CourtCase]


class PdfTextRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    file_url: str = Field(alias="fileUrl")


class PdfTextResponse(BaseModel):
    file_url: str
    text: str
    char_count: int


class PreferredDocumentTextResponse(BaseModel):
    case_id: str
    case_number: str
    document_id: str
    document: CourtDocument
    text: str
    char_count: int


class CaseDocumentTextResponse(BaseModel):
    case_id: str
    case_number: str
    role: str
    document_id: str
    document: CourtDocument
    text: str
    char_count: int


class SearchFilters(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    case_number: str | None = Field(default=None, alias="caseNumber", max_length=100)
    inn: str | None = Field(default=None, max_length=300)
    court: str | None = Field(default=None, max_length=300)
    date_from: date | None = Field(default=None, alias="dateFrom")
    date_to: date | None = Field(default=None, alias="dateTo")
    dispute_type: str | None = Field(default=None, alias="disputeType", max_length=300)
    dispute_category: str | None = Field(default=None, alias="disputeCategory", max_length=100)

    @field_validator("dispute_type")
    @classmethod
    def validate_dispute_type(cls, value: str | None) -> str | None:
        return _validated_dispute_type(value)

    @field_validator("dispute_category")
    @classmethod
    def validate_dispute_category(cls, value: str | None) -> str | None:
        return _validated_dispute_category(value)

    @model_validator(mode="after")
    def validate_dates(self) -> "SearchFilters":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("dateFrom must be earlier than or equal to dateTo")
        return self


SearchTerm = Annotated[str, Field(min_length=1, max_length=300)]


class SearchPlan(BaseModel):
    filters: SearchFilters = Field(default_factory=SearchFilters)
    queries: list[SearchTerm] = Field(min_length=2, max_length=10)
    must_have: list[SearchTerm] = Field(default_factory=list, max_length=10)
    exclude: list[SearchTerm] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def clean_terms(self) -> "SearchPlan":
        def clean(values):
            seen, result = set(), []
            for value in values:
                value = unicodedata.normalize("NFC", value).strip()
                key = value.casefold().replace("ё", "е")
                if value and key not in seen:
                    seen.add(key)
                    result.append(value)
            return result
        self.queries = clean(self.queries)
        self.must_have = clean(self.must_have)
        self.exclude = clean(self.exclude)
        if len(self.queries) < 2:
            raise ValueError("At least two distinct search queries are required")
        return self


class SemanticSearchRequest(BaseModel):
    filters: SearchFilters | None = None
    description: str = Field(min_length=20, max_length=5000)
    max_pages_per_query: int = Field(default=1, ge=1, le=5)
    plan: SearchPlan | None = None


class RetrievedCase(BaseModel):
    case: CourtCase
    matched_queries: list[str]


class SemanticSearchResponse(BaseModel):
    diagnostics: "SearchDiagnostics | None" = None
    partial: bool = False
    warnings: list[str] = Field(default_factory=list)
    plan: SearchPlan
    query_count: int
    document_count: int
    case_count: int
    items: list[RetrievedCase]


class EvidenceSearchRequest(SemanticSearchRequest):
    max_cases: int = Field(default=6, ge=1, le=20)
    semantic_reranking: bool = False


class TextHighlight(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class EvidenceMatch(BaseModel):
    term: str
    quote: str
    highlights: list[TextHighlight] = Field(default_factory=list)


class CriterionAssessment(BaseModel):
    term: str
    status: Literal['supported', 'not_shown', 'unclear']
    quote: str | None = None
    highlights: list[TextHighlight] = Field(default_factory=list)


class SearchEvent(BaseModel):
    stage: str
    operation: str
    attempt: int = 1
    seconds: float
    outcome: str
    http_status: int | None = None
    error_code: str | int | None = None


class SearchDiagnostics(BaseModel):
    request_id: str
    total_seconds: float = 0
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    search_calls: int = 0
    pdf_calls: int = 0
    http_attempts: int = 0
    pdf_downloaded: int = 0
    pdf_checked: int = 0
    pdf_bytes: int = 0
    filtered_by_date: int = 0
    queries_executed: list[str] = Field(default_factory=list)
    events: list[SearchEvent] = Field(default_factory=list)
    limits: dict[str, float | int] = Field(default_factory=dict)


class EvidenceCase(BaseModel):
    source_document: CourtDocument | None = None
    evidence_document: CourtDocument | None = None
    evidence: list[EvidenceMatch] = Field(default_factory=list)
    semantic_criteria: list[CriterionAssessment] = Field(default_factory=list)
    exclusion_evidence: list[EvidenceMatch] = Field(default_factory=list)
    verification_status: str = "unverified"
    case: CourtCase
    matched_queries: list[str]
    excerpt: str | None = None
    matched_terms: list[str] = Field(default_factory=list)
    missing_terms: list[str] = Field(default_factory=list)
    excluded_terms: list[str] = Field(default_factory=list)
    coverage: float | None = None
    text_error: str | None = None
    relevance_score: int | None = Field(default=None, ge=0, le=5)
    relevance_reason: str | None = None
    relevance_quote: str | None = None
    relevance_status: str = "unverified"


class EvidenceSearchResponse(BaseModel):
    diagnostics: SearchDiagnostics | None = None
    partial: bool = False
    warnings: list[str] = Field(default_factory=list)
    cases_attempted: int = 0
    plan: SearchPlan
    document_count: int
    case_count: int
    cases_checked: int
    items: list[EvidenceCase]

SemanticSearchResponse.model_rebuild()
