from datetime import date
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    preferred_document_id: str
    preferred_document: CourtDocument
    documents: list[CourtDocument]


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
