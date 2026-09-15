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
        return self


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


class PdfTextRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    file_url: str = Field(alias="fileUrl")


class PdfTextResponse(BaseModel):
    file_url: str
    text: str
    char_count: int
