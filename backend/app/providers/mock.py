import re
from datetime import date
from pathlib import Path

from ..models import CourtDocument, DocumentSearchParams, DocumentSearchResult
from .base import CourtProvider


_DEMO_ACTS: tuple[tuple[CourtDocument, str], ...] = (
    (
        CourtDocument(
            document_id="demo-tax-decision",
            case_id="demo-tax-case",
            case_number="DEMO-001",
            registration_date=date(2024, 3, 15),
            instance_level=1,
            court="Учебный арбитражный суд",
            document_type="Решение",
            content_types=["Отказать в удовлетворении требования"],
            file_name="demo-tax-decision.pdf",
            file_url="/api/demo/documents/demo-tax-decision/pdf",
        ),
        "Учебный пример. Вымышленное дело DEMO-001. При реорганизации общество Альфа присоединило "
        "общество Бета, которое ранее получило заем. После присоединения Альфа "
        "учла проценты по займу и убытки присоединенной организации при расчете "
        "налога на прибыль. Налоговый орган оспорил налоговую выгоду, сославшись "
        "на отсутствие деловой цели. Суд исследовал деловую цель присоединения, движение "
        "заемных средств и документы по начислению процентов. Текст создан только "
        "для демонстрации и не является судебным актом.",
    ),
    (
        CourtDocument(
            document_id="demo-tax-procedural",
            case_id="demo-tax-case",
            case_number="DEMO-001",
            registration_date=date(2024, 4, 1),
            instance_level=2,
            court="Учебный арбитражный суд",
            document_type="Определение",
            content_types=["Принять к производству апелляционную жалобу"],
            file_name="demo-tax-procedural.pdf",
            file_url="/api/demo/documents/demo-tax-procedural/pdf",
        ),
        "Учебный пример. Вымышленное дело DEMO-001. Принять апелляционную жалобу "
        "к производству и назначить судебное заседание. Это процессуальный акт.",
    ),
    (
        CourtDocument(
            document_id="demo-reorganization-decision",
            case_id="demo-reorganization-case",
            case_number="DEMO-002",
            registration_date=date(2023, 11, 20),
            instance_level=1,
            court="Учебный арбитражный суд",
            document_type="Решение",
            content_types=["Удовлетворить требование"],
            file_name="demo-reorganization-decision.pdf",
            file_url="/api/demo/documents/demo-reorganization-decision/pdf",
        ),
        "Учебный пример. Вымышленное дело DEMO-002. Компания провела "
        "реорганизацию в форме присоединения. Спор касался передачи имущества "
        "и регистрации прав на склад. Текст создан для демонстрации.",
    ),
    (
        CourtDocument(
            document_id="demo-loan-decision",
            case_id="demo-loan-case",
            case_number="DEMO-003",
            registration_date=date(2022, 9, 8),
            instance_level=1,
            court="Учебный арбитражный суд",
            document_type="Решение",
            content_types=["Отказать в удовлетворении требования"],
            file_name="demo-loan-decision.pdf",
            file_url="/api/demo/documents/demo-loan-decision/pdf",
        ),
        "Учебный пример. Вымышленное дело DEMO-003. Общество привлекло заем и "
        "учло проценты при расчете налога на прибыль. Налоговый орган оспорил "
        "налоговую выгоду из-за отсутствия подтверждающих документов. "
        "Текст создан для демонстрации.",
    ),
)


def _font_path() -> str:
    for candidate in (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    ):
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("No Cyrillic font available for demo PDF")


def _make_pdf(text: str) -> bytes:
    import fitz

    document = fitz.open()
    try:
        page = document.new_page()
        remaining = page.insert_textbox(
            fitz.Rect(50, 60, 545, 780),
            text,
            fontname="DemoFont",
            fontfile=_font_path(),
            fontsize=12,
            lineheight=1.5,
        )
        if remaining < 0:
            raise RuntimeError("Demo PDF text does not fit on the page")
        return document.tobytes()
    finally:
        document.close()


def _word_roots(value: str) -> set[str]:
    return {
        word[:4]
        for word in re.findall(r"[а-яёa-z]{4,}", value.casefold().replace("ё", "е"))
    }


class MockCourtProvider(CourtProvider):
    async def search_documents(self, params: DocumentSearchParams) -> DocumentSearchResult:
        if params.case_number:
            matches = [
                document for document, _ in _DEMO_ACTS
                if document.case_number == params.case_number
            ]
        else:
            query_roots = _word_roots(params.text or "")
            scored = [
                (
                    len(query_roots & _word_roots(text)),
                    document,
                )
                for document, text in _DEMO_ACTS
            ]
            matches = [
                document for score, document in sorted(
                    scored, key=lambda item: (-item[0], item[1].document_id)
                ) if score > 0
            ]
        page_size = 10
        start = (params.page - 1) * page_size
        return DocumentSearchResult(
            count=len(matches),
            pages=(len(matches) + page_size - 1) // page_size,
            page=params.page,
            items=matches[start:start + page_size],
        )

    async def download_pdf(self, file_url: str) -> bytes | None:
        for document, text in _DEMO_ACTS:
            if document.file_url == file_url:
                return _make_pdf(text)
        return None
