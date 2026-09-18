class PdfExtractionError(RuntimeError):
    pass


def extract_pdf_text(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        return ""

    try:
        import fitz

        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PdfExtractionError("Could not open PDF document") from exc

    try:
        return "\n".join(page.get_text("text") for page in document).strip()
    except Exception as exc:
        raise PdfExtractionError("Could not extract text from PDF document") from exc
    finally:
        document.close()
