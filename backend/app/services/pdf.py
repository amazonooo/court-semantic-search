import asyncio
import json
from pathlib import Path
import sys

MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 200
MAX_TEXT_CHARS = 500_000


class PdfExtractionError(RuntimeError):
    pass


def extract_pdf_text(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        return ''
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise PdfExtractionError('PDF exceeds the 10 MiB processing limit')
    try:
        import pymupdf
        document = pymupdf.open(stream=pdf_bytes, filetype='pdf')
    except Exception as exc:
        raise PdfExtractionError('Could not open PDF document') from exc
    try:
        if document.page_count > MAX_PDF_PAGES:
            raise PdfExtractionError('PDF exceeds the 200 page processing limit')
        parts, length = [], 0
        for page in document:
            text = page.get_text('text')
            length += len(text)
            if length > MAX_TEXT_CHARS:
                raise PdfExtractionError('PDF text exceeds the processing limit')
            parts.append(text)
        return '\n'.join(parts).strip()
    except PdfExtractionError:
        raise
    except Exception as exc:
        raise PdfExtractionError('Could not extract text from PDF document') from exc
    finally:
        document.close()


async def extract_pdf_text_async(pdf_bytes: bytes) -> str:
    """Isolate PDF parsing; cancellation terminates CPU work as well as waiting.

    PyMuPDF does not support concurrent threads. A subprocess also keeps a
    damaged/large PDF from blocking the API event loop or surviving its budget.
    """
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise PdfExtractionError('PDF exceeds the 10 MiB processing limit')
    process = await asyncio.create_subprocess_exec(
        sys.executable, '-m', 'backend.app.services.pdf_worker',
        cwd=Path(__file__).resolve().parents[3],
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(pdf_bytes), 15)
        try:
            result = json.loads(stdout)
        except (ValueError, UnicodeDecodeError) as exc:
            raise PdfExtractionError('PDF worker did not return a valid result') from exc
        if process.returncode or result.get('error'):
            raise PdfExtractionError(result.get('error') or 'PDF worker failed')
        return result['text']
    except TimeoutError as exc:
        raise PdfExtractionError('PDF extraction exceeded its time budget') from exc
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()
