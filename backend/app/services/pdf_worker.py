"""Single-use PDF worker. Reads PDF bytes; emits text or a safe error."""
import json
import sys

from .pdf import MAX_PDF_BYTES, PdfExtractionError, extract_pdf_text

if __name__ == '__main__':
    try:
        text = extract_pdf_text(sys.stdin.buffer.read(MAX_PDF_BYTES + 1))
        print(json.dumps({'text': text}, ensure_ascii=True))
    except PdfExtractionError as exc:
        print(json.dumps({'error': str(exc)}))
        sys.exit(1)
