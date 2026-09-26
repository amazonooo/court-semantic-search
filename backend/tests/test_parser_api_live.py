"""Opt in with RUN_PARSER_API_SMOKE=1; these checks spend real API quota."""
import os

import pytest

from backend.app.config import get_settings
from backend.scripts.smoke_parser import run_smoke


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_parser_search_and_pdf_through_evidence_endpoint():
    if os.getenv('RUN_PARSER_API_SMOKE') != '1':
        pytest.skip('Set RUN_PARSER_API_SMOKE=1 to call real Parser API')
    report = await run_smoke(get_settings())
    if not report['ok']:
        # Deliberately use a safe summary, without locals/keys or provider body.
        failed = [name for name, ok in report['checks'].items() if not ok]
        pytest.fail('Real Parser API smoke failed: ' + ', '.join(failed or [report.get('error', 'source unavailable')]))
