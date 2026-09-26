"""Read-only startup and optional HTTPS diagnostics. Never prints credential values."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
from time import monotonic
from urllib.parse import urlsplit

import httpx
import truststore

from backend.app.build_info import BUILD_ID, ROOT
from backend.app.config import get_settings
from backend.app.llm.gigachat import GIGACHAT_OAUTH_URL, GIGACHAT_CHAT_URL


def local_report():
    s = get_settings()
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    return {
        'build_id': BUILD_ID, 'source_root': str(ROOT), 'cwd': str(Path.cwd()),
        'python': sys.version.split()[0], 'python_executable': sys.executable,
        'git_head': head, 'env_file': str(ROOT / '.env'), 'env_file_exists': (ROOT / '.env').is_file(),
        'court_provider': s.court_provider, 'llm_provider': s.llm_provider,
        'gigachat_plan_model': s.gigachat_plan_model,
        'gigachat_relevance_model': s.gigachat_relevance_model,
        'gigachat_scope': s.gigachat_scope,
        'credentials_present': {
            'PARSER_API_KEY': bool(s.parser_api_key), 'GIGACHAT_AUTH_KEY': bool(s.gigachat_auth_key),
        },
        'proxy_variables_present': {k: bool(os.getenv(k)) for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY']},
        'limits': {k: getattr(s, k) for k in [
            'parser_api_timeout_seconds','parser_api_max_retries','search_timeout_seconds',
            'retrieval_timeout_seconds','plan_timeout_seconds','search_max_queries',
            'search_max_pages_per_query','search_max_cases','search_max_pdf_downloads',
        ]},
    }


async def probe_https(label, url, *, ca_bundle=None):
    host = urlsplit(url).hostname
    events = []
    async def trace(name, info):
        # Trace info can contain headers and payload. Store event names only.
        if name.endswith(('.started', '.complete', '.failed')):
            events.append(name)
    started = monotonic()
    report = {'service': label, 'host': host}
    try:
        ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        if ca_bundle:
            ssl_context.load_verify_locations(cafile=ca_bundle)
        async with httpx.AsyncClient(timeout=8, verify=ssl_context) as client:
            response = await asyncio.wait_for(client.get('https://' + host, extensions={'trace': trace}), 10)
        report['http_status'] = response.status_code
        report['https_reachable'] = True  # even 404 proves a completed handshake
    except Exception as exc:
        report.update(https_reachable=False, error_type=type(exc).__name__)
    report.update(seconds=round(monotonic() - started, 3), stages=events)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', action='store_true', help='Credential-free HTTPS probes')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = local_report()
    if args.network:
        async def probes():
            settings = get_settings()
            return await asyncio.gather(probe_https('parser_api', settings.parser_api_base_url),
                                        probe_https('gigachat_oauth', GIGACHAT_OAUTH_URL,
                                                    ca_bundle=settings.gigachat_ca_bundle),
                                        probe_https('gigachat_chat', GIGACHAT_CHAT_URL,
                                                    ca_bundle=settings.gigachat_ca_bundle))
        report['network'] = asyncio.run(probes())
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)
    raise SystemExit(1 if any(not p['https_reachable'] for p in report.get('network', [])) else 0)


if __name__ == '__main__':
    main()
