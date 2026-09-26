"""Opt-in integration smoke through the API route using the REAL Parser API.

python -m backend.scripts.smoke_parser --output parser-smoke.json
No fake provider, fallback data, or automatic paid checks at startup.
"""
import argparse
import asyncio
import json
from pathlib import Path
from time import monotonic

import httpx

from backend.app.config import Settings, get_settings
from backend.app.llm.factory import get_query_planner, UnavailablePlanner
from backend.app.main import app
from backend.app.models import SearchPlan
from backend.app.providers.factory import get_court_provider
from backend.app.providers.parser_api import ParserApiProvider


async def run_smoke(settings: Settings, *, with_gigachat: bool = False) -> dict:
    started = monotonic()
    report = {'ok': False, 'provider': 'parser_api', 'checks': {}, 'key_present': bool(settings.parser_api_key)}
    if not settings.parser_api_key:
        report['error'] = 'PARSER_API_KEY is not configured'
        return report
    provider = ParserApiProvider(
        api_key=settings.parser_api_key, base_url=settings.parser_api_base_url,
        timeout_seconds=settings.parser_api_timeout_seconds, max_retries=settings.parser_api_max_retries,
    )
    previous = dict(app.dependency_overrides)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_court_provider] = lambda: provider
    app.dependency_overrides[get_query_planner] = lambda: UnavailablePlanner('Smoke uses a supplied plan')
    description = 'Налоговый орган исключил расходы по договору займа как экономически необоснованные.'
    plan = SearchPlan(queries=['расходы по договору займа', 'налог на прибыль проценты'], must_have=['налог'])
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://smoke.local') as client:
            if with_gigachat:
                if not settings.gigachat_auth_key:
                    report['error'] = 'GigaChat configuration is missing'
                    return report
                from backend.app.llm.gigachat import GigaChatClient, GigaChatQueryPlanner
                planner = GigaChatQueryPlanner(
                    GigaChatClient(settings.gigachat_auth_key, scope=settings.gigachat_scope,
                                   ca_bundle=settings.gigachat_ca_bundle),
                    model=settings.gigachat_plan_model,
                )
                app.dependency_overrides[get_query_planner] = lambda: planner
                planning_start = monotonic()
                response = await client.post('/api/cases/plan', json={'description': description})
                report['planning_seconds'] = round(monotonic() - planning_start, 3)
                report['checks']['gigachat_plan'] = response.status_code == 200
                if response.status_code != 200:
                    report['error'] = f'Planning HTTP {response.status_code}'
                    return report
                plan = SearchPlan.model_validate(response.json())
                # This smoke deliberately spends at most two search operations.
                plan.queries = plan.queries[:2]
            response = await asyncio.wait_for(client.post('/api/cases/search-with-evidence', json={
                'description': description, 'plan': plan.model_dump(mode='json'),
                'max_pages_per_query': 1, 'max_cases': 1,
            }), settings.search_timeout_seconds + 3)
            report['http_status'] = response.status_code
            if response.status_code != 200:
                report['error'] = 'Search endpoint failed'
                report['error_code'] = response.json().get('error_code')
                return report
            body = response.json()
            diagnostics = body.get('diagnostics') or {}
            report['diagnostics'] = diagnostics
            report['warnings'] = body.get('warnings', [])
            report['checks'].update({
                'two_queries_executed': diagnostics.get('search_calls') == 2,
                'real_http_attempts_recorded': diagnostics.get('http_attempts', 0) >= 3,
                'source_calls_succeeded': bool(diagnostics.get('events')) and all(
                    event['outcome'] == 'ok' for event in diagnostics.get('events', [])),
                'candidates_received': body.get('case_count', 0) > 0,
                'one_pdf_downloaded': diagnostics.get('pdf_downloaded') == 1,
                'one_pdf_extracted': body.get('cases_checked') == 1,
                'document_source_preserved': bool(body['items']) and all(
                    item.get('evidence_document') and
                    item['evidence_document']['case_id'] == item['case']['case_id']
                    for item in body['items']),
            })
            report['ok'] = all(report['checks'].values())
    except Exception as exc:
        # Never echo exception URLs, request headers, credential-bearing settings,
        # provider responses, or locals in a smoke report.
        report['error'] = type(exc).__name__
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)
        report['total_seconds'] = round(monotonic() - started, 3)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--with-gigachat', action='store_true')
    args = parser.parse_args()
    settings = Settings(_env_file=args.env_file) if args.env_file else get_settings()
    # Keep both configuration and output deterministic, even if old .env files
    # ask for three 120s attempts. This smoke tests the conservative profile.
    settings = settings.model_copy(update={'parser_api_timeout_seconds': 25., 'parser_api_max_retries': 1,
        'search_timeout_seconds': 90., 'retrieval_timeout_seconds': 45.})
    report = asyncio.run(run_smoke(settings, with_gigachat=args.with_gigachat))
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + '\n', encoding='utf-8')
    print(output)
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
