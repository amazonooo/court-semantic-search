import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import app
from backend.app.llm.factory import get_query_planner
from backend.app.providers.factory import get_court_provider
from backend.app.services import web_search
from backend.app.services.web_search import (
    YANDEX_SEARCH_URL, YandexWebSearch, UnavailableWebSearch, get_web_search, safe_web_url,
)
from backend.tests.test_semantic_search import FakePlanner, FakeProvider

DESCRIPTION = 'Покупатель приобрёл компанию, затем присоединился к приобретённой компании.'
SECRET = 'test-key-never-disclose'
ACT = {'title': 'Судебный акт', 'url': 'https://court.example/act', 'used': True}


async def preview(body, status=200):
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json=body))) as client:
        return await YandexWebSearch(SECRET, 'test-folder', client=client).search(DESCRIPTION)


@pytest.mark.asyncio
async def test_yandex_request_uses_official_search_contract_and_only_used_safe_sources():
    def handle(request):
        assert request.method == 'POST' and str(request.url) == YANDEX_SEARCH_URL
        assert request.headers['Authorization'] == 'Api-Key ' + SECRET
        body = json.loads(request.content)
        assert body['folderId'] == 'test-folder'
        assert body['searchType'] == 'SEARCH_TYPE_RU'
        assert body['getPartialResults'] is False
        assert len(body['messages']) == 1
        assert body['messages'][0]['role'] == 'ROLE_USER'
        assert body['messages'][0]['content'].endswith(DESCRIPTION)
        assert not {'site', 'host', 'url'} & body.keys()  # Search the whole index.
        return httpx.Response(200, json={
            'message': {'content': 'Найдено дело. [2]', 'role': 'ROLE_ASSISTANT'},
            'sources': [dict(ACT, used=False, url='https://court.example/unused'), ACT,
                        dict(ACT, url='javascript:alert(1)'), dict(ACT, url='https://user:pass@court.example/act')],
            'searchQueries': [{'text': ' приобретение компании суд '}, {'text': 'приобретение компании суд'},
                              {'reqId': 'ignored'}, None],
        })
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await YandexWebSearch(SECRET, 'test-folder', client=client).search(DESCRIPTION)
    assert result.status == 'found'
    assert [row.text for row in result.findings] == ['Найдено дело. [2]']
    assert [(source.url, source.reference_index) for source in result.findings[0].sources] == [(ACT['url'], 2)]
    assert result.queries == ['приобретение компании суд']
    assert SECRET not in result.model_dump_json()
    assert 'search_suggestions_html' not in result.model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize('extra', [
    {'isAnswerRejected': True}, {'problematicAnswer': True}, {'message': {}},
    {'sources': [dict(ACT, used=False)]}, {'sources': [{'title': ACT['title'], 'url': ACT['url']}]},
])
async def test_rejected_or_uncited_summary_keeps_pages_without_presenting_generated_claims(extra):
    body = {'message': {'content': 'Непроверенное утверждение.'}, 'sources': [ACT]}
    body.update(extra)
    result = await preview(body)
    assert result.status == 'found'
    assert result.findings[0].sources[0].url == ACT['url']
    assert result.findings[0].text == 'Найденные страницы'
    assert 'Непроверенное утверждение.' not in result.model_dump_json()
    assert result.message


@pytest.mark.asyncio
async def test_bullet_answer_is_explicitly_preliminary():
    result = await preview({'message': {'content': 'Обзор найденного.'}, 'sources': [ACT], 'isBulletAnswer': True})
    assert result.status == 'found' and result.message
    assert result.findings[0].text == 'Обзор найденного.'


@pytest.mark.asyncio
@pytest.mark.parametrize('body', [{}, {'message': {'content': 'Ничего не нашлось.'}},
                                 {'sources': [dict(ACT, url='data:text/html,unsafe')]},
                                 {'isAnswerRejected': True}])
async def test_optional_response_fields_and_no_usable_sources_are_empty(body):
    result = await preview(body)
    assert result.status == 'empty' and not result.findings
    assert result.message


@pytest.mark.asyncio
@pytest.mark.parametrize('body', [[], 'invalid', {'sources': {}}, {'searchQueries': 'invalid'},
                                 {'message': None}, {'message': {'content': 123}}])
async def test_malformed_provider_response_is_an_explicit_failure(body):
    result = await preview(body)
    assert result.status == 'error' and not result.findings


@pytest.mark.asyncio
@pytest.mark.parametrize('status, text', [(401, 'API-ключ'), (403, 'search-api.webSearch.user'),
                                        (429, 'лимит'), (500, 'недоступен')])
async def test_provider_errors_are_actionable_secret_free_and_do_not_retry(status, text):
    calls = []
    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(status, text=SECRET)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await YandexWebSearch(SECRET, 'test-folder', client=client).search(DESCRIPTION)
    assert result.status == 'error' and text in result.message
    assert SECRET not in result.model_dump_json()
    assert calls == [YANDEX_SEARCH_URL]


@pytest.mark.asyncio
async def test_search_deadline_is_bounded_and_does_not_expose_exception():
    async def handle(request):
        await asyncio.sleep(1)
        return httpx.Response(200, json={})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await YandexWebSearch(SECRET, 'test-folder', timeout=0.01, client=client).search(DESCRIPTION)
    assert result.status == 'error' and 'время' in result.message
    assert SECRET not in result.model_dump_json()


@pytest.mark.parametrize('search_key, old_key, folder, expected', [
    (' new-key ', 'old-key', ' folder ', 'new-key'),
    ('', ' old-key ', 'folder', 'old-key'),
    ('  ', 'old-key', 'folder', 'old-key'),
    (None, None, 'folder', None), ('new-key', None, None, None), ('new-key', None, '  ', None),
])
def test_factory_requires_folder_and_search_key_and_can_reuse_existing_yandex_key(
        monkeypatch, search_key, old_key, folder, expected):
    settings = Settings(_env_file=None, yandex_search_api_key=search_key,
                        yandex_api_key=old_key, yandex_folder_id=folder)
    monkeypatch.setattr(web_search, 'get_settings', lambda: settings)
    search = get_web_search()
    if expected:
        assert isinstance(search, YandexWebSearch)
        assert search._api_key == expected and search._folder_id == 'folder'
    else:
        assert isinstance(search, UnavailableWebSearch)


@pytest.mark.parametrize('url', ['javascript:alert(1)', 'file:///etc/passwd',
    'https://user:secret@court.example/act', 'https://court.example/a b', 'https://court.example/\nact',
    'https://[invalid', 'https://', None])
def test_unsafe_search_source_urls_are_not_links(url):
    assert not safe_web_url(url)


def test_preview_unavailable_or_failed_never_calls_court_source_or_blocks_plan():
    provider = FakeProvider()
    class FailingPreview(YandexWebSearch):
        async def _request(self, description):
            raise httpx.ConnectError(SECRET)
    app.dependency_overrides[get_court_provider] = lambda: provider
    app.dependency_overrides[get_query_planner] = FakePlanner
    try:
        with TestClient(app) as client:
            for search in [UnavailableWebSearch(), FailingPreview(SECRET, 'test-folder')]:
                app.dependency_overrides[get_web_search] = lambda: search
                result = client.post('/api/cases/web-preview', json={'description': DESCRIPTION})
                assert result.status_code == 200
                assert result.json()['status'] in {'unavailable', 'error'}
                assert SECRET not in result.text
                assert not provider.calls
                plan = client.post('/api/cases/plan', json={'description': DESCRIPTION})
                assert plan.status_code == 200 and plan.json()['queries']
                assert not provider.calls
    finally:
        app.dependency_overrides.clear()
