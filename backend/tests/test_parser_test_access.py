import httpx
import pytest

from backend.app.providers.base import CourtProviderAccessError
from backend.app.providers.parser_api import ParserApiProvider


@pytest.mark.asyncio
async def test_disabled_trial_names_the_source_and_never_retries_access_denial():
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(403, json={'error': 'Test access is disabled for this account'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = ParserApiProvider(api_key='private-test-key', base_url='https://parser.example',
                                     max_retries=3, client=client)
        with pytest.raises(CourtProviderAccessError) as error:
            await provider._request_json('search', {'text': 'компания'})
    assert 'Parser API' in str(error.value) and 'тестовый доступ' in str(error.value)
    assert 'ras.arbitr.ru' in str(error.value)
    assert 'private-test-key' not in str(error.value)
    assert len(requests) == 1
