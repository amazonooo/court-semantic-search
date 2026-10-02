import httpx
import pytest

from backend.app.llm.gigachat import GIGACHAT_OAUTH_URL
from backend.scripts import audit_startup


@pytest.mark.asyncio
async def test_https_probe_preserves_the_real_oauth_port_without_credentials(monkeypatch):
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(404)
    original_client = httpx.AsyncClient
    monkeypatch.setattr(audit_startup.httpx, 'AsyncClient',
                        lambda **kwargs: original_client(transport=httpx.MockTransport(handle)))
    result = await audit_startup.probe_https('gigachat_oauth', GIGACHAT_OAUTH_URL)
    assert result['https_reachable'] and result['port'] == 9443
    assert len(requests) == 1
    assert str(requests[0].url) == 'https://ngw.devices.sberbank.ru:9443'
    assert 'authorization' not in requests[0].headers
