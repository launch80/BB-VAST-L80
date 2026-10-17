import pytest
import respx

from bbvastl80.vast.client import VastClient

# Full base used by the client. Tests register full URLs to avoid any
# base-URL path-joining ambiguity in the mock.
BASE = "https://console.vast.ai/api/v0"
BASE_V1 = "https://console.vast.ai/api/v1"


@pytest.fixture
def router():
    with respx.mock(assert_all_called=False) as rsp:
        yield rsp


@pytest.fixture
def client(router):
    return VastClient("test-key", base_url=BASE)
