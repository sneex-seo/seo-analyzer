import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_site import start_server  # noqa: E402

from seo_analyzer.config import Settings  # noqa: E402


@pytest.fixture(scope="session")
def server():
    srv, port = start_server()
    yield port
    srv.shutdown()


@pytest.fixture
def offline_settings(tmp_path):
    s = Settings()
    s.embedding_provider = "lsa"
    s.cache_path = tmp_path / "cache.sqlite"
    s.crawl_delay = 0
    return s
