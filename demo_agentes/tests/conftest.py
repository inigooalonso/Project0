import pytest


@pytest.fixture
def settings():
    from tests.fakes import fake_settings

    return fake_settings()
