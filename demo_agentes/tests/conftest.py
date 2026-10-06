import os

import pytest

# Sin latencias simuladas en los tests.
os.environ.setdefault("ADA_SPEED", "0")


@pytest.fixture(scope="session")
def settings():
    from core.settings import load_settings

    return load_settings().with_overrides(speed=0.0, agent_mode="mock", rag_mode="mock", executor_mode="mock")


@pytest.fixture(scope="session")
def services(settings):
    from services.factory import build_services

    return build_services(settings)


@pytest.fixture(scope="session")
def scenarios():
    from services.scenarios import load_scenarios

    return load_scenarios()
