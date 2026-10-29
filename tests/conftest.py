"""Shared test cleanup."""

import pytest

from src import agents
from src.data_tools import clear_datasets


@pytest.fixture(autouse=True)
def clean_runtime():
    agents.NODE_DELAY_SECONDS = 0
    clear_datasets()
    yield
    agents.NODE_DELAY_SECONDS = 0
    clear_datasets()
