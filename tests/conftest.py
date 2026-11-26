"""Shared fixtures. One small synthetic pipeline run backs the output and API tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from fraud import pipeline
from fraud.config import Config, load_config


def small_config(root: Path) -> Config:
    """A config whose data lives under ``root`` and whose synthetic run is quick."""
    config = load_config(root=root)
    smoke = config.smoke.model_copy(
        update={"transactions": 8_000, "uids": 600, "parity_rows": 100, "latency_requests": 50}
    )
    return config.model_copy(update={"smoke": smoke})


@pytest.fixture(scope="session")
def run_config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Run the whole pipeline once and return the config pointing at its outputs."""
    work = tmp_path_factory.mktemp("smoke")
    # smoke() relocates paths under the work directory, starting from the project root.
    return pipeline.smoke(small_config(pipeline.PROJECT_ROOT), work)
