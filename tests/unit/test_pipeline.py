from pathlib import Path

import pytest

from fraud import pipeline
from fraud.config import load_config


def test_check_data_fails_and_explains_when_files_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert pipeline.check_data(load_config(root=tmp_path)) == 1
    error = capsys.readouterr().err
    assert "train_transaction.csv" in error
    assert "Download steps" in error


def test_check_data_passes_when_files_present(tmp_path: Path) -> None:
    config = load_config(root=tmp_path)
    config.paths.raw.mkdir(parents=True)
    for name in config.data.required_files:
        (config.paths.raw / name).touch()
    assert pipeline.check_data(config) == 0


def test_main_dispatches_check_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline, "load_config", lambda: load_config(root=tmp_path))
    assert pipeline.main(["check-data"]) == 1


def test_main_run_stops_without_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pipeline, "load_config", lambda: load_config(root=tmp_path))
    assert pipeline.main(["run", "--trials", "1"]) == 1
