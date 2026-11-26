from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fraud.config import DEFAULT_CONFIG_PATH, load_config


def _write(tmp_path: Path, section: str, key: str, value: object) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw[section][key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_default_config_loads_with_absolute_paths() -> None:
    config = load_config()
    assert config.seed == 42
    assert config.budget.default_k == 100
    assert config.paths.raw.is_absolute()
    assert config.rules.file.is_file()


def test_paths_resolve_against_given_root(tmp_path: Path) -> None:
    config = load_config(root=tmp_path)
    assert config.paths.raw == tmp_path / "data" / "raw"


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("split", "train", 0.7),
        ("budget", "default_k", 0),
        ("costs", "recovery_rate", 1.5),
        ("costs", "typo_key", 1),
        ("models", "optuna_trials", 41),
        ("api", "risk_levels", {"medium": 0.5, "high": 0.3}),
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, section: str, key: str, value: object) -> None:
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, section, key, value))
