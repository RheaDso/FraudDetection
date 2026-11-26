import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud.api.main import create_app
from fraud.api.scorer import STORE_FILE, Scorer
from fraud.config import PROJECT_ROOT, Config
from fraud.feature_store import FeatureStore
from fraud.features.history import CardState, HistoryStates, UidState

BODY: dict[str, Any] = {
    "TransactionID": 1,
    "TransactionDT": 6_000_000,
    "TransactionAmt": 120.5,
    "ProductCD": "W",
    "card1": 5_000,
    "addr1": 200,
    "D1": 10,
    "DeviceInfo": "Windows",
}


@pytest.fixture
def client(run_config: Config, tmp_path: Path) -> Iterator[TestClient]:
    store_path = tmp_path / STORE_FILE
    shutil.copy(run_config.paths.artifacts / STORE_FILE, store_path)
    store = FeatureStore(store_path)
    with TestClient(create_app(run_config, Scorer(run_config, store))) as test_client:
        yield test_client
    store.close()


def test_fr_10_1_store_keeps_state_exactly(tmp_path: Path) -> None:
    state = UidState()
    state.add(100, 0.1 + 0.2, 200.0, "d", "e.com")
    state.add(200, 1 / 3, None, None, None)
    store = FeatureStore(tmp_path / "s.sqlite")
    store.load(HistoryStates({"u": state}, {"1": CardState([100, 200])}))
    with store.transaction("u", "1") as (loaded, card):
        assert loaded == state
        assert card.times == [100, 200]
    assert store.row_count() == 2
    with pytest.raises(RuntimeError), store.transaction("u", "1") as (loaded, _):
        loaded.add(300, 5.0, None, None, None)
        raise RuntimeError("scoring failed")
    with store.transaction("u", "1") as (loaded, _):
        assert loaded == state  # nothing was saved
    store.close()
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "uid_state" in readme and "card1_state" in readme


def test_fr_10_2_valid_request_updates_store(client: TestClient) -> None:
    before = client.get("/health").json()["store_rows"]
    new = {**BODY, "card1": 99_999}
    assert client.post("/score", json=new).status_code == 200
    assert client.get("/health").json()["store_rows"] == before + 2


def test_fr_10_2_state_changes_between_calls(client: TestClient) -> None:
    scorer: Scorer = client.app.state.scorer  # type: ignore[attr-defined]
    first = scorer.features({**BODY, "card1": 88_888})
    second = scorer.features({**BODY, "card1": 88_888, "TransactionDT": 6_000_060})
    assert first["UID_TXN_COUNT_1H"].iat[0] == 0
    assert second["UID_TXN_COUNT_1H"].iat[0] == 1


def test_fr_10_2_missing_field_is_422(client: TestClient) -> None:
    body = {k: v for k, v in BODY.items() if k != "TransactionAmt"}
    assert client.post("/score", json=body).status_code == 422


@pytest.mark.parametrize(
    "change",
    [{"TransactionAmt": "lots"}, {"TransactionAmt": -1}, {"V1": "text"}, {"M1": 3}],
)
def test_fr_10_2_wrong_type_is_422(client: TestClient, change: dict[str, Any]) -> None:
    before = client.get("/health").json()["store_rows"]
    assert client.post("/score", json={**BODY, "card1": 77_777, **change}).status_code == 422
    assert client.get("/health").json()["store_rows"] == before


def test_fr_10_3_response_shape(client: TestClient, run_config: Config) -> None:
    body = client.post("/score", json=BODY).json()
    assert set(body) == {
        "fraud_probability",
        "expected_loss",
        "alert",
        "risk_level",
        "reasons",
        "model_version",
    }
    assert 0 <= body["fraud_probability"] <= 1
    assert body["expected_loss"] == pytest.approx(body["fraud_probability"] * 120.5)
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert len(body["reasons"]) == 3
    assert body["model_version"] == run_config.api.model_version
    decision = json.loads(
        (run_config.paths.artifacts / "decision.json").read_text(encoding="utf-8")
    )
    assert body["alert"] == (body["fraud_probability"] >= decision["fixed_threshold"])


def test_fr_10_4_train_serve_parity(run_config: Config) -> None:
    result = pd.read_csv(run_config.paths.reports / "api_checks.csv").iloc[0]
    assert result["parity_rows"] == run_config.api.parity_rows
    assert result["feature_mismatches"] == 0
    assert result["max_probability_difference"] <= run_config.api.parity_tolerance


def test_fr_10_5_health_and_reset(client: TestClient, run_config: Config) -> None:
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["store_rows"] > 0
    assert client.post("/store/reset").status_code == 403
    scorer: Scorer = client.app.state.scorer  # type: ignore[attr-defined]
    scorer.config = run_config.model_copy(
        update={"api": run_config.api.model_copy(update={"allow_reset": True})}
    )
    assert client.post("/store/reset").status_code == 200
    assert client.get("/health").json()["store_rows"] == 0


def test_fr_10_6_latency_is_recorded(run_config: Config) -> None:
    # The target is judged on the real run; a busy CI machine would make it flaky here.
    result = pd.read_csv(run_config.paths.reports / "api_checks.csv").iloc[0]
    assert result["latency_p95_ms"] > 0
    summary = (run_config.paths.reports / "RUN_SUMMARY.md").read_text(encoding="utf-8")
    assert "p95" in summary


def test_fr_10_7_dockerfile_serves_the_api_with_artifacts() -> None:
    # The image itself is built and probed in CI.
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY ${ARTIFACTS_DIR} ./artifacts" in dockerfile
    assert "fraud.api.main:app" in dockerfile
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "docker build" in workflow and "/health" in workflow
