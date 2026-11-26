"""Model report and run summary (M11). Every number comes from computed outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import markdown
import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from fraud.config import Config

TEMPLATES = Path(__file__).resolve().parent / "templates"


def md_table(frame: pd.DataFrame) -> str:
    """Render a frame as a Markdown table."""

    def cell(value: Any) -> str:
        if isinstance(value, float):
            if pd.isna(value):
                return "-"
            return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4g}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.astype(object).itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])


def test_status(directory: Path) -> str:
    """Summarise the last ``make test`` run from the files it left in ``directory``."""
    results, coverage = directory / "test-results.xml", directory / "coverage.json"
    if not (results.is_file() and coverage.is_file()):
        return "Not recorded. Run `make test`, then the pipeline again."
    suite = ElementTree.parse(results).getroot().find("testsuite")
    assert suite is not None
    count = {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
    failed = count["failures"] + count["errors"]
    percent = json.loads(coverage.read_text(encoding="utf-8"))["totals"]["percent_covered"]
    return (
        f"{count['tests'] - failed - count['skipped']} passed, {failed} failed, "
        f"{count['skipped']} skipped; coverage {percent:.1f}% "
        f"(last test run: {suite.get('timestamp', 'unknown')})."
    )


test_status.__test__ = False  # type: ignore[attr-defined]


def targets_table(
    metrics: pd.DataFrame,
    budget: pd.DataFrame,
    best: pd.Series,
    psi: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Actual against target for section 12.3."""
    test = metrics[(metrics["split"] == "test") & (metrics["model"] == "lightgbm")].iloc[0]
    at_k = budget[budget["k"] == config.budget.default_k]
    rules = at_k[at_k["model"] == "rules"].iloc[0]
    uplift = (best["vdr"] - rules["vdr"]) * 100
    drift = psi.loc[psi["comparison"] == "train vs test", "psi"].iloc[0]
    # Compare like with like: a threshold policy that raises fewer than K alerts would
    # otherwise win on precision just by alerting less.
    other_model = (at_k["model"] != best["model"]) & (at_k["policy"] == best["policy"])
    baseline = at_k[(at_k["model"] == "rules") | other_model]
    t = config.targets
    rows = [
        ("LightGBM PR-AUC", f"≥ {t.pr_auc}", f"{test['pr_auc']:.3f}", test["pr_auc"] >= t.pr_auc),
        (
            "LightGBM ROC-AUC",
            f"≥ {t.roc_auc}",
            f"{test['roc_auc']:.3f}",
            test["roc_auc"] >= t.roc_auc,
        ),
        (
            "VDR uplift vs rules",
            f"≥ {t.vdr_uplift_vs_rules_pp} pp",
            f"{uplift:.1f} pp",
            uplift >= t.vdr_uplift_vs_rules_pp,
        ),
        (
            "Net benefit vs rules",
            f"> ₹{rules['net_benefit']:,.0f}",
            f"₹{best['net_benefit']:,.0f}",
            best["net_benefit"] > rules["net_benefit"],
        ),
        (
            "Precision@K",
            "not beaten by a baseline",
            f"{best['precision']:.1%} (best baseline {baseline['precision'].max():.1%})",
            best["precision"] >= baseline["precision"].max(),
        ),
        (
            "Score PSI train vs test",
            f"< {config.evaluation.psi_alert}",
            f"{drift:.3f}",
            drift < config.evaluation.psi_alert,
        ),
    ]
    table = pd.DataFrame(rows, columns=["metric", "target", "actual", "met"])
    table["met"] = table["met"].map({True: "yes", False: "NO"})
    return table


def render(context: dict[str, Any], reports: Path) -> None:
    """Write model_report.md, model_report.html and RUN_SUMMARY.md."""
    environment = Environment(
        loader=FileSystemLoader(TEMPLATES),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters["table"] = md_table
    report = environment.get_template("model_report.md.j2").render(**context)
    (reports / "model_report.md").write_text(report, encoding="utf-8")
    body = markdown.markdown(report, extensions=["tables"])
    page = environment.get_template("page.html.j2").render(body=body)
    (reports / "model_report.html").write_text(page, encoding="utf-8")
    summary = environment.get_template("run_summary.md.j2").render(**context)
    (reports / "RUN_SUMMARY.md").write_text(summary, encoding="utf-8")
