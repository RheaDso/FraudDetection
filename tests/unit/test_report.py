import json
from pathlib import Path

import pandas as pd

from fraud import report


def test_test_status_without_results(tmp_path: Path) -> None:
    assert "Not recorded" in report.test_status(tmp_path)


def test_test_status_reads_results_and_coverage(tmp_path: Path) -> None:
    (tmp_path / "test-results.xml").write_text(
        '<testsuites><testsuite tests="10" failures="1" errors="0" skipped="2" '
        'timestamp="2026-01-01T00:00:00"/></testsuites>',
        encoding="utf-8",
    )
    (tmp_path / "coverage.json").write_text(
        json.dumps({"totals": {"percent_covered": 91.25}}), encoding="utf-8"
    )
    text = report.test_status(tmp_path)
    assert "7 passed, 1 failed, 2 skipped" in text
    assert "91.2%" in text


def test_markdown_table_formats_missing_and_large_values() -> None:
    table = report.md_table(pd.DataFrame({"a|b": [float("nan"), 12345.6]}))
    assert "| - |" in table and "| 12,346 |" in table
