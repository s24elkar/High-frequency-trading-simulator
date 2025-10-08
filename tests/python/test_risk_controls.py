from __future__ import annotations

from pathlib import Path

from python.scripts.check_logs import analyse_directory
from python.scripts.run_risk_controls import run_suite as run_risk_controls_suite


def test_risk_controls_suite(tmp_path: Path) -> None:
    results = run_risk_controls_suite(tmp_path)

    assert results["order_rate_limit"]["strategy_halted"] is True
    assert results["cancel_rate_limit"]["strategy_halted"] is False
    assert results["cancel_rate_limit"]["throttled_order_active"] is True
    assert results["kill_switch"]["strategy_halted"] is True
    output_file = tmp_path / "risk_controls.json"
    assert output_file.exists()

    report = analyse_directory(tmp_path / "logs")
    for entry in report["logs"]:
        filename = Path(entry["file"]).name
        if "kill_switch" in filename:
            assert not entry["sequence"]["ok"]
            assert entry["sequence"]["orphan_executes"] >= 1
        else:
            assert entry["sequence"]["ok"]
