from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from python.backtester import MarketEvent, SequenceValidator


def _map_event_type(event_type: str) -> str | None:
    if event_type == "order":
        return "add_order"
    if event_type == "cancel":
        return "delete_order"
    if event_type == "fill":
        return "execute_order"
    return None


def analyse_log(path: Path) -> Dict[str, object]:
    validator = SequenceValidator()
    control_events: List[Dict[str, object]] = []
    total_records = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total_records += 1
            record = json.loads(line)
            event_type = record.get("event_type")
            if event_type == "control_violation":
                payload = record.get("payload", {})
                control_events.append(payload)
                continue
            mapped = _map_event_type(event_type)
            if mapped is None:
                continue
            payload = record.get("payload", {})
            event = MarketEvent(
                timestamp_ns=int(record.get("timestamp_ns", 0)),
                event_type=mapped,
                payload=payload,
            )
            validator.observe(event)
    report = validator.report()
    return {
        "file": str(path),
        "total_records": total_records,
        "sequence": {
            "ok": report.ok,
            "total_events": report.total_events,
            "timestamp_monotonic": report.timestamp_monotonic,
            "orphan_cancels": report.orphan_cancels,
            "orphan_executes": report.orphan_executes,
            "duplicate_order_ids": report.duplicate_order_ids,
            "max_timestamp_gap_ns": report.max_timestamp_gap_ns,
            "errors": [asdict(err) for err in report.errors],
        },
        "control_events": control_events,
    }


def analyse_directory(log_dir: Path) -> Dict[str, object]:
    reports: List[Dict[str, object]] = []
    for path in sorted(log_dir.rglob("*.jsonl")):
        reports.append(analyse_log(path))
    return {"logs": reports}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse JSONL logs for integrity")
    parser.add_argument("--log-dir", type=Path, required=True, help="Directory of JSONL logs")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the integrity report JSON.",
    )
    args = parser.parse_args()
    report = analyse_directory(args.log_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
