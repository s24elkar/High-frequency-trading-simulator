"""Writers for analytics artefacts with reproducible metadata envelopes."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _utc_timestamp() -> str:
    return (
        datetime.now(tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def detect_git_commit(repo_root: Path | None = None) -> str | None:
    """Best-effort discovery of the current git commit hash."""
    try:
        root = repo_root or Path.cwd()
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


@dataclass(slots=True)
class ReportMetadata:
    generator: str
    schema_version: str = "hftsim/v0"
    generated_at: str = field(default_factory=_utc_timestamp)
    git_commit: str | None = field(default=None)
    seed: int | None = field(default=None)
    notes: str | None = field(default=None)
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "generator": self.generator,
        }
        if self.git_commit:
            payload["git_commit"] = self.git_commit
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.notes:
            payload["notes"] = self.notes
        if self.extra:
            payload["extra"] = self.extra
        return payload


class ArtifactWriter:
    """Helper for writing JSON/CSV artefacts with companion metadata."""

    def __init__(
        self,
        base_dir: Path,
        metadata: ReportMetadata,
        *,
        overwrite: bool = True,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.metadata = metadata
        self.overwrite = overwrite
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative_path: str | Path) -> Path:
        path = self.base_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.overwrite and path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artefact: {path}")
        return path

    def write_json(
        self,
        relative_path: str | Path,
        payload: Mapping[str, object] | Sequence[object],
    ) -> Path:
        path = self._resolve(relative_path)
        envelope = {"meta": self.metadata.to_dict(), "data": payload}
        path.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        return path

    def _write_metadata_sidecar(self, artefact_path: Path) -> Path:
        meta_path = artefact_path.with_suffix(artefact_path.suffix + ".meta.json")
        if not self.overwrite and meta_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite metadata sidecar: {meta_path}"
            )
        meta_path.write_text(
            json.dumps(self.metadata.to_dict(), indent=2), encoding="utf-8"
        )
        return meta_path

    def write_csv(
        self,
        relative_path: str | Path,
        rows: Iterable[Mapping[str, object]],
        *,
        headers: Sequence[str] | None = None,
    ) -> Path:
        path = self._resolve(relative_path)
        materialised = list(rows)
        if not materialised:
            raise ValueError(f"No rows provided for CSV artefact {relative_path}")
        fieldnames: Sequence[str]
        if headers:
            fieldnames = headers
        else:
            # Preserve insertion order of keys from the first row
            fieldnames = tuple(materialised[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in materialised:
                writer.writerow(row)
        self._write_metadata_sidecar(path)
        return path

    def attach_metadata(
        self, artefact_path: str | Path, *, relative: bool = True
    ) -> Path:
        """Write metadata sidecar for an existing artefact."""
        path = Path(artefact_path)
        if relative:
            path = self.base_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            raise FileNotFoundError(
                f"Artefact does not exist for metadata attachment: {path}"
            )
        return self._write_metadata_sidecar(path)
