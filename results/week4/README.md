# Week 4 Results Layout

- `perf/` — deterministic throughput benchmarks produced by `python/scripts/run_benchmarks.py`.
- `stress/` — stress-suite outputs (`run_stress_suite.py`) plus aggregated metrics and risk checks.
- `plots/` — published figures referenced in the weekly report; metadata sidecars live alongside each image.

All artefacts are written via `python.analysis.ArtifactWriter`, ensuring per-file metadata (`*.meta.json`) with generator, seed, and commit information.
