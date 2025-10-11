# Release Notes

## v0.9-beta — Week 4 Baseline

### Core Simulation Stability
- Deterministic stress harness rerun with seed 2024; synthetic replay validates sequencing (no orphan cancels/executes, consistent digests).
- Benchmarks and stress artefacts tagged with schema `hftsim/v0` + commit hash for traceability.

### Profiling Improvements
- Introduced `python/analysis` for shared profiling, IO, and plotting utilities; `stress.py` now consumes the new cProfile context and hotspot extraction.
- Benchmarks writes derive from `ArtifactWriter`, yielding JSON/CSV pairs and standardised plots under `results/week4/`.
- Added `scripts/run_ci_checks.sh` to mirror CI lint/test steps locally with deterministic Matplotlib caching.

### Stress-Test Outcomes
- Poisson/burst suite achieves 21–22.5k msg/s with ~30 µs latency across multipliers; latency histograms and summaries exported with metadata.
- Architecture comparison confirms single-thread loop remains ~75% faster than the concurrent runner; all repetitions share digest `79067c52…`.

### Distribution
- Tag recommendation: `git tag -a v0.9-beta -m "Week 4 validated baseline" && git push origin v0.9-beta` once artefacts are committed.
