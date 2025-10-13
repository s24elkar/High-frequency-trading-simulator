"""Generate synthetic Poisson and Hawkes event streams for calibration tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from kernels import ExpKernel
    from simulate import simulate_poisson_process, simulate_thinning_exp_fast
else:  # pragma: no cover
    from ..kernels import ExpKernel
    from ..simulate import (
        simulate_poisson_process,
        simulate_thinning_exp_fast,
    )


def _write_dataset(
    base_name: str,
    times: np.ndarray,
    marks: np.ndarray,
    horizon: float,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{base_name}.csv"
    np.savetxt(
        csv_path,
        np.column_stack((times, marks)),
        delimiter=",",
        header="time,mark",
        comments="",
    )
    np.savez(
        out_dir / f"{base_name}.npz",
        times=times,
        marks=marks,
        horizon=horizon,
    )
    metadata = {
        "name": base_name,
        "event_count": int(times.size),
        "horizon": float(horizon),
        "mean_mark": float(np.mean(marks)) if marks.size else 0.0,
        "time_of_last_event": float(times[-1]) if times.size else None,
    }
    with (out_dir / f"{base_name}.meta.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic order-flow event streams."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--horizon", type=float, default=3_600.0)

    parser.add_argument("--poisson-mu", type=float, default=0.35)
    parser.add_argument(
        "--poisson-mark-scale",
        type=float,
        default=1.0,
        help="Scale parameter for exponential mark sampler (mean = scale).",
    )

    parser.add_argument("--hawkes-mu", type=float, default=0.2)
    parser.add_argument("--hawkes-alpha", type=float, default=0.6)
    parser.add_argument("--hawkes-beta", type=float, default=1.3)

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    def poisson_mark_sampler(local_rng: np.random.Generator) -> float:
        return float(local_rng.exponential(args.poisson_mark_scale))

    poisson_times, poisson_marks = simulate_poisson_process(
        args.poisson_mu,
        mark_sampler=poisson_mark_sampler,
        T=args.horizon,
        seed=args.seed,
    )
    _write_dataset(
        f"poisson_mu{args.poisson_mu:.2f}_seed{args.seed}",
        poisson_times,
        poisson_marks,
        args.horizon,
        args.output_dir,
    )

    kernel = ExpKernel(args.hawkes_alpha, args.hawkes_beta)

    def hawkes_mark_sampler(local_rng: np.random.Generator) -> float:
        return float(local_rng.lognormal(mean=0.0, sigma=0.4))

    hawkes_times, hawkes_marks = simulate_thinning_exp_fast(
        args.hawkes_mu,
        kernel,
        mark_sampler=hawkes_mark_sampler,
        T=args.horizon,
        seed=args.seed,
    )
    _write_dataset(
        f"hawkes_mu{args.hawkes_mu:.2f}_alpha{args.hawkes_alpha:.2f}_beta{args.hawkes_beta:.2f}_seed{args.seed}",
        hawkes_times,
        hawkes_marks,
        args.horizon,
        args.output_dir,
    )

    print("Synthetic datasets written to", args.output_dir.resolve())


if __name__ == "__main__":
    main()
