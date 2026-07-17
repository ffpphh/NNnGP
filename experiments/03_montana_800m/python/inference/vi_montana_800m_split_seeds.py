"""Run Montana 800m PRISM VI for ten reproducible train/test splits.

The spatial field is saved once without a split column.  A separate wide CSV
stores one complete point-index ordering per random seed; its first 800 entries
are maximin-ordered training points and the remaining entries are prediction
points.  Seed 43 reuses the existing split and ordering exactly. Each seed gets
its own posterior files. Tail-event metrics are computed separately by
``summarize_montana_800m_split_seeds.py``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[4]
CORE_DIR = PACKAGE_ROOT / "core"
DATA_PREPARATION_DIR = EXPERIMENT_DIR / "python" / "data_preparation"
for search_dir in (CORE_DIR, DATA_PREPARATION_DIR):
    if str(search_dir) not in sys.path:
        sys.path.insert(0, str(search_dir))

from build_montana_prism_800m import VALUE_COL, maximin_order
from vi_inference import build_param_mode, parse_csv_list, run_one_simulation


BASE_DIR = Path(__file__).resolve().parent
RUN_NAME = "prism_montana_october_ppt_2025_800m_split_seeds"
DEFAULT_SEEDS = tuple(range(43, 53))
DEFAULT_DATA_CSV = EXPERIMENT_DIR / "data" / "source" / "prism_october_ppt_2025_log_standardized_800m_downsampled_40k_mt_train800.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / "nnngp"
ORIGINAL_DATA_FILENAME = "montana_800m_original_data.csv"
ORDER_TABLE_FILENAME = "montana_800m_split_point_orders.csv"
SEED_RESULTS_DIRNAME = "seed_results"


def default_gpu_ids() -> str:
    configured = os.environ.get("MONTANA_GPU_IDS") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return ",".join(line.strip() for line in result.stdout.splitlines() if line.strip())


def default_parallel_workers() -> int:
    gpu_ids = [item.strip() for item in default_gpu_ids().split(",") if item.strip()]
    return max(1, len(gpu_ids))


def parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"--seeds contains duplicates: {seeds}")
    return seeds


def prepare_split_csvs(
    source_csv: Path,
    data_dir: Path,
    split_dir: Path,
    seeds: tuple[int, ...],
    train_size: int,
    reuse_source_seed: int = 43,
) -> dict[int, Path]:
    source = pd.read_csv(source_csv)
    required = {"lon", "lat", "split", VALUE_COL}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Source CSV is missing columns: {sorted(missing)}")
    if source[["lon", "lat"]].duplicated().any():
        raise ValueError("Source CSV contains duplicate lon/lat locations")

    data_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    original = source[["lon", "lat", VALUE_COL]].copy().reset_index(drop=True)
    original_path = data_dir / ORIGINAL_DATA_FILENAME
    original.to_csv(original_path, index=False)

    paths: dict[int, Path] = {}
    point_orders: list[np.ndarray] = []
    # Always retain the canonical ten rows in the persistent order table, even
    # when --seeds requests only a subset for inference.
    order_seeds = tuple(dict.fromkeys((*DEFAULT_SEEDS, *seeds)))
    for seed in order_seeds:
        output_csv = split_dir / f"montana_800m_train{train_size}_seed{seed}.csv"
        if seed == reuse_source_seed:
            n_train = int(source["split"].eq("S").sum())
            if n_train != train_size:
                raise ValueError(
                    f"Seed {seed} source split has {n_train} "
                    f"training rows, expected {train_size}"
                )
            source_s = source.index[source["split"].eq("S")].to_numpy(dtype=np.int64)
            source_u = source.index[source["split"].eq("U")].to_numpy(dtype=np.int64)
            point_order = np.concatenate([source_s, source_u])
        else:
            # Sampling is based only on the fixed original row indices.  Maximin
            # changes the order of S, not which locations were sampled.
            sampled = original.sample(n=train_size, random_state=seed).index.to_numpy(dtype=np.int64)
            local_order = maximin_order(original.loc[sampled, ["lon", "lat"]].to_numpy(dtype=float))
            ordered_s = sampled[local_order]
            is_s = np.zeros(len(original), dtype=bool)
            is_s[sampled] = True
            ordered_u = np.flatnonzero(~is_s)
            point_order = np.concatenate([ordered_s, ordered_u])

        if len(point_order) != len(original) or len(np.unique(point_order)) != len(original):
            raise ValueError(f"Seed {seed} did not produce a permutation of all point indices")
        ordered = original.iloc[point_order].copy()
        ordered.insert(2, "split", np.where(np.arange(len(ordered)) < train_size, "S", "U"))
        split = ordered[["lon", "lat", "split", VALUE_COL]]
        if seed in seeds:
            split.to_csv(output_csv, index=False)
            paths[seed] = output_csv
        point_orders.append(point_order)

    order_columns = [f"point_index_{idx:05d}" for idx in range(len(original))]
    order_table = pd.DataFrame(np.vstack(point_orders), columns=order_columns)
    order_table.insert(0, "random_seed", order_seeds)
    order_path = data_dir / ORDER_TABLE_FILENAME
    order_table.to_csv(order_path, index=False)
    print(
        f"Prepared data: {len(seeds)} requested seeds, {train_size} S + "
        f"{len(original) - train_size} U per seed"
    )
    return paths


def inference_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        iters=args.iters,
        lr=args.lr,
        particles=args.particles,
        vi_samples=args.vi_samples,
        flows=args.flows,
        guide=args.guide,
        optimize_z=args.optimize_z,
        resume_from=None,
        pred_max_samples=args.pred_max_samples,
        compare_hmc=None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--gpu-ids", default=default_gpu_ids(), help="Comma-separated physical GPU IDs.")
    parser.add_argument("--parallel-workers", type=int, default=default_parallel_workers())
    parser.add_argument("--worker-seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-data", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip a seed when both posterior result NPZ files already exist.",
    )
    parser.add_argument("--preset", default="all_eb", choices=["all_fixed", "all_eb", "fixed_base", "fixed_base_fixed_noise"])
    parser.add_argument("--fixed", default="")
    parser.add_argument("--eb", default="")
    parser.add_argument("--guide", default=os.environ.get("VI_GUIDE", "flow"), choices=["flow", "lowrank", "diagonal"])
    parser.add_argument("--iters", type=int, default=int(os.environ.get("VI_ITERS", "15000")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("VI_LR", "0.0002")))
    parser.add_argument("--particles", type=int, default=int(os.environ.get("VI_PARTICLES", "1")))
    parser.add_argument("--vi-samples", type=int, default=int(os.environ.get("VI_SAMPLES", "1000")))
    parser.add_argument("--flows", type=int, default=int(os.environ.get("VI_FLOWS", "3")))
    parser.add_argument("--hidden-dims", default=os.environ.get("VI_HIDDEN_DIMS", "1024,1024"))
    parser.add_argument("--pred-max-samples", default=os.environ.get("PRED_MAX_SAMPLES", "1000"))
    parser.set_defaults(optimize_z=True)
    parser.add_argument("--fixed-z", dest="optimize_z", action="store_false")
    parser.add_argument("--optimize-z", dest="optimize_z", action="store_true")
    return parser


def run_seed(seed: int, data_path: Path, args: argparse.Namespace, param_mode, hidden_dims) -> None:
    result_dir = args.output_dir / SEED_RESULTS_DIRNAME / f"seed_{seed}"
    completed = (result_dir / "vi_results.npz").exists() and (
        result_dir / "vi_prediction_results.npz"
    ).exists()
    if args.skip_completed and completed:
        print(f"Skipping completed seed {seed}: {result_dir}")
        return
    print(f"\nRunning Montana 800m split seed {seed}")
    run_one_simulation(
        run_kind=f"prism_montana_october_ppt_2025_800m_seed_{seed}",
        data_path=str(data_path),
        save_dir=str(result_dir),
        args=inference_args(args),
        param_mode=param_mode,
        hidden_dims=hidden_dims,
        z_seed=42,
    )


def worker_command(seed: int, data_path: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-seed", str(seed),
        "--worker-data", str(data_path),
        "--output-dir", str(args.output_dir),
        "--preset", args.preset,
        "--fixed", args.fixed,
        "--eb", args.eb,
        "--guide", args.guide,
        "--iters", str(args.iters),
        "--lr", str(args.lr),
        "--particles", str(args.particles),
        "--vi-samples", str(args.vi_samples),
        "--flows", str(args.flows),
        "--hidden-dims", args.hidden_dims,
        "--pred-max-samples", str(args.pred_max_samples),
        "--parallel-workers", "1",
    ]
    cmd.append("--optimize-z" if args.optimize_z else "--fixed-z")
    if args.skip_completed:
        cmd.append("--skip-completed")
    return cmd


def run_parallel(
    seeds: tuple[int, ...],
    split_paths: dict[int, Path],
    args: argparse.Namespace,
) -> None:
    gpu_ids = [item.strip() for item in str(args.gpu_ids).split(",") if item.strip()]
    if not gpu_ids:
        gpu_ids = [os.environ.get("CUDA_VISIBLE_DEVICES", "")]
    max_workers = max(1, int(args.parallel_workers))
    pending = list(seeds)
    running: list[tuple[subprocess.Popen, int, str, object, Path]] = []
    launched = 0
    completed = 0
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Progress: 0/{len(seeds)} complete | workers={max_workers} | GPUs={','.join(gpu_ids) or 'default'}")

    while pending or running:
        while pending and len(running) < max_workers:
            seed = pending.pop(0)
            gpu_id = gpu_ids[launched % len(gpu_ids)]
            env = os.environ.copy()
            if gpu_id:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
            cmd = worker_command(seed, split_paths[seed], args)
            log_path = log_dir / f"seed_{seed}.log"
            log_file = log_path.open("w", encoding="utf-8")
            print(f"  start seed={seed} GPU={gpu_id or 'default'}")
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            running.append((process, seed, gpu_id, log_file, log_path))
            launched += 1

        still_running = []
        for process, seed, gpu_id, log_file, log_path in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((process, seed, gpu_id, log_file, log_path))
            elif return_code != 0:
                log_file.close()
                for other, other_seed, _, other_log, _ in running:
                    if other_seed != seed and other.poll() is None:
                        other.terminate()
                    other_log.close()
                raise RuntimeError(
                    f"Seed {seed} failed on GPU {gpu_id} with exit code {return_code}; "
                    f"see {log_path}"
                )
            else:
                log_file.close()
                completed += 1
                print(
                    f"  done  seed={seed} GPU={gpu_id or 'default'} | "
                    f"progress={completed}/{len(seeds)}"
                )
        running = still_running
        if running:
            time.sleep(5)


def main() -> None:
    warnings.filterwarnings("ignore")
    args = build_parser().parse_args()
    args.source_csv = args.source_csv.resolve()
    args.output_dir = args.output_dir.resolve()
    seeds = parse_seeds(args.seeds)
    hidden_dims = tuple(int(value) for value in parse_csv_list(args.hidden_dims)) or (128, 128)
    param_mode = build_param_mode(args.preset, fixed_text=args.fixed, eb_text=args.eb)

    if args.worker_seed is not None:
        if args.worker_data is None:
            raise ValueError("--worker-data is required with --worker-seed")
        run_seed(args.worker_seed, args.worker_data.resolve(), args, param_mode, hidden_dims)
        return

    data_dir = args.output_dir / "data_splits"
    with tempfile.TemporaryDirectory(prefix="montana_800m_splits_") as tmp_dir:
        split_paths = prepare_split_csvs(
            args.source_csv,
            data_dir,
            Path(tmp_dir),
            seeds,
            args.train_size,
        )
        if args.prepare_only:
            print("Prepared original-data and point-order tables only; inference was not run.")
            return

        # Use the worker subprocess path even with one worker so detailed JAX/VI
        # output consistently goes to per-seed log files.
        run_parallel(seeds, split_paths, args)

    print(f"All split-seed results saved under: {args.output_dir}")
    print("Run summarize_montana_800m_split_seeds.py separately to compute tail metrics.")


if __name__ == "__main__":
    main()
