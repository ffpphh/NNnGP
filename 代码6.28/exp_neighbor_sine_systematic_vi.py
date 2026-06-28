"""Systematic VI-NF experiment for the exp_neighbor_sine dataset.

This script keeps the generated data fixed and varies only the ordered reference
set and the NNGP neighbor count.  It runs 10 random reference orders crossed with
m = 2, 4, ..., 20 by default, saving compact posterior samples for each combo.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from vi_inference import (
    build_param_mode,
    load_npz_data,
    prepare_fit_data,
    run_vi_inference,
    save_vi_metric_summary,
)
from inference_utils import make_parent_matern_kernel
from model import (
    build_neighbor_indices,
    compute_sparse_reverse_cholesky,
    precompute_eb_reference_geometry,
    precompute_fic_reference_geometry,
    predict,
    use_noncentered_from_param_mode,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = (
    BASE_DIR
    / "results"
    / "simulation_outputs"
    / "exp_neighbor_sine"
    / "exp_neighbor_sine_data.npz"
)
DEFAULT_SAVE_DIR = BASE_DIR / "results" / "vi_results" / "exp_neighbor_sine_systematic_vi_nf"
DEFAULT_M_VALUES = tuple(range(4, 22, 2))
M_TILDE_OVERRIDES = {4: 20}


def default_gpu_ids() -> str:
    configured = os.environ.get("SYSTEMATIC_GPU_IDS") or os.environ.get("CUDA_VISIBLE_DEVICES")
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

    gpu_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ",".join(gpu_ids)


def default_parallel_workers() -> int:
    gpu_ids = [item.strip() for item in default_gpu_ids().split(",") if item.strip()]
    return max(1, len(gpu_ids))


def parse_int_list(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def combo_name(order_id: int, m: int) -> str:
    return f"order{order_id:02d}_m{m:02d}"


def combo_label(order_id: int, m: int) -> str:
    return f"{order_id}-{m}"


def m_tilde_for_m(m: int, default_m_tilde: int, small_m_tilde: int) -> int:
    overrides = dict(M_TILDE_OVERRIDES)
    overrides[4] = int(small_m_tilde)
    return int(overrides.get(int(m), default_m_tilde))


def make_order_specs(S: np.ndarray, order_count: int, seed: int) -> list[dict]:
    """Return maximin, coordinate, then random reference orders."""
    S = np.asarray(S, dtype=np.float64)
    rng = np.random.default_rng(seed)
    specs = [
        {"order_type": "maximin", "permutation": np.arange(len(S), dtype=int)},
        {"order_type": "x1_x2", "permutation": np.lexsort((S[:, 1], S[:, 0])).astype(int)},
    ]
    for random_id in range(max(0, int(order_count) - len(specs))):
        specs.append(
            {
                "order_type": f"random{random_id + 1}",
                "permutation": rng.permutation(len(S)).astype(int),
            }
        )
    return specs[: int(order_count)]


def ordered_training_data(data: dict, permutation: np.ndarray) -> dict:
    out = dict(data)
    for key in ("S", "X_S", "y_S", "s_indices"):
        if key in out:
            out[key] = np.asarray(out[key])[permutation]
    return out


def write_ordered_csv(path: Path, data: dict, order_specs: list[dict]) -> None:
    """Write one R-friendly CSV containing every reference order."""
    all_points = np.asarray(data["all_points"], dtype=np.float64)
    X_all = np.asarray(data["X_all"], dtype=np.float64)
    y_all = np.asarray(data["y_all"], dtype=np.float64)
    s_indices_original = np.asarray(data["s_indices"], dtype=int)
    u_indices = np.asarray(data["u_indices"], dtype=int)

    split = np.full(len(all_points), "U", dtype=object)
    split[s_indices_original] = "S"

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "order_id",
                "order_type",
                "row_order",
                "point_index",
                "split",
                "reference_order",
                "x",
                "y",
                "w",
                "y_obs",
                "x0",
                "x1",
                "x2",
            ]
        )
        for order_id, spec in enumerate(order_specs, start=1):
            permutation = np.asarray(spec["permutation"], dtype=int)
            order_type = spec["order_type"]
            s_indices_ordered = s_indices_original[permutation]
            reference_order = {int(point_index): row for row, point_index in enumerate(s_indices_ordered)}
            ordered_indices = np.concatenate([s_indices_ordered, u_indices])
            for row_order, point_index in enumerate(ordered_indices):
                point_index = int(point_index)
                writer.writerow(
                    [
                        order_id,
                        order_type,
                        row_order,
                        point_index,
                        split[point_index],
                        reference_order.get(point_index, ""),
                        f"{all_points[point_index, 0]:.17g}",
                        f"{all_points[point_index, 1]:.17g}",
                        "",
                        f"{y_all[point_index]:.17g}",
                        f"{X_all[point_index, 0]:.17g}",
                        f"{X_all[point_index, 1]:.17g}",
                        f"{X_all[point_index, 2]:.17g}",
                    ]
                )
    tmp_path.replace(path)


def ordered_csv_matches(path: Path, order_specs: list[dict]) -> bool:
    """Return whether an existing R CSV contains the current order specs."""
    if not path.exists():
        return False
    expected = [(i, spec["order_type"]) for i, spec in enumerate(order_specs, start=1)]
    try:
        seen: list[tuple[int, str]] = []
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "order_id" not in (reader.fieldnames or []) or "order_type" not in (reader.fieldnames or []):
                return False
            previous = None
            for row in reader:
                current = (int(row["order_id"]), str(row["order_type"]))
                if current != previous:
                    seen.append(current)
                    previous = current
        return seen == expected
    except (OSError, ValueError, KeyError):
        return False


def compact_float_array(values):
    return np.asarray(values, dtype=np.float32)


def json_ready(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def finite_fraction(values) -> float:
    array = np.asarray(values)
    if array.size == 0:
        return 1.0
    return float(np.mean(np.isfinite(array)))


def write_summary_json(path: Path, row: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, sort_keys=True)
    return row


def run_combo(args, combo_index: int) -> dict:
    data = load_npz_data(str(args.data))
    S0 = np.asarray(data["S"], dtype=np.float64)
    m_values = parse_int_list(args.m_values)
    order_count = int(args.order_count)
    combos = [(order_idx + 1, int(m)) for order_idx in range(order_count) for m in m_values]
    if combo_index < 0 or combo_index >= len(combos):
        raise IndexError(f"combo_index={combo_index} outside [0, {len(combos) - 1}]")

    order_id, m = combos[combo_index]
    m_tilde = m_tilde_for_m(m, args.m_tilde, args.small_m_tilde)
    order_specs = make_order_specs(S0, order_count=order_count, seed=args.order_seed)
    order_spec = order_specs[order_id - 1]
    permutation = np.asarray(order_spec["permutation"], dtype=int)
    order_type = order_spec["order_type"]
    combo = combo_name(order_id, m)
    save_root = Path(args.save_dir)
    posterior_dir = save_root / "posterior_samples"
    summary_dir = save_root / "combo_summaries"
    posterior_path = posterior_dir / f"{combo}_posterior_samples.npz"
    summary_path = summary_dir / f"{combo}_summary.json"
    ordered_csv = Path(args.save_dir) / "ordered_csv" / "sine_systematic_data.csv"

    if (
        posterior_path.exists()
        and summary_path.exists()
        and ordered_csv_matches(ordered_csv, order_specs)
        and not args.overwrite
    ):
        print(f"Skipping existing combo {combo}; use --overwrite to rerun.")
        with summary_path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        row["ordered_csv"] = str(ordered_csv)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, sort_keys=True)
        return row

    print(f"\n[{combo}] order_id={order_id}, m={m}")
    posterior_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite or not ordered_csv_matches(ordered_csv, order_specs):
        write_ordered_csv(ordered_csv, data, order_specs)

    ordered_data = ordered_training_data(data, permutation)
    true_params, Z = prepare_fit_data(
        ordered_data,
        "exp_neighbor_sine",
        fit_overrides={"m": int(m), "m_tilde": int(m_tilde)},
        force_rebuild_z=True,
        z_seed=int(args.z_seed + order_id * 1000 + m),
        verbose=False,
    )

    S = np.asarray(ordered_data["S"], dtype=np.float64)
    y_S = np.asarray(ordered_data["y_S"], dtype=np.float64).flatten()
    X_S = np.asarray(ordered_data["X_S"], dtype=np.float64)
    U = np.asarray(ordered_data["U"], dtype=np.float64)
    y_U = np.asarray(ordered_data["y_U"], dtype=np.float64).flatten()
    X_U = np.asarray(ordered_data["X_U"], dtype=np.float64)

    param_mode = build_param_mode(args.preset, fixed_text=args.fixed, eb_text=args.eb)
    neighbors = build_neighbor_indices(S, m)
    matern_kernel = make_parent_matern_kernel(true_params)
    L_true = compute_sparse_reverse_cholesky(S, matern_kernel, m)
    eb_reference_geometry = precompute_eb_reference_geometry(S, neighbors, m)
    fic_reference_geometry = None
    if use_noncentered_from_param_mode(param_mode):
        sigma_f0, length_scale0 = true_params["matern_params"]
        fic_reference_geometry = precompute_fic_reference_geometry(
            S,
            neighbors,
            m,
            sigma_f0,
            length_scale0,
        )

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    start_time = time.time()
    vi_w_S, vi_beta, vi_sigma, vi_params, losses, vi_Z = run_vi_inference(
        X_S=X_S,
        y_S=y_S,
        S=S,
        init_params=true_params,
        param_mode=param_mode,
        m=m,
        m_tilde=int(m_tilde),
        neighbors=neighbors,
        Z=Z,
        L=L_true,
        fic_reference_geometry=fic_reference_geometry,
        eb_reference_geometry=eb_reference_geometry,
        num_iterations=int(args.iters),
        learning_rate=float(args.lr),
        num_particles=int(args.particles),
        num_vi_samples=int(args.vi_samples),
        num_flows=int(args.flows),
        hidden_dims=hidden_dims,
        guide_type="flow",
        optimize_z=bool(args.optimize_z),
        z_jitter=float(args.z_jitter),
        save_dir=str(save_root),
        save_artifacts=False,
        verbose=False,
    )

    if finite_fraction(losses) < 1.0 or finite_fraction(vi_w_S) < 1.0 or finite_fraction(vi_Z) < 1.0:
        elapsed_sec = time.time() - start_time
        row = {
            "combo": combo,
            "combo_label": combo_label(order_id, m),
            "combo_index": combo_index,
            "order_id": order_id,
            "order_type": order_type,
            "m": m,
            "m_tilde": int(m_tilde),
            "status": "failed_nan_vi",
            "preset": args.preset,
            "lr": float(args.lr),
            "z_jitter": float(args.z_jitter),
            "optimize_z": bool(args.optimize_z),
            "vi_samples": int(args.vi_samples),
            "elapsed_sec": elapsed_sec,
            "loss_finite_fraction": finite_fraction(losses),
            "w_S_finite_fraction": finite_fraction(vi_w_S),
            "vi_Z_finite_fraction": finite_fraction(vi_Z),
            "posterior_npz": "",
            "ordered_csv": str(ordered_csv),
        }
        print(f"[{combo}] VI produced NaN; wrote failure summary only.")
        return write_summary_json(summary_path, row)

    pred_max = None if str(args.pred_max_samples).lower() == "none" else int(args.pred_max_samples)
    pred_mean, pred_std, y_U_samples, w_U_mean, w_U_std, w_U_samples = predict(
        vi_w_S,
        S,
        U,
        X_U,
        vi_beta,
        vi_sigma,
        vi_params,
        m=m,
        Z=vi_Z,
        neighbors_S=neighbors,
        max_samples=pred_max,
        seed=int(args.pred_seed + combo_index),
        return_w=True,
        verbose=False,
        z_jitter=float(args.z_jitter),
    )
    elapsed_sec = time.time() - start_time

    metric_summary = save_vi_metric_summary(
        str(save_root),
        y_U=y_U,
        pred_mean=pred_mean,
        pred_std=pred_std,
        pred_samples=y_U_samples,
        w_pred_mean=w_U_mean,
        w_pred_samples=w_U_samples,
        w_S_samples=vi_w_S,
        write_file=False,
    )

    np.savez_compressed(
        posterior_path,
        combo=np.array(combo),
        combo_label=np.array(combo_label(order_id, m)),
        order_id=np.asarray(order_id, dtype=np.int32),
        order_type=np.array(order_type),
        m=np.asarray(m, dtype=np.int32),
        m_tilde=np.asarray(m_tilde, dtype=np.int32),
        permutation=permutation.astype(np.int32),
        s_indices_ordered=np.asarray(ordered_data["s_indices"], dtype=np.int32),
        u_indices=np.asarray(ordered_data["u_indices"], dtype=np.int32),
        S=compact_float_array(S),
        U=compact_float_array(U),
        y_S=compact_float_array(y_S),
        y_U=compact_float_array(y_U),
        w_S_samples=compact_float_array(vi_w_S),
        w_U_samples=compact_float_array(w_U_samples),
        y_U_samples=compact_float_array(y_U_samples),
        w_U_mean=compact_float_array(w_U_mean),
        w_U_std=compact_float_array(w_U_std),
        y_U_mean=compact_float_array(pred_mean),
        y_U_std=compact_float_array(pred_std),
        beta_samples=compact_float_array(vi_beta),
        sigma_epsilon_samples=compact_float_array(vi_sigma),
        vi_Z=compact_float_array(vi_Z),
        losses=np.asarray(losses, dtype=np.float32),
        true_params=np.array(true_params, dtype=object),
        vi_params=np.array(vi_params, dtype=object),
        metrics=np.array(metric_summary, dtype=object),
        ordered_csv=np.array(str(ordered_csv)),
    )

    row = {
        "combo": combo,
        "combo_label": combo_label(order_id, m),
        "combo_index": combo_index,
        "order_id": order_id,
        "order_type": order_type,
        "m": m,
        "m_tilde": int(m_tilde),
        "status": "ok",
        "preset": args.preset,
        "lr": float(args.lr),
        "z_jitter": float(args.z_jitter),
        "optimize_z": bool(args.optimize_z),
        "vi_samples": int(args.vi_samples),
        "prediction_samples": int(len(y_U_samples)),
        "elapsed_sec": elapsed_sec,
        "final_loss": float(np.asarray(losses)[-1]),
        "posterior_npz": str(posterior_path),
        "ordered_csv": str(ordered_csv),
    }
    row.update({key: json_ready(value) for key, value in metric_summary.items()})
    row["vi_sigma_epsilon"] = float(vi_params["sigma_epsilon"])
    row["vi_sigma_f"] = float(vi_params["matern_params"][0])
    row["vi_length_scale"] = float(vi_params["matern_params"][1])
    row["vi_theta_tau1"] = float(vi_params["tau_params"][0])
    row["vi_theta_tau2"] = float(vi_params["tau_params"][1])
    row["vi_theta_g1"] = float(vi_params["g_params"][0])
    row["vi_theta_g2"] = float(vi_params["g_params"][1])

    return write_summary_json(summary_path, row)


def collect_summary(save_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((save_dir / "combo_summaries").glob("order*_m*_summary.json")):
        with path.open("r", encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        return rows

    def summary_row(row):
        return {
            "combo": row.get("combo", ""),
            "order_id": row.get("order_id", ""),
            "order_type": row.get("order_type", ""),
            "m": row.get("m", ""),
            "m_tilde": row.get("m_tilde", ""),
            "status": row.get("status", ""),
            "RMSPE": row.get("y_U_rmse", ""),
            "RSR": row.get("y_U_rsr", ""),
            "CRPS": row.get("y_U_crps", ""),
            "CI_coverage_percent": (
                100.0 * row["y_U_coverage_quantile_95"]
                if "y_U_coverage_quantile_95" in row and row["y_U_coverage_quantile_95"] != ""
                else ""
            ),
            "CI_width": row.get("y_U_mean_interval_width_quantile_95", ""),
        }

    summary_rows = [summary_row(row) for row in rows]
    fieldnames = [
        "combo",
        "order_id",
        "order_type",
        "m",
        "m_tilde",
        "status",
        "RMSPE",
        "RSR",
        "CRPS",
        "CI_coverage_percent",
        "CI_width",
    ]

    out_path = save_dir / "systematic_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Systematic summary saved to: {out_path}")
    return rows


def command_for_worker(args, combo_index: int) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data",
        str(args.data),
        "--save-dir",
        str(args.save_dir),
        "--order-count",
        str(args.order_count),
        "--order-seed",
        str(args.order_seed),
        "--m-values",
        str(args.m_values),
        "--m-tilde",
        str(args.m_tilde),
        "--small-m-tilde",
        str(args.small_m_tilde),
        "--preset",
        str(args.preset),
        "--fixed",
        str(args.fixed),
        "--eb",
        str(args.eb),
        "--iters",
        str(args.iters),
        "--lr",
        str(args.lr),
        "--z-jitter",
        str(args.z_jitter),
        "--particles",
        str(args.particles),
        "--vi-samples",
        str(args.vi_samples),
        "--pred-max-samples",
        str(args.pred_max_samples),
        "--flows",
        str(args.flows),
        "--hidden-dims",
        str(args.hidden_dims),
        "--z-seed",
        str(args.z_seed),
        "--pred-seed",
        str(args.pred_seed),
        "--combo-index",
        str(combo_index),
        "--parallel-workers",
        "1",
    ]
    cmd.append("--optimize-z" if args.optimize_z else "--fixed-z")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def run_parallel(args) -> None:
    m_values = parse_int_list(args.m_values)
    total = int(args.order_count) * len(m_values)
    gpu_ids = [item.strip() for item in str(args.gpu_ids).split(",") if item.strip()]
    if not gpu_ids:
        gpu_ids = [os.environ.get("CUDA_VISIBLE_DEVICES", "")]

    pending = list(range(total))
    running: list[tuple[subprocess.Popen, int]] = []
    max_workers = int(args.parallel_workers)
    launched_count = 0
    print(f"Launching {total} combos with {max_workers} worker(s), gpu_ids={gpu_ids}")

    ordered_csv = Path(args.save_dir) / "ordered_csv" / "sine_systematic_data.csv"
    data = load_npz_data(str(args.data))
    order_specs = make_order_specs(
        np.asarray(data["S"]),
        order_count=int(args.order_count),
        seed=int(args.order_seed),
    )
    if args.overwrite or not ordered_csv_matches(ordered_csv, order_specs):
        write_ordered_csv(ordered_csv, data, order_specs)

    while pending or running:
        while pending and len(running) < max_workers:
            combo_index = pending.pop(0)
            env = os.environ.copy()
            gpu_id = gpu_ids[launched_count % len(gpu_ids)]
            if gpu_id != "":
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
            cmd = command_for_worker(args, combo_index)
            print(f"Starting combo_index={combo_index} on CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '(default)')}")
            running.append((subprocess.Popen(cmd, env=env), combo_index))
            launched_count += 1

        still_running = []
        for proc, combo_index in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, combo_index))
            elif ret != 0:
                raise RuntimeError(f"combo_index={combo_index} failed with exit code {ret}")
            else:
                print(f"Finished combo_index={combo_index}")
        running = still_running
        if running:
            time.sleep(5)

    collect_summary(Path(args.save_dir))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--order-count", type=int, default=5)
    parser.add_argument("--order-seed", type=int, default=20260624)
    parser.add_argument("--m-values", default=",".join(str(x) for x in DEFAULT_M_VALUES))
    parser.add_argument("--m-tilde", type=int, default=50)
    parser.add_argument("--small-m-tilde", type=int, default=20)
    parser.add_argument(
        "--preset",
        default="all_eb",
        choices=["all_fixed", "all_eb", "fixed_base", "fixed_base_fixed_noise"],
    )
    parser.add_argument("--fixed", default="")
    parser.add_argument("--eb", default="")
    parser.add_argument("--iters", type=int, default=int(os.environ.get("VI_ITERS", "15000")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("VI_LR", "0.0002")))
    parser.add_argument("--z-jitter", type=float, default=float(os.environ.get("VI_Z_JITTER", "0.0001")))
    parser.add_argument("--particles", type=int, default=int(os.environ.get("VI_PARTICLES", "1")))
    parser.add_argument("--vi-samples", type=int, default=int(os.environ.get("VI_SAMPLES", "1000")))
    parser.add_argument("--pred-max-samples", default=os.environ.get("PRED_MAX_SAMPLES", "1000"))
    parser.add_argument("--flows", type=int, default=int(os.environ.get("VI_FLOWS", "3")))
    parser.add_argument("--hidden-dims", default=os.environ.get("VI_HIDDEN_DIMS", "256,256"))
    parser.set_defaults(optimize_z=True)
    parser.add_argument("--fixed-z", dest="optimize_z", action="store_false")
    parser.add_argument("--optimize-z", dest="optimize_z", action="store_true")
    parser.add_argument("--z-seed", type=int, default=42)
    parser.add_argument("--pred-seed", type=int, default=202)
    parser.add_argument("--combo-index", type=int, default=None, help="Run only one 0-based combo index.")
    parser.add_argument("--parallel-workers", type=int, default=default_parallel_workers())
    parser.add_argument("--gpu-ids", default=default_gpu_ids())
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.data = Path(args.data).resolve()
    args.save_dir = Path(args.save_dir).resolve()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    if args.combo_index is None and int(args.parallel_workers) > 1:
        run_parallel(args)
        return

    if args.combo_index is not None:
        row = run_combo(args, int(args.combo_index))
        print(json.dumps(row, indent=2, sort_keys=True))
        return

    m_values = parse_int_list(args.m_values)
    rows = []
    for combo_index in range(int(args.order_count) * len(m_values)):
        rows.append(run_combo(args, combo_index))
        collect_summary(args.save_dir)
    print(f"\nCompleted {len(rows)} combos.")
    collect_summary(args.save_dir)


if __name__ == "__main__":
    main()
