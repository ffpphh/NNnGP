from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
THRESHOLD = -1.28
DEFAULT_QUANTILE_PROBS = tuple(np.round(np.arange(0.80, 0.99, 0.01), 2))
VALUE_COL = "log_ppt_2025_standardized"
TAIL_CHOICES = ("lower", "upper")


STATE_CONFIG = {
    "tx": {
        "run_kind": "prism_texas_october_ppt_2025",
        "train_size": 500,
        "resolution_dir": "prism_october_ppt_4km",
        "state_dir": "tx",
        "stem": "prism_october_ppt_2025_log_standardized_downsampled_40k_tx_train500",
        "method_dir": "tx",
    },
    "mt": {
        "run_kind": "prism_montana_october_ppt_2025",
        "train_size": 800,
        "resolution_dir": "prism_october_ppt_4km",
        "state_dir": "mt",
        "stem": "prism_october_ppt_2025_log_standardized_downsampled_40k_mt_train800",
        "method_dir": "mt",
    },
    "mt_800m": {
        "run_kind": "prism_montana_october_ppt_2025_800m",
        "train_size": 800,
        "resolution_dir": "prism_october_ppt_800m",
        "state_dir": "mt",
        "stem": "prism_october_ppt_2025_log_standardized_800m_downsampled_40k_mt_train800",
        "method_dir": "mt_800m",
    },
}


def threshold_slug(threshold: float) -> str:
    sign = "minus" if threshold < 0 else "plus"
    value = f"{abs(threshold):.6g}".replace(".", "p")
    return f"{sign}{value}"


def event_slug(tail: str, threshold: float) -> str:
    op = "lt" if tail == "lower" else "gt"
    return f"y_{op}_{threshold_slug(threshold)}"


def state_paths(state: str) -> dict[str, Path]:
    state = state.lower()
    if state not in STATE_CONFIG:
        raise ValueError(f"Unsupported state {state!r}; expected one of {sorted(STATE_CONFIG)}")
    if state != "mt_800m":
        raise ValueError("This submission package contains data only for state='mt_800m'.")
    config = STATE_CONFIG[state]
    return {
        "result_dir": EXPERIMENT_DIR / "outputs" / "nnngp",
        "data_csv": EXPERIMENT_DIR / "data" / "source" / f"{config['stem']}.csv",
        "nngp_dir": EXPERIMENT_DIR / "outputs" / "r_nngp",
        "nnmp_dir": EXPERIMENT_DIR / "outputs" / "r_nnmp",
    }


def crps_ensemble(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    if samples.ndim != 2 or samples.shape[0] != len(truth):
        raise ValueError(f"samples shape {samples.shape} incompatible with truth length {len(truth)}")

    n_samples = samples.shape[1]
    term1 = np.mean(np.abs(samples - truth[:, None]), axis=1)
    sorted_samples = np.sort(samples, axis=1)
    coeff = 2.0 * np.arange(1, n_samples + 1, dtype=np.float64) - n_samples - 1.0
    half_pair_term = sorted_samples @ coeff / (n_samples**2)
    return term1 - half_pair_term


def left_tail_twcrps(samples: np.ndarray, truth: np.ndarray, threshold: float) -> np.ndarray:
    return crps_ensemble(np.minimum(samples, threshold), np.minimum(truth, threshold))


def tail_twcrps(samples: np.ndarray, truth: np.ndarray, threshold: float, tail: str) -> np.ndarray:
    if tail == "lower":
        return crps_ensemble(np.minimum(samples, threshold), np.minimum(truth, threshold))
    if tail == "upper":
        return crps_ensemble(np.maximum(samples, threshold), np.maximum(truth, threshold))
    raise ValueError(f"Unsupported tail {tail!r}")


def tail_probabilities_and_events(samples: np.ndarray, truth: np.ndarray, threshold: float, tail: str):
    if tail == "lower":
        return np.mean(samples < threshold, axis=1), truth < threshold
    if tail == "upper":
        return np.mean(samples > threshold, axis=1), truth > threshold
    raise ValueError(f"Unsupported tail {tail!r}")


def brier_skill_score(bs: float, events: np.ndarray) -> tuple[float, float]:
    event_rate = float(np.mean(events))
    bs_ref = event_rate * (1.0 - event_rate)
    if bs_ref <= 0.0:
        return float("nan"), bs_ref
    return 1.0 - bs / bs_ref, bs_ref


def summarize_method(
    method: str,
    coords: np.ndarray,
    truth: np.ndarray,
    samples: np.ndarray,
    point_set: np.ndarray,
    threshold: float,
    tail: str,
    quantile_probability: float | None = None,
):
    event_name = event_slug(tail, threshold)
    probabilities, events = tail_probabilities_and_events(samples, truth, threshold, tail)
    bs_point = (probabilities - events.astype(np.float64)) ** 2
    crps_point = crps_ensemble(samples, truth)
    twcrps_point = tail_twcrps(samples, truth, threshold, tail)
    lower_95, upper_95 = np.quantile(samples, [0.025, 0.975], axis=1)
    covered_95 = (truth >= lower_95) & (truth <= upper_95)
    interval_width_95 = upper_95 - lower_95

    mask = point_set == "U"
    bs = float(np.mean(bs_point[mask]))
    bss, bs_ref = brier_skill_score(bs, events[mask])
    event_mask = events & mask
    extreme_coverage = float(np.mean(covered_95[event_mask])) if np.any(event_mask) else float("nan")
    extreme_ci_width = float(np.mean(interval_width_95[event_mask])) if np.any(event_mask) else float("nan")
    op = "<" if tail == "lower" else ">"
    return {
        "event": f"Y {op} {threshold:.2f}",
        "method": method,
        "evaluation_set": "y_U",
        "tail": tail,
        "quantile_probability": quantile_probability,
        "threshold": threshold,
        "threshold_abs": abs(threshold),
        "n_locations": int(np.sum(mask)),
        "n_S": int(np.sum(point_set[mask] == "S")),
        "n_U": int(np.sum(point_set[mask] == "U")),
        "event_count": int(np.sum(events[mask])),
        "event_rate": float(np.mean(events[mask])),
        "mean_predicted_event_probability": float(np.mean(probabilities[mask])),
        "brier_score": bs,
        "brier_reference_climatology": bs_ref,
        "brier_skill_score": bss,
        "crps": float(np.mean(crps_point[mask])),
        "coverage_95_all_locations": float(np.mean(covered_95[mask])),
        "twcrps_tail_weighted": float(np.mean(twcrps_point[mask])),
        "extreme_value_coverage_95": extreme_coverage,
        "extreme_value_ci_width_95": extreme_ci_width,
    }


def load_state_data(data_csv: Path) -> pd.DataFrame:
    return pd.read_csv(data_csv)


def load_nnngp_samples(result_dir: Path, data_csv: Path):
    tx = load_state_data(data_csv)
    pred_data = np.load(result_dir / "vi_prediction_results.npz", allow_pickle=True)
    vi_data = np.load(result_dir / "vi_results.npz", allow_pickle=True)

    raw_all = tx[["lon", "lat"]].to_numpy(dtype=np.float64)
    truth_all = tx[VALUE_COL].to_numpy(dtype=np.float64)
    is_s = tx["split"].eq("S").to_numpy()
    is_u = tx["split"].eq("U").to_numpy()

    mins = raw_all.min(axis=0)
    spans = np.maximum(np.ptp(raw_all, axis=0), 1e-12)
    scaled = (raw_all - mins) / spans
    x_s = np.column_stack([np.ones(is_s.sum()), scaled[is_s, 0], scaled[is_s, 1]])

    w_s_samples = np.asarray(vi_data["w_S_samples"], dtype=np.float64)
    beta_samples = np.asarray(vi_data["beta_samples"], dtype=np.float64)
    sigma_samples = np.asarray(vi_data["sigma_epsilon_samples"], dtype=np.float64).reshape(-1)
    y_s_mean = beta_samples @ x_s.T + w_s_samples
    rng = np.random.default_rng(203)
    y_s_samples = y_s_mean + rng.normal(size=y_s_mean.shape) * sigma_samples[:, None]

    y_u_samples = np.asarray(pred_data["pred_samples"], dtype=np.float64)
    samples = np.vstack([y_s_samples.T, y_u_samples.T])
    coords = np.vstack([raw_all[is_s], np.asarray(pred_data["raw_U"], dtype=np.float64)])
    truth = np.concatenate([truth_all[is_s], np.asarray(pred_data["y_U"], dtype=np.float64).reshape(-1)])
    point_set = np.concatenate([np.full(is_s.sum(), "S"), np.full(is_u.sum(), "U")])
    return coords, truth, samples, point_set


def load_csv_method_samples(method_dir: Path):
    pred = pd.read_csv(method_dir / "predictions_y.csv")
    parts = []
    for split in ("S", "U"):
        pred_split = pred.loc[pred["split"].eq(split) & pred["variable"].eq("y")].copy()
        samples = pd.read_csv(method_dir / f"posterior_samples_y_{split}.csv").to_numpy(dtype=np.float64)
        if samples.shape[0] != len(pred_split):
            raise ValueError(
                f"{method_dir.name} split {split}: sample rows {samples.shape[0]} "
                f"!= prediction rows {len(pred_split)}"
            )
        direct_diff = np.max(np.abs(samples.mean(axis=1) - pred_split["pred_mean"].to_numpy(dtype=np.float64)))
        row_order = pred_split["row_order"].to_numpy(dtype=int)
        if (
            len(np.unique(row_order)) == len(row_order)
            and row_order.min(initial=0) == 0
            and row_order.max(initial=-1) == len(row_order) - 1
        ):
            row_order_samples = samples[row_order]
            row_order_diff = np.max(
                np.abs(row_order_samples.mean(axis=1) - pred_split["pred_mean"].to_numpy(dtype=np.float64))
            )
            if row_order_diff < direct_diff:
                samples = row_order_samples
        parts.append(
            (
                pred_split[["x", "y"]].to_numpy(dtype=np.float64),
                pred_split["true_value"].to_numpy(dtype=np.float64),
                samples,
                np.full(len(pred_split), split),
            )
        )

    coords = np.vstack([part[0] for part in parts])
    truth = np.concatenate([part[1] for part in parts])
    samples = np.vstack([part[2] for part in parts])
    point_set = np.concatenate([part[3] for part in parts])
    return coords, truth, samples, point_set


def parse_thresholds(text: str | None, single_threshold: float) -> list[float]:
    if text is None or text.strip() == "":
        return [float(single_threshold)]
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_quantile_probs(text: str | None) -> list[float]:
    if text is None or text.strip() == "":
        return []
    probs = [float(item.strip()) for item in text.split(",") if item.strip()]
    invalid = [prob for prob in probs if prob <= 0.0 or prob >= 1.0]
    if invalid:
        raise ValueError(f"Quantile probabilities must be in (0, 1); got {invalid}")
    return probs


def parse_tails(text: str) -> list[str]:
    tails = [item.strip() for item in text.split(",") if item.strip()]
    invalid = [tail for tail in tails if tail not in TAIL_CHOICES]
    if invalid:
        raise ValueError(f"Invalid tails {invalid}; expected values from {TAIL_CHOICES}")
    return tails


def threshold_output_slug(thresholds: list[float]) -> str:
    return "_".join(threshold_slug(abs(threshold)) for threshold in thresholds)


def plot_metric_lines(summary_df: pd.DataFrame, output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("brier_skill_score", "Brier Skill Score", "BSS (%)", 100.0),
        ("twcrps_tail_weighted", "Tail-weighted CRPS", "twCRPS", 1.0),
        ("extreme_value_coverage_95", "95% Extreme Value Coverage", "Coverage (%)", 100.0),
        ("extreme_value_ci_width_95", "95% Extreme Value CI Width", "CI width", 1.0),
    ]
    tail_labels = {"lower": "Left tail", "upper": "Right tail"}

    for metric, title, ylabel, scale in metrics:
        for tail in TAIL_CHOICES:
            tail_df = summary_df.loc[summary_df["tail"].eq(tail)].copy()
            if tail_df.empty:
                continue

            fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
            for method, method_df in tail_df.groupby("method", sort=False):
                method_df = method_df.sort_values("quantile_probability")
                ax.plot(
                    method_df["quantile_probability"],
                    method_df[metric] * scale,
                    marker="o",
                    linewidth=1.8,
                    markersize=4.5,
                    label=method,
                )
            if metric == "extreme_value_coverage_95":
                ax.axhline(95.0, color="#555555", linestyle="--", linewidth=1.2, alpha=0.85, label="95% target")
            ax.set_title(f"{tail_labels[tail]} {title} by Standard Normal Quantile")
            ax.set_xlabel("Standard normal quantile probability")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0.80, 0.99)
            ax.set_xticks(np.round(np.arange(0.80, 1.00, 0.02), 2))
            ax.grid(True, linewidth=0.35, alpha=0.45)
            ax.legend(frameon=True)
            fig.savefig(plot_dir / f"{metric}_{tail}_tail.png", dpi=300, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Optional comma-separated positive threshold magnitudes. Overrides --quantile-probs when provided.",
    )
    parser.add_argument(
        "--quantile-probs",
        default=",".join(f"{value:.2f}" for value in DEFAULT_QUANTILE_PROBS),
        help="Comma-separated standard-normal quantile probabilities. Default: 0.80,0.81,...,0.98.",
    )
    parser.add_argument("--tail", choices=TAIL_CHOICES, default="lower")
    parser.add_argument(
        "--tails",
        default="lower,upper",
        help="Comma-separated tails to evaluate: lower, upper, or lower,upper.",
    )
    parser.add_argument("--state", choices=sorted(STATE_CONFIG), default="tx")
    args = parser.parse_args()

    if args.thresholds:
        threshold_specs = [(abs(value), None) for value in parse_thresholds(args.thresholds, args.threshold)]
    else:
        quantile_probs = parse_quantile_probs(args.quantile_probs)
        threshold_specs = [(float(norm.ppf(prob)), prob) for prob in quantile_probs]
    tails = parse_tails(args.tails)
    paths = state_paths(args.state)
    result_dir = paths["result_dir"]
    output_dir = result_dir / "tail_event_metric_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    method_loaders = [("NNnGP", lambda: load_nnngp_samples(result_dir, paths["data_csv"]))]
    if (paths["nngp_dir"] / "predictions_y.csv").exists():
        method_loaders.append(("NNGP", lambda: load_csv_method_samples(paths["nngp_dir"])))
    else:
        print(f"Skipping NNGP: missing {paths['nngp_dir'] / 'predictions_y.csv'}")
    if (paths["nnmp_dir"] / "predictions_y.csv").exists():
        method_loaders.append(("NNMP", lambda: load_csv_method_samples(paths["nnmp_dir"])))
    else:
        print(f"Skipping NNMP: missing {paths['nnmp_dir'] / 'predictions_y.csv'}")

    method_data = []
    for method, loader in method_loaders:
        coords, truth, samples, point_set = loader()
        method_data.append((method, coords, truth, samples, point_set))

    rows = []
    for threshold_mag, quantile_probability in threshold_specs:
        threshold_mag = abs(float(threshold_mag))
        for tail in tails:
            threshold = -threshold_mag if tail == "lower" else threshold_mag
            for method, coords, truth, samples, point_set in method_data:
                rows.append(
                    summarize_method(
                        method,
                        coords,
                        truth,
                        samples,
                        point_set,
                        threshold,
                        tail,
                        quantile_probability=quantile_probability,
                    )
                )

    summary_df = pd.DataFrame(rows)
    summary_path = output_dir / "tail_event_metric_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    plot_metric_lines(summary_df, output_dir)

    print(f"Saved summary to {summary_path}")
    print(f"Saved plots to {output_dir / 'plots'}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
