from __future__ import annotations

import argparse
import csv
import shutil
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from matplotlib.colors import TwoSlopeNorm

try:
    from .extract_texas_prism_downsampled import maximin_order, points_in_parts, state_parts
except ImportError:  # Support direct execution: python results/rain/build_montana_prism_800m.py
    from extract_texas_prism_downsampled import maximin_order, points_in_parts, state_parts


RAIN_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = EXPERIMENT_DIR / "data" / "raw"
ZIP_DIR = DATA_DIR / "zip"
EXTRACT_DIR = DATA_DIR / "extracted"
OUT_DIR = EXPERIMENT_DIR / "data" / "source"
SUMMARY_PATH = DATA_DIR / "download_summary.csv"

BASE_URL = "https://services.nacse.org/prism/data/get/us/800m/ppt"
BASELINE_START_YEAR = 1995
BASELINE_END_YEAR = 2024
TARGET_YEAR = 2025
MONTH = 10
EPSILON = 1e-4
VALUE_COL = "log_ppt_2025_standardized"
TRAIN_SIZE = 800
TARGET_DOWNSAMPLED_SIZE = 40000
RANDOM_SEED = 43
STATE = "MT"
CROPPED_DIR = DATA_DIR / "cropped" / STATE.lower()
OUTPUT_STEM = "prism_october_ppt_2025_log_standardized_800m_downsampled_40k_mt_train800"
RAW_OUTPUT_STEM = "prism_october_ppt_2025_log_standardized_800m_mt"


def filename_from_response(response: requests.Response, fallback: str) -> str:
    disposition = response.headers.get("content-disposition", "")
    for item in disposition.split(";"):
        item = item.strip()
        if item.startswith("filename="):
            return unquote(item.split("=", 1)[1].strip('"'))
    return fallback


def is_valid_zip(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0 and zipfile.is_zipfile(path)


def nc_path_for_month(yyyymm: str) -> Path | None:
    month_dir = EXTRACT_DIR / yyyymm
    nc_files = sorted(month_dir.glob("*.nc"))
    return nc_files[0] if nc_files else None


def cropped_nc_path_for_month(yyyymm: str) -> Path:
    return CROPPED_DIR / yyyymm / f"prism_ppt_{STATE.lower()}_800m_{yyyymm}.nc"


def crop_month_to_state(nc_path: Path, yyyymm: str, bbox: tuple[float, float, float, float]) -> Path:
    out_path = cropped_nc_path_for_month(yyyymm)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(nc_path) as ds:
        var_name = "ppt" if "ppt" in ds.data_vars else "Band1"
        cropped = subset_bbox(ds[[var_name]], bbox).load()
        cropped.to_netcdf(out_path)
    return out_path


def download_month(
    year: int,
    bbox: tuple[float, float, float, float],
    discard_full_us: bool,
) -> dict[str, str | int | float]:
    yyyymm = f"{year}{MONTH:02d}"
    cropped_path = cropped_nc_path_for_month(yyyymm)
    if cropped_path.exists() and cropped_path.stat().st_size > 0:
        return {
            "year": year,
            "month": MONTH,
            "yyyymm": yyyymm,
            "url": "",
            "zip_path": "",
            "zip_size_mb": "",
            "extract_dir": "",
            "netcdf_file": "",
            "cropped_netcdf_file": str(cropped_path),
        }

    url = f"{BASE_URL}/{yyyymm}?format=nc"
    fallback_name = f"prism_ppt_us_800m_{yyyymm}.zip"
    zip_path = ZIP_DIR / fallback_name
    month_extract_dir = EXTRACT_DIR / yyyymm

    if not is_valid_zip(zip_path):
        with requests.get(url, stream=True, timeout=240) as response:
            response.raise_for_status()
            filename = filename_from_response(response, fallback_name)
            zip_path = ZIP_DIR / filename
            if not is_valid_zip(zip_path):
                tmp_path = zip_path.with_suffix(zip_path.suffix + ".part")
                with tmp_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp_path.replace(zip_path)
        time.sleep(2)

    month_extract_dir.mkdir(parents=True, exist_ok=True)
    if nc_path_for_month(yyyymm) is None:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(month_extract_dir)

    nc_path = nc_path_for_month(yyyymm)
    if nc_path is None:
        raise FileNotFoundError(f"No NetCDF found after extracting {zip_path} into {month_extract_dir}")

    cropped_path = crop_month_to_state(nc_path, yyyymm, bbox)
    zip_size_mb = round(zip_path.stat().st_size / 1024 / 1024, 2)
    if discard_full_us:
        if zip_path.exists():
            zip_path.unlink()
        if month_extract_dir.exists():
            shutil.rmtree(month_extract_dir)

    return {
        "year": year,
        "month": MONTH,
        "yyyymm": yyyymm,
        "url": url,
        "zip_path": str(zip_path),
        "zip_size_mb": zip_size_mb,
        "extract_dir": str(month_extract_dir),
        "netcdf_file": str(nc_path) if nc_path else "",
        "cropped_netcdf_file": str(cropped_path),
    }


def download_all(discard_full_us: bool) -> None:
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    CROPPED_DIR.mkdir(parents=True, exist_ok=True)
    bbox = state_bbox()
    rows = []
    for year in range(BASELINE_START_YEAR, TARGET_YEAR + 1):
        print(f"Downloading/cropping PRISM 800m monthly ppt for {STATE} {year}-{MONTH:02d}...")
        rows.append(download_month(year, bbox=bbox, discard_full_us=discard_full_us))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def state_bbox(pad: float = 0.05) -> tuple[float, float, float, float]:
    parts = state_parts(STATE)
    all_points = np.vstack(parts)
    lon_min, lat_min = all_points.min(axis=0)
    lon_max, lat_max = all_points.max(axis=0)
    return lon_min - pad, lon_max + pad, lat_min - pad, lat_max + pad


def subset_bbox(da: xr.DataArray, bbox: tuple[float, float, float, float]) -> xr.DataArray:
    lon_min, lon_max, lat_min, lat_max = bbox
    da = da.where(
        (da["lon"] >= lon_min)
        & (da["lon"] <= lon_max)
        & (da["lat"] >= lat_min)
        & (da["lat"] <= lat_max),
        drop=True,
    )
    return da


def open_log_ppt(year: int, bbox: tuple[float, float, float, float]) -> xr.DataArray:
    yyyymm = f"{year}{MONTH:02d}"
    nc_path = cropped_nc_path_for_month(yyyymm)
    if not nc_path.exists():
        raise FileNotFoundError(f"Missing cropped {STATE} PRISM NetCDF for {yyyymm}: {nc_path}")

    with xr.open_dataset(nc_path) as ds:
        var_name = "ppt" if "ppt" in ds.data_vars else "Band1"
        ppt = subset_bbox(ds[var_name], bbox)
        ppt = ppt.where(ppt >= 0)
        return np.log(ppt + EPSILON).rename("log_ppt").load()


def compute_montana_standardized() -> pd.DataFrame:
    bbox = state_bbox()
    baseline_logs = []
    for year in range(BASELINE_START_YEAR, BASELINE_END_YEAR + 1):
        print(f"Reading Montana bbox baseline {year}-{MONTH:02d}...")
        baseline_logs.append(open_log_ppt(year, bbox).assign_coords(year=year))

    baseline = xr.concat(baseline_logs, dim="year")
    mean_baseline = baseline.mean(dim="year", skipna=True)
    std_baseline = baseline.std(dim="year", skipna=True, ddof=1)

    print(f"Reading Montana bbox target {TARGET_YEAR}-{MONTH:02d}...")
    log_target = open_log_ppt(TARGET_YEAR, bbox)
    standardized = ((log_target - mean_baseline) / std_baseline).rename(VALUE_COL)
    standardized = standardized.where(np.isfinite(standardized))

    df = xr.merge(
        [
            mean_baseline.rename("log_ppt_mean_1995_2024"),
            std_baseline.rename("log_ppt_std_1995_2024"),
            log_target.rename("log_ppt_2025"),
            standardized,
        ]
    ).to_dataframe().reset_index()
    df = df.dropna(subset=[VALUE_COL])

    parts = state_parts(STATE)
    state_mask = points_in_parts(df[["lon", "lat"]].to_numpy(dtype=float), parts)
    df = df.loc[state_mask].copy()
    df.insert(0, "epsilon_added_before_log", EPSILON)
    return df


def apply_random_split(state_df: pd.DataFrame, train_size: int, random_seed: int) -> pd.DataFrame:
    if len(state_df) < train_size:
        raise ValueError(f"Cannot sample {train_size} training rows from {len(state_df)} rows.")

    out = state_df.copy()
    train_idx = out.sample(n=train_size, random_state=random_seed).index
    out["split"] = "U"
    out.loc[train_idx, "split"] = "S"

    train = out.loc[train_idx].copy()
    train_order = maximin_order(train[["lon", "lat"]].to_numpy())
    train = train.iloc[train_order]

    test = out.drop(index=train_idx)
    return pd.concat([train, test], axis=0, ignore_index=True)[["lon", "lat", "split", VALUE_COL]]


def downsample_to_target(state_df: pd.DataFrame, target_size: int) -> tuple[pd.DataFrame, int, int]:
    if len(state_df) <= target_size:
        return state_df[["lon", "lat", VALUE_COL]].copy(), 1, 1

    factor = max(1, int(round(np.sqrt(len(state_df) / target_size))))
    lat_factor = factor
    lon_factor = factor

    df = state_df.copy()
    unique_lats = np.sort(df["lat"].unique())
    unique_lons = np.sort(df["lon"].unique())
    lat_lookup = {lat: idx for idx, lat in enumerate(unique_lats)}
    lon_lookup = {lon: idx for idx, lon in enumerate(unique_lons)}
    df["lat_block"] = df["lat"].map(lat_lookup) // lat_factor
    df["lon_block"] = df["lon"].map(lon_lookup) // lon_factor

    out = (
        df.groupby(["lat_block", "lon_block"], as_index=False)
        .agg(
            lon=("lon", "mean"),
            lat=("lat", "mean"),
            log_ppt_2025_standardized=(VALUE_COL, "mean"),
        )
    )
    return out[["lon", "lat", VALUE_COL]], lat_factor, lon_factor


def plot_montana(state_df: pd.DataFrame, output_png: Path) -> None:
    parts = state_parts(STATE)
    limit = max(float(np.nanpercentile(np.abs(state_df[VALUE_COL]), 98)), 1.0)

    fig, ax = plt.subplots(figsize=(9.2, 7.2), constrained_layout=True)
    mesh = ax.scatter(
        state_df["lon"],
        state_df["lat"],
        c=state_df[VALUE_COL],
        s=8,
        marker="s",
        linewidths=0,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    for part in parts:
        ax.plot(part[:, 0], part[:, 1], color="#252525", linewidth=0.8, alpha=0.9)

    train = state_df.loc[state_df["split"].eq("S")]
    ax.scatter(
        train["lon"],
        train["lat"],
        s=10,
        facecolors="none",
        edgecolors="black",
        linewidths=0.35,
        alpha=0.85,
        label="Training points",
    )
    ax.legend(loc="lower left", frameon=True, framealpha=0.85)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.015, extend="both")
    cbar.set_label("Downsampled 800m standardized log precipitation")
    ax.set_title("MT Downsampled 800m Standardized October 2025 Log Precipitation")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(state_df["lon"].min()) - 0.45, float(state_df["lon"].max()) + 0.45)
    ax.set_ylim(float(state_df["lat"].min()) - 0.35, float(state_df["lat"].max()) + 0.35)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="white", linewidth=0.25, alpha=0.35)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--discard-full-us",
        action="store_true",
        help="After each month is cropped to Montana, delete the full-US zip and extracted files.",
    )
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--target-size", type=int, default=TARGET_DOWNSAMPLED_SIZE)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    if not args.skip_download:
        download_all(discard_full_us=args.discard_full_us)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = OUTPUT_STEM.replace("train800", f"train{args.train_size}")
    output_csv = OUT_DIR / f"{output_stem}.csv"
    output_png = OUT_DIR / f"{output_stem}.png"
    raw_output_csv = OUT_DIR / f"{RAW_OUTPUT_STEM}.csv"

    raw_state_df = compute_montana_standardized()
    raw_state_df[["lon", "lat", VALUE_COL]].to_csv(raw_output_csv, index=False)

    state_df, lat_factor, lon_factor = downsample_to_target(raw_state_df, target_size=args.target_size)
    state_df = apply_random_split(state_df, train_size=args.train_size, random_seed=args.random_seed)
    state_df.to_csv(output_csv, index=False)
    plot_montana(state_df, output_png)

    print(f"Resolution directory: {DATA_DIR}")
    print(f"Raw MT 800m rows: {len(raw_state_df)}")
    print(f"Saved raw MT 800m CSV without split to {raw_output_csv}")
    print(f"Downsample target rows: {args.target_size}")
    print(f"Downsample factors: lat={lat_factor}, lon={lon_factor}")
    print(f"Downsampled MT rows: {len(state_df)}")
    print(f"S/train rows: {int(state_df['split'].eq('S').sum())}")
    print(f"U/test rows: {int(state_df['split'].eq('U').sum())}")
    print(f"Random split seed: {args.random_seed}")
    print(state_df[[VALUE_COL]].describe())
    print(f"Saved MT 800m CSV to {output_csv}")
    print(f"Saved MT 800m figure to {output_png}")


if __name__ == "__main__":
    main()
