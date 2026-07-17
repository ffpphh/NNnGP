from __future__ import annotations

import struct
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.path import Path as MplPath


DATA_DIR = Path(__file__).with_name("prism_october_ppt_4km")
INPUT_CSV = DATA_DIR / "us/prism_october_ppt_2025_log_standardized_downsampled_40k.csv"
STATE_SHP = DATA_DIR / "boundaries/cb_2024_us_state_500k.shp"
STATE_DBF = DATA_DIR / "boundaries/cb_2024_us_state_500k.dbf"
OUTPUT_STEM = "prism_october_ppt_2025_log_standardized_downsampled_40k"
VALUE_COL = "log_ppt_2025_standardized"


def maximin_order(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n == 0:
        return np.array([], dtype=int)

    center = points.mean(axis=0, keepdims=True)
    first = int(np.argmax(np.linalg.norm(points - center, axis=1)))
    order = np.empty(n, dtype=int)
    order[0] = first

    remaining = np.ones(n, dtype=bool)
    remaining[first] = False
    min_dist = np.linalg.norm(points - points[first], axis=1)

    for pos in range(1, n):
        masked_dist = np.where(remaining, min_dist, -np.inf)
        next_idx = int(np.argmax(masked_dist))
        order[pos] = next_idx
        remaining[next_idx] = False
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[next_idx], axis=1))

    return order


def read_dbf_records(path: Path) -> list[dict[str, str]]:
    with path.open("rb") as f:
        header = f.read(32)
        n_records = struct.unpack("<I", header[4:8])[0]
        header_len = struct.unpack("<H", header[8:10])[0]
        record_len = struct.unpack("<H", header[10:12])[0]

        fields = []
        while True:
            desc = f.read(32)
            if desc[0] == 0x0D:
                break
            name = desc[:11].split(b"\x00", 1)[0].decode("ascii")
            fields.append((name, desc[16]))

        f.seek(header_len)
        records = []
        for _ in range(n_records):
            record = f.read(record_len)
            if not record or record[0:1] == b"*":
                continue
            offset = 1
            values = {}
            for name, length in fields:
                raw = record[offset : offset + length]
                offset += length
                values[name] = raw.decode("latin1").strip()
            records.append(values)
    return records


def read_shp_polygons(path: Path) -> list[list[np.ndarray]]:
    polygons = []
    with path.open("rb") as f:
        f.seek(100)
        while True:
            header = f.read(8)
            if len(header) < 8:
                break

            _, content_words = struct.unpack(">2i", header)
            content = f.read(content_words * 2)
            if len(content) < 44:
                polygons.append([])
                continue

            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type not in {5, 15, 25}:
                polygons.append([])
                continue

            num_parts, num_points = struct.unpack("<2i", content[36:44])
            parts_start = 44
            points_start = parts_start + 4 * num_parts
            parts = list(struct.unpack(f"<{num_parts}i", content[parts_start:points_start]))
            parts.append(num_points)

            points = np.frombuffer(
                content[points_start : points_start + 16 * num_points],
                dtype="<f8",
            ).reshape(num_points, 2)
            polygons.append([points[start:end].copy() for start, end in zip(parts[:-1], parts[1:])])
    return polygons


def state_parts(state: str) -> list[np.ndarray]:
    state = state.upper()
    records = read_dbf_records(STATE_DBF)
    polygons = read_shp_polygons(STATE_SHP)
    if len(records) != len(polygons):
        raise ValueError(f"DBF/SHP record mismatch: {len(records)} records vs {len(polygons)} shapes")

    for record, parts in zip(records, polygons):
        if record.get("STUSPS") == state:
            return parts
    raise ValueError(f"State boundary (STUSPS == {state!r}) not found.")


def points_in_parts(points: np.ndarray, parts: list[np.ndarray]) -> np.ndarray:
    mask = np.zeros(len(points), dtype=bool)
    for part in parts:
        if len(part) < 3:
            continue
        path = MplPath(part)
        mask |= path.contains_points(points, radius=1e-10)
    return mask


def apply_random_split(state_df: pd.DataFrame, train_size: int, random_seed: int) -> pd.DataFrame:
    if len(state_df) < train_size:
        raise ValueError(f"Cannot sample {train_size} training rows from {len(state_df)} {train_size=}.")

    out = state_df.copy()
    train_idx = out.sample(n=train_size, random_state=random_seed).index
    out["split"] = "U"
    out.loc[train_idx, "split"] = "S"

    train = out.loc[train_idx].copy()
    train_order = maximin_order(train[["lon", "lat"]].to_numpy())
    train = train.iloc[train_order]

    test = out.drop(index=train_idx)
    return pd.concat([train, test], axis=0, ignore_index=True)[["lon", "lat", "split", VALUE_COL]]


def extract_state(state: str, output_csv: Path, train_size: int, random_seed: int = 202510) -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    parts = state_parts(state)
    points = df[["lon", "lat"]].to_numpy(dtype=float)
    state_df = df.loc[points_in_parts(points, parts)].copy()
    state_df = apply_random_split(state_df, train_size=train_size, random_seed=random_seed)
    state_df.to_csv(output_csv, index=False)
    return state_df


def plot_state(state_df: pd.DataFrame, state: str, output_png: Path) -> None:
    if state_df.empty:
        raise ValueError(f"No {state.upper()} rows to plot.")

    state = state.upper()
    parts = state_parts(state)
    grid = state_df.pivot(index="lat", columns="lon", values=VALUE_COL).sort_index()
    lats = grid.index.to_numpy()
    lons = grid.columns.to_numpy()
    values = grid.to_numpy()
    limit = max(float(np.nanpercentile(np.abs(state_df[VALUE_COL]), 98)), 1.0)

    lon_min, lon_max = state_df["lon"].min(), state_df["lon"].max()
    lat_min, lat_max = state_df["lat"].min(), state_df["lat"].max()
    pad_lon = 0.45
    pad_lat = 0.35

    fig, ax = plt.subplots(figsize=(9.2, 7.2), constrained_layout=True)
    mesh = ax.pcolormesh(
        lons,
        lats,
        values,
        shading="auto",
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    for part in parts:
        ax.plot(part[:, 0], part[:, 1], color="#252525", linewidth=0.8, alpha=0.9)

    train = state_df.loc[state_df["split"].eq("S")]
    if not train.empty:
        ax.scatter(
            train["lon"],
            train["lat"],
            s=14,
            facecolors="none",
            edgecolors="black",
            linewidths=0.45,
            alpha=0.9,
            label="Training points",
        )
        ax.legend(loc="lower left", frameon=True, framealpha=0.85)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.015, extend="both")
    cbar.set_label("Downsampled standardized log precipitation")

    ax.set_title(f"{state} Downsampled Standardized October 2025 Log Precipitation")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(lon_min - pad_lon, lon_max + pad_lon)
    ax.set_ylim(lat_min - pad_lat, lat_max + pad_lat)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="white", linewidth=0.25, alpha=0.35)

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", default="TX", help="Two-letter state abbreviation, e.g. TX or FL.")
    parser.add_argument("--train-size", type=int, required=True, help="Randomly choose this many state rows as S; mark the rest U.")
    parser.add_argument("--random-seed", type=int, default=202510)
    args = parser.parse_args()

    state = args.state.upper()
    state_slug = state.lower()
    split_suffix = f"_train{args.train_size}"
    output_dir = DATA_DIR / state_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{OUTPUT_STEM}_{state_slug}{split_suffix}.csv"
    output_png = output_dir / f"{OUTPUT_STEM}_{state_slug}{split_suffix}.png"

    state_df = extract_state(state, output_csv, train_size=args.train_size, random_seed=args.random_seed)
    plot_state(state_df, state, output_png)

    print(f"Input CSV: {INPUT_CSV}")
    print(f"{state} rows: {len(state_df)}")
    print(f"S/train rows: {int(state_df['split'].eq('S').sum())}")
    print(f"U/test rows: {int(state_df['split'].eq('U').sum())}")
    print(f"Random split seed: {args.random_seed}")
    print(state_df[[VALUE_COL]].describe())
    print(f"Saved {state} CSV to {output_csv}")
    print(f"Saved {state} figure to {output_png}")


if __name__ == "__main__":
    main()
