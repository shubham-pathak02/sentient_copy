from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import FEATURES_ROOT, PROCESSED_ROOT, ensure_dir


FEATURE_COLUMNS = [
    "road_way_count",
    "road_length_km",
    "s1_backscatter_mean",
    "s1_backscatter_p90",
    "s1_flood_fraction",
    "s2_ndvi_mean",
    "s2_ndwi_mean",
    "s2_green_p90",
    "landsat_thermal_mean_k",
    "landsat_heat_exposure_fraction",
    "nightlights_mean",
    "nightlights_p90",
    "population_mean",
    "population_p90",
    "era5_total_precipitation_mean",
    "era5_total_precipitation_sum",
    "era5_2m_temperature_mean",
]

# Columns whose within-window dynamics (deltas, trends, volatility) carry signal.
DYNAMIC_COLUMNS = [
    "s1_backscatter_mean",
    "s1_flood_fraction",
    "s2_ndvi_mean",
    "s2_ndwi_mean",
    "landsat_thermal_mean_k",
    "landsat_heat_exposure_fraction",
    "nightlights_mean",
    "era5_total_precipitation_sum",
    "era5_2m_temperature_mean",
]


def city_key(city: str) -> str:
    return city.lower().replace(" ", "_")


def load_processed_monthlies(city: str) -> pd.DataFrame:
    city_dir = PROCESSED_ROOT / city_key(city)
    files = sorted(city_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No processed parquet files found in {city_dir}")

    frames = [pd.read_parquet(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df["city"] = city_key(city)
    if "tile_id" not in df.columns:
        df["tile_id"] = "tile_00_00"
    df["tile_id"] = df["city"].astype(str) + "__" + df["tile_id"].astype(str)
    df["date"] = pd.to_datetime(df["month_id"] + "-01")
    df = df.sort_values(["city", "tile_id", "date"]).reset_index(drop=True)
    return df


def normalize_features(
    df: pd.DataFrame,
    train_mask: pd.Series,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    stats: dict[str, dict[str, float]] = {}
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        fit_vals = out.loc[train_mask, col]
        mu = float(fit_vals.mean())
        sigma = float(fit_vals.std())
        if sigma <= 1e-9:
            sigma = 1.0
        out[col] = (out[col] - mu) / sigma
        stats[col] = {"mean": mu, "std": sigma}
    return out, stats


def build_windows(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for tile_id, g in df.groupby("tile_id", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) <= window_size:
            continue

        for idx in range(window_size - 1, len(g) - 1):
            window = g.iloc[idx - window_size + 1 : idx + 1]
            next_row = g.iloc[idx + 1]
            cur = g.iloc[idx]

            # Enforce contiguous month sequences so windows don't bridge long data gaps.
            window_dates = pd.to_datetime(window["month_id"] + "-01")
            next_date = pd.to_datetime(str(next_row["month_id"]) + "-01")
            expected_dates = pd.date_range(window_dates.iloc[0], periods=window_size + 1, freq="MS")
            actual_dates = list(window_dates) + [next_date]
            if list(expected_dates) != actual_dates:
                continue

            # Relative future risk proxy (stress accumulation) using next-month climate+flood stress.
            target_proxy = (
                float(next_row["era5_total_precipitation_sum"]) +
                2.0 * float(next_row["landsat_heat_exposure_fraction"]) +
                3.0 * float(next_row["s1_flood_fraction"])
            )

            record: dict[str, float | int | str] = {
                "city": str(cur["city"]),
                "tile_id": str(tile_id),
                "time_window": f"{window.iloc[0]['month_id']}__{window.iloc[-1]['month_id']}",
                "target_month": str(next_row["month_id"]),
                "target_proxy": target_proxy,
                "imagery_reference": "compact_geotiff_monthly",
                "sample_month": str(cur["month_id"]),
            }

            for lag in range(window_size):
                row = window.iloc[window_size - 1 - lag]
                for col in FEATURE_COLUMNS:
                    record[f"{col}_lag{lag}"] = float(row[col])

            for city_name in sorted(df["city"].astype(str).unique()):
                record[f"city_{city_name}"] = 1.0 if str(cur["city"]) == city_name else 0.0

            record["stress_accum_rain_3m"] = float(window["era5_total_precipitation_sum"].tail(3).sum())
            record["stress_accum_heat_3m"] = float(window["landsat_heat_exposure_fraction"].tail(3).mean())
            record["stress_accum_flood_3m"] = float(window["s1_flood_fraction"].tail(3).sum())

            # Within-window temporal dynamics (all computed from past observations only).
            for col in DYNAMIC_COLUMNS:
                newest = float(window.iloc[-1][col])
                oldest = float(window.iloc[0][col])
                vals = window[col].astype(float)
                record[f"{col}_delta1"] = newest - float(window.iloc[-2][col])
                record[f"{col}_trend"] = (newest - oldest) / max(1, window_size - 1)
                record[f"{col}_winmax"] = float(vals.max())
                record[f"{col}_winstd"] = float(vals.std(ddof=0))

            # Calendar seasonality of the month being predicted (deterministic, not observed data).
            target_month_num = int(str(next_row["month_id"]).split("-")[1])
            record["target_month_sin"] = math.sin(2.0 * math.pi * target_month_num / 12.0)
            record["target_month_cos"] = math.cos(2.0 * math.pi * target_month_num / 12.0)

            # Cross-stress interactions.
            record["ix_rain_flood"] = record["stress_accum_rain_3m"] * record["stress_accum_flood_3m"]
            record["ix_heat_temp"] = record["stress_accum_heat_3m"] * float(cur["era5_2m_temperature_mean"])
            record["ix_rain_ndwi"] = record["stress_accum_rain_3m"] * float(cur["s2_ndwi_mean"])
            record["ix_load_flood"] = float(cur["nightlights_mean"]) * record["stress_accum_flood_3m"]
            rows.append(record)

    if not rows:
        raise ValueError(
            "Not enough contiguous tile/month rows to build windows. "
            "Increase overlapping date range or lower --window-size."
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML-ready temporal windows from processed monthly stress data.")
    parser.add_argument("--city", required=True, help="City name, or a comma-separated list for one general model.")
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    args = parser.parse_args()

    if args.window_size < 2:
        raise ValueError("--window-size must be >= 2")

    if args.train_fraction <= 0.0 or args.train_fraction >= 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")

    cities = [c.strip() for c in args.city.split(",") if c.strip()]
    if not cities:
        raise ValueError("--city must include at least one city")

    df = pd.concat([load_processed_monthlies(city) for city in cities], ignore_index=True)
    df = df.sort_values(["date", "city", "tile_id"]).reset_index(drop=True)
    split_idx = max(1, int(len(df) * args.train_fraction))
    split_idx = min(split_idx, len(df) - 1)
    train_cutoff_date = df["date"].sort_values().iloc[split_idx - 1]
    train_mask = df["date"] <= train_cutoff_date

    df, norm_stats = normalize_features(df, train_mask, FEATURE_COLUMNS)
    out = build_windows(df, args.window_size)

    version_seed = {
        "columns": sorted(out.columns.tolist()),
        "cities": [city_key(city) for city in cities],
        "window_size": args.window_size,
        "train_fraction": args.train_fraction,
        "target_month_min": str(out["target_month"].min()) if len(out) else "",
        "target_month_max": str(out["target_month"].max()) if len(out) else "",
        "row_count": int(len(out)),
    }
    version_hash = hashlib.sha256(json.dumps(version_seed, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    dataset_version = f"v1-{version_hash}"
    out["dataset_version"] = dataset_version

    out_dir = ensure_dir(FEATURES_ROOT)
    out_path = out_dir / "dataset.parquet"
    out.to_parquet(out_path, index=False)

    stats_path = out_dir / "normalization_stats.json"
    stats_path.write_text(json.dumps(norm_stats, indent=2), encoding="utf-8")
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": dataset_version,
                "cities": [city_key(city) for city in cities],
                "window_size": args.window_size,
                "train_fraction_for_normalization": args.train_fraction,
                "train_cutoff_date": str(train_cutoff_date.date()),
                "rows": int(len(out)),
                "target_month_min": str(out["target_month"].min()) if len(out) else None,
                "target_month_max": str(out["target_month"].max()) if len(out) else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
