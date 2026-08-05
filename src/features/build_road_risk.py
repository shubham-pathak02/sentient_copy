"""
Build road-level risk by overlaying OSM roads on tile risk.
Output: ranked roads with geometry for map display.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import FEATURES_ROOT, RAW_ROOT, RESULTS_ROOT, ensure_dir


BBOX_BY_CITY = {
    "bengaluru": [77.45, 12.8, 77.75, 13.1],
    "mumbai": [72.65, 18.75, 73.10, 19.40],
    "hyderabad": [78.20, 17.20, 78.70, 17.65],
}
GRID_SIZE = 4


def city_key(city: str) -> str:
    return city.lower().replace(" ", "_")


def point_to_tile(lon: float, lat: float, bbox: list[float], city: str) -> str:
    west, south, east, north = bbox
    j = max(0, min(GRID_SIZE - 1, int((lon - west) / (east - west) * GRID_SIZE)))
    i = max(0, min(GRID_SIZE - 1, int((north - lat) / (north - south) * GRID_SIZE)))
    return f"{city}__tile_{i:02d}_{j:02d}"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    a = math.sin(math.radians(lat2 - lat1) / 2) ** 2
    a += math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


DRIVER_LABELS = {
    "rain": "Repeated heavy rainfall",
    "flood": "Water accumulation & drainage stress",
    "heat": "High thermal expansion exposure",
    "moisture": "Persistent surface moisture",
    "load": "Heavy usage corridor",
    "veg_loss": "Vegetation loss along corridor",
}

HIGH_ACTIONS = {
    "rain": "Preventive resurfacing before next monsoon",
    "flood": "Drainage audit within 2 weeks",
    "heat": "Crack & joint sealing within 30 days",
    "moisture": "Sub-surface moisture inspection within 30 days",
    "load": "Priority structural inspection within 30 days",
    "veg_loss": "Corridor stability inspection within 30 days",
}

def _tile_center(tile_id: str) -> tuple[float, float] | None:
    try:
        city, rest = tile_id.split("__", 1)
        _, sy, sx = rest.split("_")
        ty, tx = int(sy), int(sx)
    except Exception:
        return None
    bbox = BBOX_BY_CITY.get(city)
    if bbox is None:
        return None
    west, south, east, north = bbox
    lon = west + (east - west) * (tx + 0.5) / GRID_SIZE
    lat = north - (north - south) * (ty + 0.5) / GRID_SIZE
    return round(lon, 5), round(lat, 5)


def _months_to_critical(series: list[float | None], threshold: float) -> int | None:
    vals = [v for v in series if v is not None][-8:]
    if len(vals) < 4:
        return None
    current = vals[-1]
    if current >= threshold:
        return 0
    slope = float(np.polyfit(np.arange(len(vals)), np.asarray(vals, dtype=float), 1)[0])
    if slope <= 1e-4:
        return None
    months = math.ceil((threshold - current) / slope)
    return int(months) if months <= 24 else None


def _city_stress_series(dataset_path: Path, cities: list[str], months: list[str]) -> dict[str, dict[str, list]]:
    """Per-city monthly driver indices (0-100, rank-normalized across months)."""
    df = pd.read_parquet(dataset_path)
    df["city"] = df["tile_id"].astype(str).str.split("__").str[0]
    cols = {
        "rain": "stress_accum_rain_3m",
        "flood": "stress_accum_flood_3m",
        "heat": "stress_accum_heat_3m",
    }
    out: dict[str, dict[str, list]] = {}
    for city in cities:
        sub = df[df["city"] == city]
        monthly = sub.groupby(sub["target_month"].astype(str))[list(cols.values())].mean()
        series: dict[str, list] = {}
        for key, col in cols.items():
            vals = monthly[col].reindex(months)
            pct = vals.rank(pct=True) * 100.0
            series[key] = [None if pd.isna(v) else int(round(v)) for v in pct]
        out[city] = series
    return out


def _tile_drivers(dataset_path: Path) -> dict[str, dict[str, float]]:
    """Per-tile stress driver percentiles (0-100, ranked within each city)."""
    df = pd.read_parquet(dataset_path)
    df = df.sort_values("target_month").groupby("tile_id", as_index=False).last()

    raw = pd.DataFrame({"tile_id": df["tile_id"].astype(str)})
    raw["city"] = raw["tile_id"].str.split("__").str[0]
    raw["rain"] = df["stress_accum_rain_3m"].astype(float)
    raw["flood"] = df["stress_accum_flood_3m"].astype(float)
    raw["heat"] = df["stress_accum_heat_3m"].astype(float)
    raw["moisture"] = df.get("s2_ndwi_mean_lag0", pd.Series(0.0, index=df.index)).astype(float)
    load_cols = [c for c in ["nightlights_mean_lag0", "population_mean_lag0"] if c in df.columns]
    raw["load"] = df[load_cols].astype(float).mean(axis=1) if load_cols else 0.0
    trend_col = "s2_ndvi_mean_trend"
    raw["veg_loss"] = -df[trend_col].astype(float) if trend_col in df.columns else 0.0

    out: dict[str, dict[str, float]] = {}
    keys = list(DRIVER_LABELS.keys())
    for _, group in raw.groupby("city"):
        pct = group[keys].rank(pct=True) * 100.0
        for idx, tile_id in zip(group.index, group["tile_id"]):
            out[tile_id] = {k: round(float(pct.loc[idx, k]), 1) for k in keys}
    return out


def build_dashboard_payload(
    roads: list[dict],
    all_roads: list[dict],
    risk_df: pd.DataFrame,
    cities: list[str],
    city_cuts: dict[str, dict[str, float]],
) -> dict:
    tile_month = (
        risk_df.groupby([risk_df["tile_id"].astype(str), risk_df["target_month"].astype(str)])["risk_score"]
        .mean()
        .unstack()
    )
    months = sorted(tile_month.columns.tolist())
    tile_month = tile_month.reindex(columns=months)

    drivers_by_tile = _tile_drivers(FEATURES_ROOT / "dataset.parquet")
    stress_series = _city_stress_series(FEATURES_ROOT / "dataset.parquet", cities, months)

    model_name = str(risk_df["model_name"].iloc[0]) if "model_name" in risk_df.columns else "risk-model"

    for r in roads:
        tiles = [t for t in r.get("tiles", []) if t in tile_month.index]
        if tiles:
            series = tile_month.loc[tiles].mean(axis=0)
            trend = [None if pd.isna(v) else round(float(v), 3) for v in series]
        else:
            trend = [None] * len(months)
        r["trend"] = trend
        r["months_to_critical"] = _months_to_critical(trend, city_cuts[str(r["city"])]["high"])

        tile_drivers = [drivers_by_tile[t] for t in r.get("tiles", []) if t in drivers_by_tile]
        if tile_drivers:
            merged = {k: round(float(np.mean([d[k] for d in tile_drivers])), 1) for k in DRIVER_LABELS}
        else:
            merged = {k: 0.0 for k in DRIVER_LABELS}
        ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
        r["drivers"] = [
            {"key": k, "label": DRIVER_LABELS[k], "score": v} for k, v in ranked[:3] if v > 0
        ]

        top_key = ranked[0][0] if ranked else "rain"
        if r["risk_level"] == "High":
            r["action"] = HIGH_ACTIONS.get(top_key, "Priority inspection within 30 days")
        elif r["risk_level"] == "Medium":
            r["action"] = "Schedule inspection within 60 days"
        else:
            r["action"] = "Routine monitoring — next quarterly review"

    summaries: dict[str, dict] = {}
    for city in cities:
        city_all = [r for r in all_roads if r["city"] == city]
        city_sel = [r for r in roads if r["city"] == city]
        levels = {"High": 0, "Medium": 0, "Low": 0}
        for r in city_all:
            levels[r["risk_level"]] += 1

        city_tiles = [t for t in tile_month.index if t.startswith(city + "__")]
        monsoon_yoy = None
        if city_tiles:
            city_series = tile_month.loc[city_tiles].mean(axis=0)
            by_year: dict[str, list[float]] = {}
            for m, v in city_series.items():
                year, mon = m.split("-")
                if mon in {"06", "07", "08", "09"} and not pd.isna(v):
                    by_year.setdefault(year, []).append(float(v))
            years = sorted(y for y, vals in by_year.items() if vals)
            if len(years) >= 2:
                prev, latest = float(np.mean(by_year[years[-2]])), float(np.mean(by_year[years[-1]]))
                if prev > 1e-9:
                    monsoon_yoy = round((latest - prev) / prev * 100.0, 1)

        zone_risk: dict[str, list[float]] = {}
        for r in city_all:
            zone_risk.setdefault(r["zone"], []).append(r["risk_score"])
        worst_zone = max(zone_risk.items(), key=lambda kv: float(np.mean(kv[1])))[0] if zone_risk else None

        entering = [
            r for r in city_sel
            if r["months_to_critical"] is not None and 0 < r["months_to_critical"] <= 6
        ]

        top_decile_cut = float(np.quantile([r["risk_score"] for r in city_all], 0.9)) if city_all else 0.0
        top_decile = [r for r in city_all if r["risk_score"] >= top_decile_cut]
        rest = [r for r in city_all if r["risk_score"] < top_decile_cut]
        risk_concentration = None
        if top_decile and rest and float(np.mean([r["risk_score"] for r in rest])) > 1e-9:
            risk_concentration = round(
                float(np.mean([r["risk_score"] for r in top_decile]))
                / float(np.mean([r["risk_score"] for r in rest])),
                1,
            )

        summaries[city] = {
            "roads_analyzed": len(city_all),
            "levels": levels,
            "pct_high": round(levels["High"] / max(1, len(city_all)) * 100.0, 1),
            "avg_risk": round(float(np.mean([r["risk_score"] for r in city_all])), 3) if city_all else 0.0,
            "entering_critical_6m": len(entering),
            "critical_now": len([r for r in city_sel if r["months_to_critical"] == 0]),
            "monsoon_stress_yoy_pct": monsoon_yoy,
            "worst_zone": worst_zone,
            "risk_concentration": risk_concentration,
            "total_length_km": round(sum(r["length_m"] for r in city_all) / 1000.0, 1),
        }

    city_payload: dict[str, dict] = {}
    for city in cities:
        tiles = []
        for tile_id in tile_month.index:
            if not tile_id.startswith(city + "__"):
                continue
            center = _tile_center(tile_id)
            if center is None:
                continue
            series = [
                None if pd.isna(v) else round(float(v), 3) for v in tile_month.loc[tile_id]
            ]
            tiles.append({"id": tile_id, "lon": center[0], "lat": center[1], "series": series})

        city_payload[city] = {
            "label": city.replace("_", " ").title(),
            "bbox": BBOX_BY_CITY[city],
            "summary": summaries[city],
            "tiles": tiles,
            "stress_series": stress_series.get(city, {}),
        }

    road_out = []
    for r in roads:
        road_out.append({k: v for k, v in r.items() if k != "tiles"})

    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model_name": model_name,
        "city_cuts": city_cuts,
        "months": months,
        "cities": city_payload,
        "roads": road_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build road-level risk from OSM + tile risk.")
    parser.add_argument("--city", default="Bengaluru", help="City name or comma-separated city list.")
    parser.add_argument("--risk-scores", default=str(RESULTS_ROOT / "risk_scores.parquet"))
    parser.add_argument("--osm", default=None, help="Path to OSM JSON; auto-detect if not set")
    parser.add_argument("--max-roads", type=int, default=500)
    parser.add_argument("--min-length-m", type=float, default=50)
    args = parser.parse_args()

    risk_df = pd.read_parquet(args.risk_scores)
    # Rank roads by recent risk (last 6 predicted months), not the all-time mean.
    recent_months = sorted(risk_df["target_month"].astype(str).unique())[-6:]
    recent = risk_df[risk_df["target_month"].astype(str).isin(recent_months)]
    tile_risk = recent.groupby("tile_id", as_index=False)["risk_score"].mean()
    tile_risk_map = dict(zip(tile_risk["tile_id"].astype(str), tile_risk["risk_score"]))

    roads: list[dict] = []
    sources: dict[str, str] = {}
    cities = [city_key(c.strip()) for c in args.city.split(",") if c.strip()]
    for city in cities:
        bbox = BBOX_BY_CITY.get(city)
        if bbox is None:
            raise ValueError(f"No bbox configured for road risk city: {city}")

        osm_path = args.osm
        if not osm_path:
            osm_dir = RAW_ROOT / "osm"
            files = sorted(osm_dir.rglob(f"osm_roads_{city}_*.json"))
            if not files:
                raise FileNotFoundError(f"No OSM files for {city}")
            osm_path = files[-1]
        osm_path = Path(osm_path)
        sources[city] = str(osm_path)

        with open(osm_path, encoding="utf-8") as f:
            osm = json.load(f)

        nodes: dict[int, tuple[float, float]] = {}
        for el in osm.get("elements", []):
            if el.get("type") == "node" and "id" in el and "lat" in el and "lon" in el:
                nodes[int(el["id"])] = (float(el["lat"]), float(el["lon"]))

        for el in osm.get("elements", []):
            if el.get("type") != "way":
                continue
            tags = el.get("tags") or {}
            name = tags.get("name") or tags.get("ref") or "Unnamed Road"
            highway = tags.get("highway", "road")
            node_ids = el.get("nodes") or []
            if len(node_ids) < 2:
                continue

            path: list[tuple[float, float]] = []
            length_m = 0.0
            tiles_seen: set[str] = set()
            for i, nid in enumerate(node_ids):
                coord = nodes.get(int(nid))
                if coord is None:
                    continue
                lat, lon = coord
                path.append((lon, lat))
                if i > 0:
                    prev = nodes.get(int(node_ids[i - 1]))
                    if prev:
                        length_m += haversine_m(prev[0], prev[1], lat, lon)
                tiles_seen.add(point_to_tile(lon, lat, bbox, city))

            if length_m < args.min_length_m or len(path) < 2:
                continue

            path = [(round(lon, 5), round(lat, 5)) for lon, lat in path]

            risks = [tile_risk_map[t] for t in tiles_seen if t in tile_risk_map]
            risk_score = float(sum(risks) / len(risks)) if risks else 0.0

            roads.append({
                "city": city,
                "name": str(name),
                "highway": str(highway),
                "path": path,
                "tiles": sorted(tiles_seen),
                "length_m": round(length_m, 1),
                "risk_score": round(risk_score, 4),
                "risk_pct": round(risk_score * 100, 1),
            })

    # Risk tiers are relative within each city (top 15% High, next 35% Medium).
    # The system predicts relative prioritization, not absolute failure probability.
    city_cuts: dict[str, dict[str, float]] = {}
    for city in cities:
        scores = [r["risk_score"] for r in roads if r["city"] == city]
        if scores:
            city_cuts[city] = {
                "high": float(np.quantile(scores, 0.85)),
                "medium": float(np.quantile(scores, 0.50)),
            }
        else:
            city_cuts[city] = {"high": 0.7, "medium": 0.4}

    # Assign risk level and zone (quadrant) for geographic spread
    for r in roads:
        west, south, east, north = BBOX_BY_CITY[str(r["city"])]
        cuts = city_cuts[str(r["city"])]
        if r["risk_score"] >= cuts["high"]:
            r["risk_level"] = "High"
        elif r["risk_score"] >= cuts["medium"]:
            r["risk_level"] = "Medium"
        else:
            r["risk_level"] = "Low"
        # Zone from centroid (NW, NE, SW, SE)
        cx = sum(p[0] for p in r["path"]) / len(r["path"])
        cy = sum(p[1] for p in r["path"]) / len(r["path"])
        zone_x = "E" if cx > (west + east) / 2 else "W"
        zone_y = "N" if cy > (south + north) / 2 else "S"
        r["zone"] = zone_y + zone_x

    # Priority percentile vs ALL analyzed roads in the city (0-100).
    scores_by_city = {c: sorted(r["risk_score"] for r in roads if r["city"] == c) for c in cities}
    for r in roads:
        arr = scores_by_city[str(r["city"])]
        pct = bisect.bisect_left(arr, r["risk_score"]) / (len(arr) - 1) if len(arr) > 1 else 1.0
        r["priority"] = int(round(min(1.0, pct) * 100))

    all_roads = list(roads)

    # Stratified sampling: ensure High, Medium, Low from across the city
    selected: list[dict] = []
    city_quota = max(1, args.max_roads // max(1, len(cities)))
    for city in cities:
        city_roads = [r for r in roads if r["city"] == city]
        named = [r for r in city_roads if r["name"] != "Unnamed Road"]
        unnamed = [r for r in city_roads if r["name"] == "Unnamed Road"]
        high = [r for r in named + unnamed if r["risk_level"] == "High"]
        med = [r for r in named + unnamed if r["risk_level"] == "Medium"]
        low = [r for r in named + unnamed if r["risk_level"] == "Low"]
        high = sorted(high, key=lambda r: r["risk_score"], reverse=True)
        med = sorted(med, key=lambda r: r["risk_score"], reverse=True)
        low = sorted(low, key=lambda r: r["risk_score"], reverse=True)
        n_high = min(len(high), int(city_quota * 0.4))
        n_med = min(len(med), int(city_quota * 0.35))
        n_low = min(len(low), city_quota - n_high - n_med)
        chosen = high[:n_high] + med[:n_med] + low[:n_low]
        if len(chosen) < city_quota:
            seen = {id(r) for r in chosen}
            remainder = [r for r in sorted(city_roads, key=lambda r: r["risk_score"], reverse=True) if id(r) not in seen]
            chosen.extend(remainder[: city_quota - len(chosen)])
        selected.extend(chosen)

    if len(selected) < args.max_roads:
        seen = {id(r) for r in selected}
        remainder = [r for r in sorted(roads, key=lambda r: r["risk_score"], reverse=True) if id(r) not in seen]
        selected.extend(remainder[: args.max_roads - len(selected)])
    roads = selected[: args.max_roads]
    # Final sort: high first, then medium, then low; within tier by risk
    roads = sorted(roads, key=lambda r: (-(1 if r["risk_level"] == "High" else 0.5 if r["risk_level"] == "Medium" else 0), -r["risk_score"]))

    for i, r in enumerate(roads):
        r["rank"] = i + 1
        r["id"] = f"{r['city']}_{i + 1}"

    # Within-city rank by risk (what a commissioner actually cares about).
    for city in cities:
        city_roads = sorted((r for r in roads if r["city"] == city), key=lambda r: -r["risk_score"])
        for i, r in enumerate(city_roads):
            r["city_rank"] = i + 1

    out_dir = ensure_dir(RESULTS_ROOT)
    out_path = out_dir / "road_risk_ranking.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"roads": roads, "sources": sources, "tile_risk_source": args.risk_scores}, f, indent=2)
    print(f"Wrote {len(roads)} roads to {out_path}")

    dashboard = build_dashboard_payload(roads, all_roads, risk_df, cities, city_cuts)
    dash_path = out_dir / "dashboard.json"
    with open(dash_path, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, separators=(",", ":"))
    print(f"Wrote dashboard payload to {dash_path}")


if __name__ == "__main__":
    main()
