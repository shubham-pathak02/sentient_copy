from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.common.paths import RESULTS_ROOT

app = FastAPI(title="Road Risk API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WEB_ROOT = Path(__file__).resolve().parents[1] / "frontend" / "web"

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "pipeline.bengaluru.2020_2024.json"
BBOX = [77.45, 12.8, 77.75, 13.1]
GRID_SIZE = 4


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {"bbox": "77.45,12.8,77.75,13.1", "grid_size": 4}


def _tile_to_center(tile_id: str, bbox: list[float], grid_size: int) -> tuple[float, float]:
    """Return (lon, lat) center of tile. tile_id format: tile_YY_XX."""
    try:
        _, sy, sx = str(tile_id).split("_")
        ty, tx = int(sy), int(sx)
    except Exception:
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    west, south, east, north = bbox
    lon = west + (east - west) * (tx + 0.5) / grid_size
    lat = south + (north - south) * (grid_size - ty - 0.5) / grid_size
    return lon, lat


def _load_scores(model: str = "tabular") -> pd.DataFrame:
    if model == "cnn_temporal" or model == "cnn":
        path = RESULTS_ROOT / "risk_scores_cnn_temporal.parquet"
    else:
        path = RESULTS_ROOT / "risk_scores.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No risk scores found: {path}")
    return pd.read_parquet(path)


@app.get("/metadata")
def metadata() -> dict[str, object]:
    manifest = RESULTS_ROOT / "data_inventory_manifest.json"
    validation = RESULTS_ROOT / "raw_data_validation.json"
    evaluation = RESULTS_ROOT / "evaluation.json"
    cfg = _load_config()
    payload: dict[str, object] = {
        "results_root": str(RESULTS_ROOT),
        "bbox": cfg.get("bbox", "77.45,12.8,77.75,13.1"),
        "grid_size": int(cfg.get("grid_size", 4)),
        "city": cfg.get("city", "Bengaluru"),
    }
    if manifest.exists():
        payload["data_inventory_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    if validation.exists():
        payload["raw_data_validation"] = json.loads(validation.read_text(encoding="utf-8"))
    if evaluation.exists():
        payload["evaluation"] = json.loads(evaluation.read_text(encoding="utf-8"))
    return payload


@app.get("/risk/latest")
def risk_latest(
    limit: int = Query(100, ge=1, le=2000),
    model: str = Query("tabular", description="tabular or cnn_temporal"),
) -> list[dict[str, object]]:
    try:
        df = _load_scores(model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    cfg = _load_config()
    bbox = [float(x) for x in str(cfg.get("bbox", "77.45,12.8,77.75,13.1")).split(",")]
    grid_size = int(cfg.get("grid_size", 4))

    cols = [c for c in ["tile_id", "zone_id", "road_segment_id", "risk_score", "model_track", "model_name", "target_month", "tile_risk", "zone_risk"] if c in df.columns]
    out = df.sort_values("risk_score", ascending=False)[cols].head(limit)
    records = out.to_dict(orient="records")
    for i, r in enumerate(records):
        r["rank"] = i + 1
        if "tile_id" in r:
            lon, lat = _tile_to_center(str(r["tile_id"]), bbox, grid_size)
            r["lon"] = lon
            r["lat"] = lat
    return records


@app.get("/risk/ranking")
def risk_ranking(
    limit: int = Query(100, ge=1, le=2000),
    model: str = Query("tabular", description="tabular or cnn_temporal"),
    by: str = Query("tile", description="tile, zone, or row"),
) -> list[dict[str, object]]:
    """Return risk-ranked list. by=tile aggregates by tile_id, by=zone by zone_id, by=row keeps each row."""
    try:
        df = _load_scores(model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    cfg = _load_config()
    bbox = [float(x) for x in str(cfg.get("bbox", "77.45,12.8,77.75,13.1")).split(",")]
    grid_size = int(cfg.get("grid_size", 4))

    if by == "zone" and "zone_id" in df.columns:
        agg = df.groupby("zone_id", as_index=False).agg(risk_score=("risk_score", "mean"))
        if "target_month" in df.columns:
            first_month = df.groupby("zone_id")["target_month"].first().reset_index()
            agg = agg.merge(first_month, on="zone_id", how="left")
        agg = agg.sort_values("risk_score", ascending=False).head(limit)
        records = agg.to_dict(orient="records")
        for i, r in enumerate(records):
            r["rank"] = i + 1
    elif by == "tile" and "tile_id" in df.columns:
        agg = df.groupby("tile_id", as_index=False).agg(risk_score=("risk_score", "mean"))
        agg = agg.sort_values("risk_score", ascending=False).head(limit)
        records = agg.to_dict(orient="records")
        for i, r in enumerate(records):
            r["rank"] = i + 1
            lon, lat = _tile_to_center(str(r["tile_id"]), bbox, grid_size)
            r["lon"] = lon
            r["lat"] = lat
    else:
        cols = [c for c in ["tile_id", "zone_id", "road_segment_id", "risk_score", "target_month"] if c in df.columns]
        out = df.sort_values("risk_score", ascending=False)[cols].head(limit)
        records = out.to_dict(orient="records")
        for i, r in enumerate(records):
            r["rank"] = i + 1
            if "tile_id" in r:
                lon, lat = _tile_to_center(str(r["tile_id"]), bbox, grid_size)
                r["lon"] = lon
                r["lat"] = lat
    return records


@app.get("/risk/by_zone")
def risk_by_zone(
    model: str = Query("tabular", description="tabular or cnn_temporal"),
) -> list[dict[str, object]]:
    try:
        df = _load_scores(model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if "zone_id" not in df.columns or "risk_score" not in df.columns:
        raise HTTPException(status_code=422, detail="risk score file missing required columns: zone_id, risk_score")

    by_zone = (
        df.groupby("zone_id", as_index=False)["risk_score"]
        .mean()
        .rename(columns={"risk_score": "zone_risk"})
        .sort_values("zone_risk", ascending=False)
    )
    by_zone["rank"] = range(1, len(by_zone) + 1)
    return by_zone.to_dict(orient="records")


@app.get("/risk/roads")
def risk_roads(
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, object]:
    """Return road-level risk ranking with geometry for map."""
    path = RESULTS_ROOT / "road_risk_ranking.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run build_road_risk first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    roads = data.get("roads", [])[:limit]
    return {"roads": roads, "total": len(data.get("roads", []))}


@app.get("/dashboard")
def dashboard() -> FileResponse:
    """Full precomputed payload for the interactive command-center UI."""
    path = RESULTS_ROOT / "dashboard.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run build_road_risk first.")
    return FileResponse(path, media_type="application/json")


@app.get("/risk/heatmap")
def risk_heatmap(
    model: str = Query("tabular", description="tabular or cnn_temporal"),
    target_month: str | None = Query(None),
    limit: int = Query(500, ge=1, le=5000),
) -> list[dict[str, object]]:
    """Return points with lon, lat, risk_score for map visualization."""
    try:
        df = _load_scores(model)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    cfg = _load_config()
    bbox = [float(x) for x in str(cfg.get("bbox", "77.45,12.8,77.75,13.1")).split(",")]
    grid_size = int(cfg.get("grid_size", 4))

    if target_month and "target_month" in df.columns:
        df = df[df["target_month"].astype(str) == target_month]
    tile_agg = df.groupby("tile_id", as_index=False)["risk_score"].mean()
    tile_agg = tile_agg.sort_values("risk_score", ascending=False).head(limit)
    out = []
    for _, row in tile_agg.iterrows():
        lon, lat = _tile_to_center(str(row["tile_id"]), bbox, grid_size)
        out.append({"tile_id": str(row["tile_id"]), "lon": lon, "lat": lat, "risk_score": float(row["risk_score"])})
    return out


# Interactive command-center UI (must be mounted last so API routes win).
if WEB_ROOT.exists():
    app.mount("/", StaticFiles(directory=str(WEB_ROOT), html=True), name="web")
