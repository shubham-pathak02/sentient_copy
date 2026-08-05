param(
  [string]$ConfigPath = "config/pipeline.bengaluru.2020_2024.json",
  [ValidateSet("cnn", "tabular", "both")]
  [string]$ModelTrack = "both"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ConfigPath)) {
  throw "Config file not found: $ConfigPath"
}

$cfg = Get-Content $ConfigPath | ConvertFrom-Json
$city = $cfg.city
$bbox = $cfg.bbox
$startDate = $cfg.start_date
$endDate = $cfg.end_date
$windowSize = $cfg.window_size
$gridSize = $cfg.grid_size
$s1Scale = $cfg.sentinel1_scale
$s2Scale = $cfg.sentinel2_scale
$s2MaxCloud = $cfg.sentinel2_max_cloud

Write-Host "Running pipeline for $city ($startDate to $endDate)"

python -m src.ingestion.era5_ingest --city $city --bbox $bbox --start-date $startDate --end-date $endDate
python -m src.ingestion.osm_ingest --city $city --bbox $bbox --start-date $startDate --end-date $endDate
python -m src.ingestion.population_ingest --city $city --bbox $bbox --start-date $startDate --end-date $endDate
python -m src.ingestion.landsat_ingest --city $city --bbox $bbox --start-date $startDate --end-date $endDate
python -m src.ingestion.nightlights_ingest --city $city --bbox $bbox --start-date $startDate --end-date $endDate
python -m src.ingestion.sentinel1_ingest --city $city --bbox $bbox --start-date $startDate --end-date $endDate --scale $s1Scale
python -m src.ingestion.sentinel2_ingest --city $city --bbox $bbox --start-date $startDate --end-date $endDate --max-cloud $s2MaxCloud --scale $s2Scale

python -m src.preprocessing.validate_raw_data --city $city --start-date $startDate --end-date $endDate
python -m src.preprocessing.monthly_stress --city $city --bbox $bbox --start-date $startDate --end-date $endDate --grid-size $gridSize
python -m src.features.build_dataset --city $city --window-size $windowSize

if ($ModelTrack -eq "tabular" -or $ModelTrack -eq "both") {
  python -m src.training.train_models --dataset data/features/dataset.parquet --val-fraction 0.3
  python -m src.inference.score_risk --dataset data/features/dataset.parquet --model models/best_model.joblib
  python -m src.evaluation.evaluate --dataset data/features/dataset.parquet --predictions data/results/risk_scores.parquet --training-metrics models/training_metrics.json
  python -m src.features.build_road_risk --city $city --max-roads 900
}

if ($ModelTrack -eq "cnn" -or $ModelTrack -eq "both") {
  python -m src.training.train_cnn_temporal --dataset data/features/dataset.parquet --city $city --grid-size $gridSize
  python -m src.inference.score_risk_cnn_temporal --dataset data/features/dataset.parquet --model models/cnn_temporal_best.pt --city $city --grid-size $gridSize
}

python "scripts/generate_data_inventory.py"
Write-Host "Pipeline completed. See data/results and models for artifacts."
