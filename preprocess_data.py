"""
One-time preprocessing for South Carolina road segment data.

All four TomTom periods share the same segment IDs and road geometry,
so geometry is written once and speed metrics are written per-period:

  data/sc_geometry.parquet              – segmentId, path, streetName, frc, speedLimit
  data/sc_2026q1_weekday_metrics.parquet
  data/sc_2026q1_weekend_metrics.parquet
  data/sc_2025q1_weekday_metrics.parquet
  data/sc_2025q1_weekend_metrics.parquet

Run once:
    python preprocess_data.py
"""
import json
import os
import numpy as np
import pandas as pd

BASE_DIR   = r"E:\Projects\sc_safety"
DATA_DIR   = os.path.join(BASE_DIR, "data")
TOMTOM_DIR = os.path.join(BASE_DIR, "TomTomData")

DATASETS = {
    "2026q1_weekday": os.path.join(TOMTOM_DIR, "SC_safety_analysis_2026q1_weekday.geojson"),
    "2026q1_weekend": os.path.join(TOMTOM_DIR, "SC_safety_analysis_2026q1_weekend.geojson"),
    "2025q1_weekday": os.path.join(TOMTOM_DIR, "SC_safety_analysis_2025q1_weekday.geojson"),
    "2025q1_weekend": os.path.join(TOMTOM_DIR, "SC_safety_analysis_2025q1_weekend.geojson"),
}
GEOMETRY_SOURCE = "2026q1_weekday"

TIME_BUCKETS = {
    "0:00 - 6:00":    list(range(2,  8)),
    "6:00 - 11:00":   list(range(8,  13)),
    "11:00 - 16:00":  list(range(13, 18)),
    "16:00 - 20:00":  list(range(18, 22)),
    "20:00 - 0:00":   list(range(22, 26)),
}
P85_IDX, P15_IDX = 16, 2
HOUR_TO_TIMESET  = {h: h + 2 for h in range(24)}


def process_geojson(path: str) -> tuple[list[dict], list[dict]]:
    """Return (geom_records, metrics_records) parsed from one GeoJSON file."""
    print(f"  Loading {os.path.basename(path)}...")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    features = raw.get("features", [])
    print(f"    {len(features):,} total features")

    geom_records: list[dict]    = []
    metrics_records: list[dict] = []

    for feat in features:
        geom  = feat.get("geometry")
        props = feat.get("properties") or {}
        if not geom or geom.get("type") != "LineString":
            continue
        coords = [[round(c[0], 5), round(c[1], 5)] for c in geom.get("coordinates", [])]
        if len(coords) < 2:
            continue
        results = props.get("segmentTimeResults") or []
        if not results:
            continue

        seg_id = str(props.get("segmentId", ""))
        ts_map = {r["timeSet"]: r for r in results if isinstance(r, dict)}
        total_ss = int(sum(r.get("sampleSize") or 0 for r in results if isinstance(r, dict)))

        geom_records.append({
            "segmentId":  seg_id,
            "path":       json.dumps(coords, separators=(",", ":")),
            "streetName": props.get("streetName") or "",
            "frc":        int(props["frc"]) if props.get("frc") is not None else -1,
            "speedLimit": float(props["speedLimit"]) if props.get("speedLimit") is not None else np.nan,
        })

        rec: dict = {"segmentId": seg_id, "total_sample_size": total_ss}

        for i, (_, timesets) in enumerate(TIME_BUCKETS.items()):
            pfx    = f"tod{i}"
            bucket = [ts_map[ts] for ts in timesets if ts in ts_map]
            if not bucket:
                for m in ("avg_speed", "ttr", "p85", "p15"):
                    rec[f"{m}_{pfx}"] = np.nan
                continue
            rec[f"avg_speed_{pfx}"] = float(np.nanmean([r.get("averageSpeed", np.nan) for r in bucket]))
            rec[f"ttr_{pfx}"]       = float(np.nanmean([r.get("travelTimeRatio", np.nan) for r in bucket]))
            p85s = [r["speedPercentiles"][P85_IDX] for r in bucket if len(r.get("speedPercentiles", [])) > P85_IDX]
            p15s = [r["speedPercentiles"][P15_IDX] for r in bucket if len(r.get("speedPercentiles", [])) > P15_IDX]
            rec[f"p85_{pfx}"] = float(np.mean(p85s)) if p85s else np.nan
            rec[f"p15_{pfx}"] = float(np.mean(p15s)) if p15s else np.nan

        for h in range(24):
            r = ts_map.get(HOUR_TO_TIMESET[h])
            if r and len(r.get("speedPercentiles", [])) > P85_IDX:
                rec[f"p85_h{h}"] = float(r["speedPercentiles"][P85_IDX])
            else:
                rec[f"p85_h{h}"] = np.nan

        metrics_records.append(rec)

    print(f"    {len(metrics_records):,} valid segments retained")
    return geom_records, metrics_records


def save_geometry(geom_records: list[dict]) -> None:
    df = pd.DataFrame(geom_records)
    df["speedLimit"] = df["speedLimit"].astype("float32")
    df["frc"]        = df["frc"].astype("int8")
    out = os.path.join(DATA_DIR, "sc_geometry.parquet")
    df.to_parquet(out, index=False, compression="zstd")
    mb = os.path.getsize(out) / 1024**2
    print(f"    geometry -> {out}  ({mb:.1f} MB, {len(df):,} segments)")


def save_metrics(metrics_records: list[dict], period_key: str) -> None:
    df = pd.DataFrame(metrics_records)
    float_cols = [c for c in df.columns if c.startswith(("avg_speed_", "ttr_", "p85_", "p15_", "p85_h"))]
    df[float_cols]          = df[float_cols].astype("float32")
    df["total_sample_size"] = df["total_sample_size"].astype("int32")
    out = os.path.join(DATA_DIR, f"sc_{period_key}_metrics.parquet")
    df.to_parquet(out, index=False, compression="zstd")
    mb = os.path.getsize(out) / 1024**2
    print(f"    metrics  -> {out}  ({mb:.1f} MB)")


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    for period_key, path in DATASETS.items():
        print(f"\nProcessing {period_key}...")
        geom_records, metrics_records = process_geojson(path)
        if period_key == GEOMETRY_SOURCE:
            save_geometry(geom_records)
        save_metrics(metrics_records, period_key)

    print("\nAll done.")
    geom = pd.read_parquet(os.path.join(DATA_DIR, "sc_geometry.parquet"))
    print(f"\nFRC distribution:\n{geom['frc'].value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    main()
