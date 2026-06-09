"""
One-time preprocessing script for South Carolina AADT data.
Run once locally:  python preprocess_aadt.py

Output: data/aadt_lookup.parquet  (segmentId + AADT)

SC AADT segment IDs match TomTom segmentIds directly, so the dashboard
merges on segmentId instead of doing a spatial join.
"""
import json
import os
import pandas as pd

INPUT  = r"E:\Projects\sc_safety\data\sc_selected_counties_aadt.geojson"
OUTPUT = r"E:\Projects\sc_safety\data\aadt_lookup.parquet"


def main() -> None:
    print("Loading AADT GeoJSON...")
    with open(INPUT, encoding="utf-8") as f:
        raw = json.load(f)

    features = raw.get("features", [])
    print(f"  {len(features):,} total features")

    records: list[dict] = []
    for feat in features:
        geom  = feat.get("geometry")
        props = feat.get("properties") or {}

        if not geom or geom.get("type") != "LineString":
            continue

        seg_id   = props.get("id")
        aadt_val = props.get("aadt")
        if seg_id is None or aadt_val is None:
            continue

        records.append({
            "segmentId": str(seg_id),
            "aadt":      float(aadt_val),
        })

    df = pd.DataFrame(records)
    df["aadt"] = df["aadt"].astype("float32")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_parquet(OUTPUT, index=False, compression="zstd")

    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"Saved {len(df):,} segments -> {OUTPUT} ({size_mb:.2f} MB)")
    print(
        f"AADT: min={df['aadt'].min():.0f}, "
        f"p50={df['aadt'].median():.0f}, "
        f"p90={df['aadt'].quantile(0.9):.0f}, "
        f"max={df['aadt'].max():.0f}"
    )


if __name__ == "__main__":
    main()
