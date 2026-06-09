"""
One-time preprocessing for South Carolina H3 resolution-7 AT trip hexagon data.

Input:  data/sc_hex7_with_crashes_svi_v2.geojson
Output: data/h3_at_trips_preprocessed.parquet

Expected GeoJSON feature structure (Polygon geometry, one hex per feature):
  properties:
    geoId         – unique H3 hex identifier (string)
    daily_trips   – numeric string e.g. "0.942"
    past_crash    – "Past Fatal Crashes" | "No Past Record"
    SVI Rating    – "High" | "Medium" | "Low" | "NA"

Run once:  python preprocess_h3_trips.py
"""
import json
import os
import pandas as pd

INPUT  = r"E:\Projects\sc_safety\data\sc_hex7_with_crashes_svi_v2.geojson"
OUTPUT = r"E:\Projects\sc_safety\data\h3_at_trips_preprocessed.parquet"


def main() -> None:
    if not os.path.exists(INPUT):
        print(f"ERROR: Input file not found:\n  {INPUT}")
        return

    print("Loading H3 AT trips GeoJSON...")
    with open(INPUT, encoding="utf-8") as f:
        raw = json.load(f)

    features = raw.get("features", [])
    print(f"  {len(features):,} total features")

    records: list[dict] = []
    for feat in features:
        geom  = feat.get("geometry")
        props = feat.get("properties") or {}

        if not geom or geom.get("type") != "Polygon":
            continue

        outer_ring = geom["coordinates"][0]
        coords = [[round(c[0], 5), round(c[1], 5)] for c in outer_ring]

        trips_raw = props.get("daily_trips")
        if trips_raw is None:
            continue
        try:
            trips = float(trips_raw)
        except (ValueError, TypeError):
            continue
        if trips <= 0:
            continue

        geo_id = props.get("geoId", "")

        svi_raw = props.get("SVI Rating", "")
        svi     = "" if svi_raw in (None, "NA", "") else str(svi_raw)

        crash_raw    = props.get("past_crash", "")
        crash_rating = "" if crash_raw in (None, "") else str(crash_raw)

        records.append({
            "geoId":        geo_id,
            "daily_trips":  trips,
            "svi":          svi,
            "crash_rating": crash_rating,
            "polygon":      json.dumps(coords, separators=(",", ":")),
        })

    df = pd.DataFrame(records)
    df["daily_trips"] = df["daily_trips"].astype("float32")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_parquet(OUTPUT, index=False, compression="zstd")

    mb = os.path.getsize(OUTPUT) / 1024**2
    print(f"Saved {len(df):,} hexagons -> {OUTPUT} ({mb:.2f} MB)")
    print(
        f"daily_trips: min={df['daily_trips'].min():.2f}, "
        f"p50={df['daily_trips'].median():.2f}, "
        f"p90={df['daily_trips'].quantile(0.9):.2f}, "
        f"max={df['daily_trips'].max():.2f}"
    )
    print(f"SVI distribution:\n{df['svi'].replace('', '(blank)').value_counts().to_string()}")
    print(f"Crash history distribution:\n{df['crash_rating'].replace('', '(blank)').value_counts().to_string()}")


if __name__ == "__main__":
    main()
