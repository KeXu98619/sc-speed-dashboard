"""
One-time preprocessing script for South Carolina Active Transportation flows.
Run once locally:  python preprocess_at_flows.py

Input:  data/sc_edge_flows_trip.geoparquet   (WKB-encoded LineString geometry)
Output: data/at_flows_preprocessed.parquet   (~10-30 MB)

To update with a new AT flow version: replace the input geoparquet file and
re-run this script.  The dashboard will pick up the new parquet on next load.
"""
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from shapely import from_wkb

INPUT  = r"E:\Projects\sc_safety\data\sc_edge_flows_trip.geoparquet"
OUTPUT = r"E:\Projects\sc_safety\data\at_flows_preprocessed.parquet"


def main() -> None:
    print("Loading AT flows geoparquet...")
    df = pq.read_table(INPUT).to_pandas()
    print(f"  {len(df):,} total rows")

    vol = df["bike_flow"].fillna(0.0) + df["foot_flow"].fillna(0.0)
    mask = vol > 0
    df  = df.loc[mask].reset_index(drop=True)
    vol = vol.loc[mask].reset_index(drop=True)
    print(f"  {len(df):,} rows with positive bike+ped volume")

    print("  Decoding WKB geometries...")
    geoms = from_wkb(df["geometry"].values)

    paths: list[str] = []
    vols:  list[float] = []
    skipped = 0

    for geom, v in zip(geoms, vol.values):
        if geom is None or geom.geom_type != "LineString":
            skipped += 1
            continue
        coords = [[round(c[0], 5), round(c[1], 5)] for c in geom.coords]
        if len(coords) < 2:
            skipped += 1
            continue
        paths.append(json.dumps(coords, separators=(",", ":")))
        vols.append(float(v))

    if skipped:
        print(f"  {skipped:,} rows skipped (non-LineString or too short)")

    out = pd.DataFrame({"path": paths, "bike_ped_volume": np.array(vols, dtype="float32")})

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    out.to_parquet(OUTPUT, index=False, compression="zstd")

    size_mb = os.path.getsize(OUTPUT) / 1024**2
    print(f"Saved {len(out):,} segments -> {OUTPUT} ({size_mb:.1f} MB)")
    print(
        f"Volume stats: min={out['bike_ped_volume'].min():.2f}, "
        f"p50={out['bike_ped_volume'].median():.2f}, "
        f"p90={out['bike_ped_volume'].quantile(0.9):.2f}, "
        f"max={out['bike_ped_volume'].max():.2f}"
    )


if __name__ == "__main__":
    main()
