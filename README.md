# Minneapolis High-Risk Link Analysis

Interactive Streamlit dashboard for identifying high-risk road segments in the Minneapolis / Hennepin County area, based on TomTom probe speed data.

---

## Running the app

```bash
streamlit run minneapolis_speed_dashboard.py
```

Password is in `.streamlit/secrets.toml` (not committed — see below).

### `.streamlit/secrets.toml` (create this file locally)

```toml
password = "locus_minneapolis"
```

---

## Data already included in this repo

All preprocessing has been completed. These parquet files are committed and ready:

| File | Description | Size |
|---|---|---|
| `data/minneapolis_geometry.parquet` | Road segment geometry + street name + FRC + speed limit | 7.9 MB |
| `data/minneapolis_2026q1_weekday_metrics.parquet` | P85/P15 speeds, hourly speeds — 2026 Q1 weekdays | 16.8 MB |
| `data/minneapolis_2026q1_weekend_metrics.parquet` | Same — 2026 Q1 weekends | 15.7 MB |
| `data/minneapolis_2025q1_weekday_metrics.parquet` | Same — 2025 Q1 weekdays | 15.6 MB |
| `data/minneapolis_2025q1_weekend_metrics.parquet` | Same — 2025 Q1 weekends | 14.6 MB |
| `data/aadt_lookup.parquet` | AADT by segment ID (Hennepin County) | 1.3 MB |
| `data/at_flows_preprocessed.parquet` | Active transportation flows (bike + ped volume) | 8.1 MB |
| `data/h3_at_trips_preprocessed.parquet` | H3 hex overlay — AT trips, SVI, crash rating | 0.1 MB |

---

## Full preprocessing pipeline (for reference)

If you ever need to regenerate from raw data:

```bash
python preprocess_data.py       # TomTom → geometry + metrics parquets
python preprocess_aadt.py       # MN AADT GeoJSON → aadt_lookup.parquet
python preprocess_at_flows.py   # AT flows GeoJSON → at_flows_preprocessed.parquet
python preprocess_h3_trips.py   # H3 hex GeoJSON → h3_at_trips_preprocessed.parquet
```

Raw source files (not in repo — too large):
- `TomTomData/minneapolis_safety_analysis_*.geojson` (~3 GB each, 4 files)
- `data/mn_hennepin_county_aadt.geojson` (50 MB)
- `data/at_flows_hennepin.geojson` (218 MB)
- `data/minneapolis-h3-at-trips.geojson` (provided by coworker)
