"""
South Carolina High-Risk Link Analysis
Periods: 2026 Q1 Weekday / Weekend · 2025 Q1 Weekday / Weekend

Data sources (generate once with the preprocess scripts):
  python preprocess_data.py
  python preprocess_aadt.py
  python preprocess_at_flows.py
  python preprocess_h3_trips.py
"""
from __future__ import annotations

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────

GEOMETRY_PARQUET = "data/sc_geometry.parquet"
AADT_PARQUET     = "data/aadt_lookup.parquet"
AT_PARQUET       = "data/at_flows_preprocessed.parquet"
H3_PARQUET       = "data/h3_at_trips_preprocessed.parquet"

PERIOD_DAYS = {
    "2026q1_weekday": 64,
    "2026q1_weekend": 26,
    "2025q1_weekday": 64,
    "2025q1_weekend": 26,
}

TOD_LABELS = [
    "0:00 – 6:00",
    "6:00 – 11:00",
    "11:00 – 16:00",
    "16:00 – 20:00",
    "20:00 – 0:00",
]

FRC_LABELS = {
    0: "0 – Motorway / Freeway",
    1: "1 – Major Road",
    2: "2 – Other Major Road",
    3: "3 – Secondary Road",
    4: "4 – Local Connecting",
    5: "5 – Local High Importance",
    6: "6 – Local Road",
    7: "7 – Minor Local",
}

CRASH_OPTIONS = ["Past Fatal Crashes", "No Past Record"]
SVI_OPTIONS   = ["High", "Medium", "Low"]

MAP_STYLE    = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
SC_CENTER    = pdk.ViewState(latitude=33.85, longitude=-80.95, zoom=9, pitch=0)
TOOLTIP_STYLE = {
    "backgroundColor": "#1a1a2e", "color": "#ffffff",
    "fontSize": "13px", "padding": "8px 12px",
    "borderRadius": "6px", "lineHeight": "1.7",
}


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data...")
def load_data(period_key: str) -> tuple[pd.DataFrame, bool, list[int]]:
    geom_df    = pd.read_parquet(GEOMETRY_PARQUET)
    metrics_df = pd.read_parquet(f"data/sc_{period_key}_metrics.parquet")
    df = geom_df.merge(metrics_df, on="segmentId", how="left")

    try:
        aadt_df = pd.read_parquet(AADT_PARQUET)[["segmentId", "aadt"]]
        df = df.merge(aadt_df, on="segmentId", how="left")
    except FileNotFoundError:
        df["aadt"] = np.nan

    df["path"]    = df["path"].apply(json.loads)
    show_street   = (df["streetName"] == "").mean() < 0.50
    available_frc = sorted(int(x) for x in df["frc"].dropna().unique() if int(x) >= 0)
    return df, show_street, available_frc


@st.cache_data(show_spinner="Loading AT flows...")
def load_at_flows() -> pd.DataFrame:
    try:
        df = pd.read_parquet(AT_PARQUET)
        df["path"] = df["path"].apply(json.loads)
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["path", "bike_ped_volume"])


@st.cache_data(show_spinner="Loading H3 AT trips...")
def load_h3_trips() -> pd.DataFrame:
    try:
        df = pd.read_parquet(H3_PARQUET)
        df["polygon"] = df["polygon"].apply(json.loads)
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["geoId", "daily_trips", "polygon"])


# ── Color helpers ──────────────────────────────────────────────────────────────

def make_risk_colors(pct_diff: np.ndarray, threshold_pct: float) -> list:
    """Orange (just above threshold) → Dark red (far above). Scaled on 95th pct."""
    p95  = float(np.nanpercentile(pct_diff, 95))
    norm = np.clip((pct_diff - threshold_pct) / max(p95 - threshold_pct, 1e-8), 0.0, 1.0)
    r = np.full(len(norm), 220, dtype=float)
    g = 160.0 * (1.0 - norm)
    b = np.zeros_like(norm)
    a = np.full_like(norm, 230.0)
    return np.stack([
        np.clip(r, 0, 255).astype(np.uint8),
        np.clip(g, 0, 255).astype(np.uint8),
        b.astype(np.uint8),
        a.astype(np.uint8),
    ], axis=1).tolist()


def make_hex_colors(trips: np.ndarray) -> list:
    """Pale green (low) -> deep green (high), normalised to p95."""
    p95  = float(np.nanpercentile(trips, 95)) or 1.0
    norm = np.clip(trips / p95, 0.0, 1.0)
    r = (200.0 * (1.0 - norm)).astype(np.uint8)
    g = (180.0 + 60.0 * norm).astype(np.uint8)
    b = (160.0 * (1.0 - norm)).astype(np.uint8)
    a = (80.0  + 140.0 * norm).astype(np.uint8)
    return np.stack([r, g, b, a], axis=1).tolist()


def get_hex_attrs_for_links(paths: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (daily_trips, svi, crash_rating) arrays matched to the H3 hex under each path midpoint."""
    from shapely.geometry import Point, Polygon
    from shapely.strtree import STRtree
    h3_df = load_h3_trips()
    n = len(paths)
    trips_out  = np.full(n, np.nan)
    svi_out    = np.full(n, "", dtype=object)
    crash_out  = np.full(n, "", dtype=object)
    if len(h3_df) == 0:
        return trips_out, svi_out, crash_out
    polys     = [Polygon(coords) for coords in h3_df["polygon"]]
    tree      = STRtree(polys)
    trips_arr = h3_df["daily_trips"].to_numpy(dtype=float)
    svi_arr   = h3_df["svi"].to_numpy(dtype=object)
    crash_arr = (
        h3_df["crash_rating"].to_numpy(dtype=object)
        if "crash_rating" in h3_df.columns
        else np.full(len(h3_df), "", dtype=object)
    )
    for i, path in enumerate(paths):
        mid  = path[len(path) // 2]
        idxs = tree.query(Point(mid[0], mid[1]), predicate="within")
        if len(idxs):
            trips_out[i] = trips_arr[idxs[0]]
            svi_out[i]   = svi_arr[idxs[0]]
            crash_out[i] = crash_arr[idxs[0]]
    return trips_out, svi_out, crash_out


def at_widths(volume: np.ndarray) -> np.ndarray:
    """120–650 m, normalised from filtered-data min to p90."""
    vol_min = float(volume.min())
    p90     = float(np.nanpercentile(volume, 90))
    norm    = np.clip((volume - vol_min) / max(p90 - vol_min, 1e-8), 0.0, 1.0)
    return (120.0 + norm * 530.0).round(0)


# ── Heatmap matrix ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Computing heatmap...")
def compute_heatmap_matrix(
    df: pd.DataFrame, frc_tuple: tuple
) -> tuple[np.ndarray, np.ndarray]:
    subset   = df[df["frc"].isin(frc_tuple)]
    p85_cols = [f"p85_h{h}" for h in range(24)]
    sl       = subset["speedLimit"].to_numpy(dtype=np.float32)
    valid    = ~np.isnan(sl) & (sl > 0)

    p85_mat = subset.loc[valid, p85_cols].to_numpy(dtype=np.float32)
    excess  = (p85_mat - sl[valid][:, None]).astype(np.float32)
    n_valid = np.sum(~np.isnan(p85_mat), axis=0)
    del p85_mat

    scan_thresh = np.arange(5, 36, 1)
    matrix = np.zeros((24, len(scan_thresh)))
    for j, t in enumerate(scan_thresh):
        flagged      = np.nansum(excess >= t, axis=0)
        matrix[:, j] = np.where(n_valid > 0, flagged / n_valid * 100, 0.0)
    return matrix, scan_thresh


# ── Shared map builder ─────────────────────────────────────────────────────────

def render_deck(layers: list, tooltip: dict, height: int = 720,
                view_state: pdk.ViewState | None = None) -> None:
    st.pydeck_chart(
        pdk.Deck(layers=layers, initial_view_state=view_state or SC_CENTER,
                 tooltip=tooltip, map_style=MAP_STYLE),
        use_container_width=True, height=height,
    )


def make_tooltip(show_street: bool, rows: list[str]) -> dict:
    parts = []
    if show_street:
        parts.append("<b>{streetName}</b>")
    parts += rows
    return {"html": "<br/>".join(parts), "style": TOOLTIP_STYLE}


def legend_html(bad_str: str, good_str: str,
                gradient: str = "linear-gradient(to right, #dc0000, #ffd000, #00b400)",
                note: str = "") -> str:
    return f"""
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:6px; flex-wrap:wrap;">
      <div style="display:flex; align-items:center; gap:6px;">
        <span style="color:#555; font-size:11px;">{bad_str}</span>
        <div style="width:110px; height:10px; background:{gradient}; border-radius:3px;"></div>
        <span style="color:#555; font-size:11px;">{good_str}</span>
      </div>
      {f'<span style="color:#888; font-size:11px;">{note}</span>' if note else ''}
    </div>
    """


# ── App entry ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="South Carolina High-Risk Link Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def check_password() -> bool:
    if st.session_state.get("authenticated"):
        return True
    st.title("South Carolina High-Risk Link Analysis")
    pwd = st.text_input("Password", type="password", key="pwd_input")
    if st.button("Login"):
        if pwd == st.secrets.get("password", ""):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()

at_df_full = load_at_flows()
h3_df_full = load_h3_trips()

st.title("South Carolina High-Risk Link Analysis")

_qcol, _dcol, _ = st.columns([1, 1, 4])
quarter  = _qcol.radio("Quarter",  ["2026 Q1", "2025 Q1"])
day_type = _dcol.radio("Day Type", ["Weekday", "Weekend"])
period_key = f"{quarter.replace(' ', '').lower()}_{day_type.lower()}"
df, show_street, available_frc = load_data(period_key)
DEFAULT_FRC = [x for x in [1, 2, 3] if x in available_frc]
st.caption(f"{len(df):,} road segments · {quarter} {day_type}")

ctrl, main_col = st.columns([1, 5], gap="medium")

with ctrl:
    st.subheader("Filters")
    t2_frc = st.multiselect(
        "Road Class (FRC)", options=available_frc, default=DEFAULT_FRC,
        format_func=lambda x: FRC_LABELS.get(x, f"FRC {x}"),
    )
    t2_tod = st.selectbox("Time of Day (map)", TOD_LABELS, index=1)
    risk_threshold = st.slider(
        "Risk Threshold (% over speed limit)",
        min_value=10, max_value=100, value=50, step=5,
        help="Show segments where P85 exceeds posted speed limit by at least this %.",
    )
    st.divider()
    show_at = st.checkbox("Show AT Flows overlay", value=False)
    if show_at:
        at_min_vol = st.slider("Min bike & ped volume", min_value=0, max_value=500, value=1, step=1)
    else:
        at_min_vol = 1

    show_h3 = st.checkbox("Show AT Trips (Hex) overlay", value=False)
    if show_h3 and len(h3_df_full) > 0:
        h3_min_trips = st.slider("Min daily AT trips (hex)", min_value=0, max_value=300, value=0, step=10)
        st.caption(
            f"Pale green = low · Deep green = high "
            f"(p95 ≈ {h3_df_full['daily_trips'].quantile(0.95):.1f} trips/day)"
        )
    else:
        h3_min_trips = 0

    st.divider()
    if st.button("Reset map view", use_container_width=True):
        for k in ("_map_lat", "_map_lon", "_map_zoom", "_sel_row"):
            st.session_state.pop(k, None)
        st.rerun()

if not t2_frc:
    main_col.warning("Select at least one road class.")
    st.stop()

with main_col:

    # ── Heatmap ───────────────────────────────────────────────────────────────
    st.subheader("P85 Speed Exceedance by Hour and Threshold")

    _hm_cols = ["frc", "speedLimit"] + [f"p85_h{h}" for h in range(24)]
    matrix, scan_thresh = compute_heatmap_matrix(df[_hm_cols], tuple(sorted(t2_frc)))

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(
        matrix, aspect="auto", origin="lower", cmap="YlOrRd",
        extent=[scan_thresh[0] - 0.5, scan_thresh[-1] + 0.5, -0.5, 23.5],
    )
    try:
        cs = ax.contour(scan_thresh, np.arange(24), matrix,
                        levels=[1, 3], colors=["#3388ff", "#22bb44"],
                        linestyles=["--", "--"], linewidths=1.5)
        ax.clabel(cs, fmt={1: "1%", 3: "3%"}, fontsize=8)
    except Exception:
        pass
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("% of segments flagged")
    ax.set_xlabel("Threshold (mph above posted speed limit)")
    ax.set_ylabel("Hour of day")
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_title(
        "% of Segments Where P85 Speed > Posted Limit + Threshold\n"
        "(blue dashed = top 1%,  green dashed = top 3%)"
    )
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()

    # ── Risk map ──────────────────────────────────────────────────────────────
    pfx2     = f"tod{TOD_LABELS.index(t2_tod)}"
    _p85_col = f"p85_{pfx2}"
    _valid   = (
        df["frc"].isin(t2_frc)
        & df["speedLimit"].notna() & (df["speedLimit"] > 0)
        & df[_p85_col].notna()
    )
    n_total = int(_valid.sum())

    _pct      = ((df.loc[_valid, _p85_col] / df.loc[_valid, "speedLimit"]) - 1) * 100
    _risk_idx = _pct.index[_pct >= risk_threshold]

    _need   = ["path", "streetName", "frc", "speedLimit", "aadt", _p85_col]
    risk_df = df.loc[_risk_idx, _need].copy()
    risk_df["pct_diff"] = _pct.loc[_risk_idx].round(0).astype(int).values

    n_flagged = len(risk_df)

    st.subheader(
        f"High-Risk Links: P85 > Speed Limit + {risk_threshold}%  ·  {t2_tod}"
    )
    st.markdown(legend_html(
        f"{risk_threshold}% (threshold)",
        "Higher risk",
        gradient="linear-gradient(to right, #dc8000, #dc0000)",
        note="Color & thickness ∝ % over speed limit",
    ), unsafe_allow_html=True)
    st.caption(
        f"{n_flagged:,} of {n_total:,} segments flagged  "
        f"({n_flagged / max(n_total, 1) * 100:.1f}%)  ·  "
        f"threshold: P85 > speed limit × {1 + risk_threshold / 100:.2f}"
    )

    if n_flagged == 0:
        st.info("No segments meet the current threshold. Try lowering it.")
    else:
        pct_arr = risk_df["pct_diff"].to_numpy(dtype=float)
        risk_df["color"]    = make_risk_colors(pct_arr, risk_threshold)
        risk_df["p85_disp"] = risk_df[_p85_col].round(0).astype("Int64").values
        risk_df["sl_disp"]  = risk_df["speedLimit"].round(0).astype("Int64").values
        risk_df["aadt_disp"] = pd.array(
            [int(round(v)) if not np.isnan(v) else pd.NA for v in risk_df["aadt"].to_numpy(dtype=float)],
            dtype="Int64",
        )

        _hex_trips, _hex_svi, _hex_crash = get_hex_attrs_for_links(risk_df["path"].tolist())
        risk_df["trips_disp"] = pd.array(
            [int(round(v)) if not np.isnan(v) else pd.NA for v in _hex_trips],
            dtype="Int64",
        )
        risk_df["svi_disp"]   = _hex_svi
        risk_df["crash_disp"] = _hex_crash

        risk_layer = pdk.Layer(
            "PathLayer",
            data=risk_df[["path", "streetName", "frc", "sl_disp",
                           "p85_disp", "pct_diff", "trips_disp", "color"]],
            get_path="path", get_color="color", get_width=600,
            width_min_pixels=4, width_max_pixels=7,
            pickable=True, auto_highlight=True,
            highlight_color=[255, 255, 0, 80],
        )

        layers = []

        if show_h3 and len(h3_df_full) > 0:
            h3_plot = h3_df_full[h3_df_full["daily_trips"] >= h3_min_trips].copy()
            if len(h3_plot) > 0:
                h3_plot["color"] = make_hex_colors(h3_plot["daily_trips"].to_numpy(dtype=float))
                layers.append(pdk.Layer(
                    "PolygonLayer",
                    data=h3_plot[["polygon", "color"]],
                    get_polygon="polygon",
                    get_fill_color="color",
                    get_line_color=[0, 0, 0, 0],
                    line_width_min_pixels=0,
                    pickable=False,
                ))

        if show_at and len(at_df_full) > 0:
            at_filtered = at_df_full[at_df_full["bike_ped_volume"] >= at_min_vol].copy()
            if len(at_filtered) > 0:
                at_filtered["width"] = at_widths(at_filtered["bike_ped_volume"].to_numpy(dtype=float))
                layers.append(pdk.Layer(
                    "PathLayer",
                    data=at_filtered[["path", "width"]],
                    get_path="path",
                    get_color=[30, 100, 220, 170],
                    get_width="width",
                    width_min_pixels=1, width_max_pixels=6,
                    pickable=False,
                ))

        layers.append(risk_layer)

        risk_tt = make_tooltip(show_street, [
            "FRC: {frc}",
            "Speed Limit: {sl_disp} mph",
            "P85 Speed: {p85_disp} mph",
            "% Over Limit: {pct_diff}%",
            "AT Trips (hex): {trips_disp}",
        ])
        _lat  = st.session_state.get("_map_lat", SC_CENTER.latitude)
        _lon  = st.session_state.get("_map_lon", SC_CENTER.longitude)
        _zoom = st.session_state.get("_map_zoom", SC_CENTER.zoom)
        render_deck(layers, risk_tt, height=650,
                    view_state=pdk.ViewState(latitude=_lat, longitude=_lon, zoom=_zoom, pitch=0))

        st.divider()

        # ── Top N table ───────────────────────────────────────────────────────
        tbl_col1, tbl_col2, tbl_col3, tbl_col4 = st.columns(4)
        top_n    = tbl_col1.slider("Top N high-risk links",
                                   min_value=20, max_value=100, value=50, step=10)
        min_aadt = tbl_col2.slider("Min AADT (table filter)",
                                   min_value=0, max_value=5000, value=0, step=50)
        crash_filter = tbl_col3.multiselect(
            "Crash History (table filter)",
            options=CRASH_OPTIONS,
            default=CRASH_OPTIONS,
        )
        svi_filter = tbl_col4.multiselect(
            "SVI Rating (table filter)",
            options=SVI_OPTIONS,
            default=SVI_OPTIONS,
        )

        st.subheader(f"Top {top_n} High-Risk Links")
        st.caption("Click a row to zoom the map to that link. Click again to deselect.")

        # Build table with a hidden _pos column so we can trace back to risk_df paths.
        risk_df_r = risk_df.reset_index(drop=True)
        table_df = pd.DataFrame({
            "Street Name":            risk_df_r["streetName"].replace("", "—").values,
            "FRC":                    risk_df_r["frc"].map(
                                          lambda x: FRC_LABELS.get(int(x), f"FRC {x}")
                                      ).values,
            "Speed Limit (mph)":      risk_df_r["sl_disp"].values,
            "P85 Speed (mph)":        risk_df_r["p85_disp"].values,
            "% Over Limit":           risk_df_r["pct_diff"].values,
            "AADT":                   risk_df_r["aadt_disp"].values,
            "Active Trans Trips (hex)": risk_df_r["trips_disp"].values,
            "SVI Rating (hex)":       np.where(
                                          risk_df_r["svi_disp"].values == "",
                                          "—",
                                          risk_df_r["svi_disp"].values,
                                      ),
            "Crash History (hex)":    np.where(
                                          risk_df_r["crash_disp"].values == "",
                                          "—",
                                          risk_df_r["crash_disp"].values,
                                      ),
            "_pos":                   np.arange(len(risk_df_r)),
        }).sort_values("% Over Limit", ascending=False)

        # Apply table filters
        if min_aadt > 0:
            table_df = table_df[table_df["AADT"].notna() & (table_df["AADT"] >= min_aadt)]

        if crash_filter and len(crash_filter) < len(CRASH_OPTIONS):
            table_df = table_df[table_df["Crash History (hex)"].isin(crash_filter)]

        if svi_filter and len(svi_filter) < len(SVI_OPTIONS):
            table_df = table_df[table_df["SVI Rating (hex)"].isin(svi_filter)]

        table_df = table_df.head(top_n)

        # Build a list of paths aligned to the current table row order.
        table_paths = [risk_df_r.iloc[pos]["path"] for pos in table_df["_pos"].tolist()]

        table_df = table_df.drop(columns=["_pos"]).reset_index(drop=True)

        event = st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=False,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Speed Limit (mph)":          st.column_config.NumberColumn(format="%d mph"),
                "P85 Speed (mph)":            st.column_config.NumberColumn(format="%d mph"),
                "% Over Limit":               st.column_config.NumberColumn(format="%d%%"),
                "AADT":                       st.column_config.NumberColumn(format="%d"),
                "Active Trans Trips (hex)":   st.column_config.NumberColumn(format="%d"),
                "Crash History (hex)":        st.column_config.TextColumn(),
                "SVI Rating (hex)":           st.column_config.TextColumn(),
            },
        )

        # Handle row selection → zoom map to that link.
        sel_rows = event.selection.rows
        new_sel  = sel_rows[0] if sel_rows else None
        old_sel  = st.session_state.get("_sel_row")
        if new_sel != old_sel:
            st.session_state["_sel_row"] = new_sel
            if new_sel is not None and new_sel < len(table_paths):
                path = table_paths[new_sel]
                mid  = path[len(path) // 2]
                st.session_state["_map_lon"]  = mid[0]
                st.session_state["_map_lat"]  = mid[1]
                st.session_state["_map_zoom"] = 14
            else:
                for k in ("_map_lat", "_map_lon", "_map_zoom"):
                    st.session_state.pop(k, None)
            st.rerun()
