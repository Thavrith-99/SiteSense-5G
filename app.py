"""
SiteSense 5G - GeoAI decision-support dashboard
Team Neural Shield (KH-002) - ASEAN GeoAI Fusion 2026

A Streamlit + Leafmap dashboard that fuses tower locations, signal KPIs,
population and terrain to:
  (1) map current coverage and quantify people / kampungs out of coverage;
  (2) flag overloaded or underperforming towers;
  (3) recommend the best location for a new 5G tower.

Run locally:
    py -m streamlit run app.py

Layout inspiration: "ANDROMEDA" network-analytics dashboard
  (left filter sidebar - central map - right status panel).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# --- Map backend: Leafmap (folium backend) with a plain-folium fallback ----
# leafmap.foliumap.Map subclasses folium.Map, so st_folium can render either.
try:
    import leafmap.foliumap as leafmap
    HAVE_LEAFMAP = True
except Exception:  # pragma: no cover - fallback if leafmap unavailable
    import folium
    HAVE_LEAFMAP = False

from streamlit_folium import st_folium

import coverage as cov
import recommend as rec

# --- Paths ----------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DATA = APP_DIR / "data"
CSV = DATA / "towers_penang" / "502.csv"

# --- Study-area scopes (the sidebar switches between these) ----------------
SCOPES = {
    "Penang State": {
        "boundary": DATA / "boundaries" / "penang_state.geojson",
        "pop": DATA / "population" / "penang_state_ppp_2020.tif",
        "places": DATA / "villages" / "penang_places_state.geojson",
        "center": [5.30, 100.40], "zoom": 11,
        "note": "island + mainland Seberang Perai",
    },
    "Penang Island": {
        "boundary": DATA / "boundaries" / "penang_island.geojson",
        "pop": DATA / "population" / "penang_island_ppp_2020.tif",
        "places": DATA / "villages" / "penang_places_island.geojson",
        "center": [5.37, 100.27], "zoom": 12,
        "note": "island only",
    },
}

# --- Penang bounding box (coarse pre-filter before the polygon clip) -------
LAT_MIN, LAT_MAX = 5.1, 5.6
LON_MIN, LON_MAX = 100.1, 100.6

RADIO_META = {
    "NR":   ("5G NR",   "#e53935"),   # red   - the ones we care about
    "LTE":  ("4G LTE",  "#1e88e5"),   # blue
    "UMTS": ("3G UMTS", "#ffb300"),   # amber
    "GSM":  ("2G GSM",  "#9e9e9e"),   # grey
}


# --------------------------------------------------------------------------
# Data layer
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_towers(scope: str) -> pd.DataFrame:
    """Load OpenCelliD Malaysia cells and clip to the chosen scope polygon."""
    import json
    from shapely.geometry import shape, Point
    from shapely.prepared import prep

    df = pd.read_csv(CSV)
    penang = df[
        df["lat"].between(LAT_MIN, LAT_MAX)
        & df["lon"].between(LON_MIN, LON_MAX)
    ].copy()
    poly = prep(shape(json.loads(SCOPES[scope]["boundary"].read_text(encoding="utf-8"))
                      ["features"][0]["geometry"]))
    inside = [poly.contains(Point(x, y)) for x, y in zip(penang["lon"], penang["lat"])]
    penang = penang[inside].copy()
    penang["label"] = penang["radio"].map(lambda r: RADIO_META.get(r, (r, ""))[0])
    return penang


@st.cache_data(show_spinner=False)
def load_population(scope: str):
    return cov.load_population(SCOPES[scope]["pop"])


@st.cache_data(show_spinner=False)
def load_places(scope: str):
    return cov.load_places(SCOPES[scope]["places"])


@st.cache_data(show_spinner="Computing coverage gap…")
def compute_gap(scope: str, picked_key: tuple, max_range: int):
    """Function 1 — coverage gap for the currently-filtered towers."""
    df = load_towers(scope)
    sub = df[df["radio"].isin(picked_key) & (df["range"] <= max_range)]
    pop, transform = load_population(scope)
    places = load_places(scope)
    return cov.coverage_gap(
        pop, transform, places,
        sub["lon"].values, sub["lat"].values, sub["range"].values,
        SCOPES[scope]["center"][0],
    )


@st.cache_data(show_spinner="Scoring candidate sites…")
def compute_sites(scope: str, picked_key: tuple, max_range: int, load_p: int,
                  new_range: int, n_sites: int):
    """Function 3 — greedy max-coverage new-5G-tower recommendations."""
    df = load_towers(scope)
    sub = df[df["radio"].isin(picked_key) & (df["range"] <= max_range)]
    pop, transform = load_population(scope)
    g = compute_gap(scope, picked_key, max_range)
    thr = df["samples"].quantile(load_p / 100.0)
    ov = sub[sub["samples"] >= thr]
    return rec.recommend_sites(
        pop, g["coverage_mask"], transform, SCOPES[scope]["center"][0],
        new_range, n_sites, ov["lon"].values, ov["lat"].values,
    )


# --------------------------------------------------------------------------
# Page config + styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="SiteSense 5G - Penang",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.6rem; padding-bottom: 1rem; }
      .app-title { font-size: 1.55rem; font-weight: 700; line-height: 1.45;
        color: #f1f4f8; margin: 0 0 2px 0; padding-top: 4px; }
      .app-sub { color: #8b96a5; font-size: 0.85rem; margin: 0 0 10px 0; }
      .kpi-card {
        background: #11151c; border: 1px solid #232a36; border-radius: 12px;
        padding: 14px 16px; height: 100%;
      }
      .kpi-label { color: #8b96a5; font-size: 0.78rem; text-transform: uppercase;
        letter-spacing: .04em; margin-bottom: 4px; }
      .kpi-value { color: #f1f4f8; font-size: 1.7rem; font-weight: 700; line-height: 1.1; }
      .kpi-sub { color: #6f7b8a; font-size: 0.72rem; margin-top: 2px; }
      .panel {
        background: #11151c; border: 1px solid #232a36; border-radius: 12px;
        padding: 14px 16px; margin-bottom: 12px;
      }
      .panel h4 { margin: 0 0 8px 0; font-size: 0.9rem; color: #cfd6e0; }
      .ai-insight {
        background: #0e1a17; border: 1px solid #1c3b32; border-radius: 10px;
        padding: 12px 14px; color: #bfe8d8; font-size: 0.85rem;
      }
      .todo { color: #d9a441; }
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
if not CSV.exists():
    st.error(f"Cannot find 502.csv at:\n{CSV}")
    st.stop()

# --------------------------------------------------------------------------
# Sidebar - scope + filters (Function inputs)
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📡 SiteSense 5G")
    st.caption("GeoAI tower-siting & 5G coverage-gap planner")
    st.divider()

    scope = st.radio(
        "Study area", list(SCOPES.keys()), index=0,
        help="Switch between the full Penang State (island + mainland "
             "Seberang Perai) and Penang Island only.",
    )
    st.caption(f"Scope: {SCOPES[scope]['note']}")
    df = load_towers(scope)
    st.divider()

    st.markdown("### Filters")
    radios_present = [r for r in RADIO_META if r in set(df["radio"])]
    radio_labels = {RADIO_META[r][0]: r for r in radios_present}
    picked_labels = st.multiselect(
        "Network generation",
        options=list(radio_labels.keys()),
        default=[RADIO_META[r][0] for r in radios_present if r in ("NR", "LTE")],
    )
    picked = [radio_labels[l] for l in picked_labels] or radios_present

    rng_max = int(df["range"].max())
    max_range = st.slider(
        "Max coverage range (m)", 100, rng_max, rng_max, step=100,
        help="Reported cell range from OpenCelliD (coverage footprint proxy). "
             "Defaults to include all towers; lower it to exclude cells with "
             "implausibly large reported range.",
    )

    load_p = st.slider(
        "Overloaded threshold (load percentile)", 50, 99, 90, step=1,
        help="Cells above this 'samples' percentile are flagged as overloaded.",
    )

    show_coverage = st.checkbox("Show coverage footprints", value=False,
                                help="Draw each cell's range as a circle (slower).")
    show_gaps = st.checkbox("Show uncovered villages", value=True,
                            help="Mark OSM places outside all coverage footprints.")

    st.markdown("### New-site recommendations")
    new_range = st.slider("New 5G tower range (m)", 300, 3000, 1000, step=100,
                          help="Assumed coverage radius of a new 5G tower.")
    n_sites = st.slider("Sites to recommend", 1, 15, 5)
    show_sites = st.checkbox("Show recommended sites", value=True)

    st.divider()
    st.caption(
        "Data: OpenCelliD Malaysia (MCC 502). "
        f"Map: {'Leafmap' if HAVE_LEAFMAP else 'Folium (leafmap not installed)'}"
    )

# --------------------------------------------------------------------------
# Apply filters + derive metrics
# --------------------------------------------------------------------------
fdf = df[df["radio"].isin(picked) & (df["range"] <= max_range)].copy()

load_threshold = df["samples"].quantile(load_p / 100.0)
fdf["overloaded"] = fdf["samples"] >= load_threshold
overloaded = fdf[fdf["overloaded"]]

n_towers = len(fdf)
n_5g = int((fdf["radio"] == "NR").sum())
n_4g = int((fdf["radio"] == "LTE").sum())
n_overloaded = len(overloaded)

# --- Function 1: coverage gap (people + villages out of coverage) ---------
gap = compute_gap(scope, tuple(sorted(picked)), int(max_range))
places = load_places(scope)

# --- Function 3: recommended new-5G-tower sites ---------------------------
sites = compute_sites(scope, tuple(sorted(picked)), int(max_range), int(load_p),
                      int(new_range), int(n_sites))

# --------------------------------------------------------------------------
# Header + KPI row
# --------------------------------------------------------------------------
st.markdown(
    f'<div class="app-title">SiteSense 5G — Coverage &amp; Site-Selection Overview</div>'
    f'<div class="app-sub">{scope} pilot ({SCOPES[scope]["note"]}) · fuse towers + '
    f'signal + population + terrain → where the next 5G tower goes</div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
kpi_card(k1, "Cells in view", f"{n_towers:,}", f"{n_4g:,} × 4G · {n_5g:,} × 5G")
kpi_card(k2, "People out of coverage", f"{gap['uncovered_pop']:,.0f}",
         f"{gap['pct_covered']:.1f}% of {gap['total_pop']:,.0f} covered")
kpi_card(k3, "Kampungs uncovered", f"{gap['n_uncovered_places']:,}",
         f"of {gap['n_places']:,} OSM places in view")
kpi_card(k4, "Overloaded towers", f"{n_overloaded:,}",
         f"≥ {load_p}th pct load ({int(load_threshold)} samples)")

st.write("")

# --------------------------------------------------------------------------
# Main: map (left) + status panel (right)
# --------------------------------------------------------------------------
map_col, panel_col = st.columns([3, 1], gap="medium")

with map_col:
    _center = SCOPES[scope]["center"]
    _zoom = SCOPES[scope]["zoom"]
    if HAVE_LEAFMAP:
        m = leafmap.Map(center=_center, zoom=_zoom,
                        draw_control=False, measure_control=False,
                        fullscreen_control=True)
        m.add_basemap("CartoDB.Positron")
    else:
        m = folium.Map(location=_center, zoom_start=_zoom, tiles="cartodbpositron")

    import folium as _folium  # available via either path
    for radio in ["GSM", "UMTS", "LTE", "NR"]:  # draw important ones last (on top)
        sub = fdf[fdf["radio"] == radio]
        if sub.empty:
            continue
        label, color = RADIO_META[radio]
        for _, r in sub.iterrows():
            is_hot = bool(r.get("overloaded", False))
            if show_coverage and pd.notna(r["range"]):
                _folium.Circle(
                    location=[r["lat"], r["lon"]], radius=float(r["range"]),
                    color=color, weight=1, fill=True, fill_opacity=0.05,
                ).add_to(m)
            _folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=5 if is_hot else 3,
                color="#ff1744" if is_hot else color,
                weight=2 if is_hot else 1,
                fill=True, fill_opacity=0.9,
                popup=_folium.Popup(
                    f"<b>{label}</b>"
                    f"{' · <span style=color:#ff1744>OVERLOADED</span>' if is_hot else ''}"
                    f"<br>range: {r['range']:.0f} m"
                    f"<br>load (samples): {r['samples']:.0f}"
                    f"<br>cell: {r['cell']}",
                    max_width=240,
                ),
            ).add_to(m)

    # Function 1 — mark the villages/places that fall outside all coverage
    if show_gaps:
        uncov = ~gap["villages_covered_mask"]
        for lon, lat, nm, pl in zip(
            places["lon"][uncov], places["lat"][uncov],
            np.asarray(places["name"])[uncov], np.asarray(places["place"])[uncov],
        ):
            _folium.CircleMarker(
                location=[lat, lon], radius=6,
                color="#ffffff", weight=2,
                fill=True, fill_color="#ff9800", fill_opacity=0.95,
                popup=_folium.Popup(
                    f"<b>{nm or '(unnamed)'}</b><br>{pl}<br>"
                    f"<span style='color:#e65100'>OUT OF 4G/5G COVERAGE</span>",
                    max_width=220,
                ),
            ).add_to(m)

    # Function 3 — recommended new-5G-tower sites (numbered, green)
    if show_sites:
        for s in sites:
            _folium.Circle(
                location=[s["lat"], s["lon"]], radius=new_range,
                color="#00e676", weight=1, fill=True, fill_opacity=0.10,
            ).add_to(m)
            _folium.Marker(
                location=[s["lat"], s["lon"]],
                icon=_folium.DivIcon(html=(
                    f'<div style="background:#00c853;color:#04160b;font-weight:800;'
                    f'width:24px;height:24px;line-height:24px;text-align:center;'
                    f'border-radius:50%;border:2px solid #fff;'
                    f'box-shadow:0 0 4px rgba(0,0,0,.5)">{s["rank"]}</div>'
                )),
                popup=_folium.Popup(
                    f"<b>Recommended 5G site #{s['rank']}</b><br>"
                    f"+{s['people_gained']:,.0f} people covered<br>"
                    f"relieves {s['overloaded_relieved']} overloaded tower(s)<br>"
                    f"<span style='color:#666'>{s['lat']:.4f}, {s['lon']:.4f}</span>",
                    max_width=250,
                ),
            ).add_to(m)

    st_folium(m, use_container_width=True, height=560, returned_objects=[])

with panel_col:
    # --- Status ---
    st.markdown(
        f'<div class="panel"><h4>Status</h4>'
        f'<div style="color:#8b96a5;font-size:.8rem">Cells in view</div>'
        f'<div style="color:#f1f4f8;font-size:1.3rem;font-weight:700">{n_towers:,}</div>'
        f'<div style="margin-top:6px">'
        f'<span style="color:#1e88e5">● 4G {n_4g:,}</span>&nbsp;&nbsp;'
        f'<span style="color:#e53935">● 5G {n_5g:,}</span>&nbsp;&nbsp;'
        f'<span style="color:#ff1744">● hot {n_overloaded:,}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # --- Flagged / overloaded towers (Function 2) ---
    st.markdown('<div class="panel"><h4>⚠️ Flagged towers</h4>', unsafe_allow_html=True)
    if n_overloaded:
        top = overloaded.nlargest(8, "samples")[["label", "samples", "range", "cell"]]
        top = top.rename(columns={"label": "type", "samples": "load", "range": "range_m"})
        st.dataframe(top, hide_index=True, width="stretch", height=230)
    else:
        st.caption("No towers above the load threshold in the current filter.")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Coverage gap (Function 1) ---
    st.markdown(
        f'<div class="panel"><h4>📶 Coverage gap (4G/5G)</h4>'
        f'<div style="color:#f1f4f8;font-size:1.3rem;font-weight:700">'
        f'{gap["uncovered_pop"]:,.0f}</div>'
        f'<div style="color:#8b96a5;font-size:.78rem">people out of coverage · '
        f'{100 - gap["pct_covered"]:.1f}% of {scope}</div>'
        f'<div style="margin-top:8px;color:#ff9800">● {gap["n_uncovered_places"]} '
        f'villages/kampungs uncovered</div></div>',
        unsafe_allow_html=True,
    )
    if gap["n_uncovered_places"]:
        uncov = ~gap["villages_covered_mask"]
        names = [n for n in np.asarray(places["name"])[uncov] if n]
        with st.expander(f"List {gap['n_uncovered_places']} uncovered places"):
            st.write(", ".join(sorted(names)) or "(unnamed places only)")

    # --- AI insight (Function 3: recommended sites) ---
    if sites:
        covered_total = sites[-1]["cumulative_gained"]
        pct_gap = covered_total / gap["uncovered_pop"] * 100 if gap["uncovered_pop"] else 0
        rows = "".join(
            f'<div style="margin:6px 0;padding-bottom:6px;'
            f'border-bottom:1px solid #1c3b32">'
            f'<b style="color:#00e676">#{s["rank"]}</b> '
            f'+{s["people_gained"]:,.0f} people'
            f'{f" · relieves {s['overloaded_relieved']} overloaded" if s["overloaded_relieved"] else ""}'
            f'<br><span style="color:#7f8a99;font-size:.75rem">'
            f'{s["lat"]:.4f}, {s["lon"]:.4f}</span></div>'
            for s in sites
        )
        st.markdown(
            f'<div class="panel"><h4>💡 AI Insights — recommended 5G sites</h4>'
            f'<div class="ai-insight">Building these <b>{len(sites)}</b> towers '
            f'(range {new_range} m) would cover <b>{covered_total:,.0f}</b> of the '
            f'{gap["uncovered_pop"]:,.0f} uncovered people (<b>{pct_gap:.0f}%</b> of the gap).'
            f'<div style="margin-top:8px">{rows}</div></div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="panel"><h4>💡 AI Insights</h4>'
            '<div class="ai-insight">No coverage gap to close in the current '
            'filter — everyone is covered.</div></div>',
            unsafe_allow_html=True,
        )

st.caption(
    "Function 1 (uncovered population/kampungs) ✓ · Function 2 (overloaded flags) ✓ "
    "· Function 3 (greedy max-coverage new-5G-site recommendations) ✓ — all live. "
    "MVP complete."
)
