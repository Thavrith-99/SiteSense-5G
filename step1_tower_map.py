"""
SiteSense 5G - Step 1: See the network
Team Neural Shield (KH-002) - ASEAN GeoAI Fusion 2026

Loads OpenCelliD Malaysia towers (502.csv), filters to Penang Island,
and renders an interactive map of every tower with its coverage footprint.

Run:  py step1_tower_map.py
Output: outputs/step1_penang_towers.html  (open in a browser)
"""

from pathlib import Path
import pandas as pd
import folium
from folium.plugins import MarkerCluster

# --- Paths ---------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
CSV = ROOT / "Bootcamp 2 - Advanced GeoAI-Day2" / "Module 6_AD1002_GeoAI Dataset" / "GeoAI Dataset" / "502.csv"
OUT = APP_DIR / "outputs" / "step1_penang_towers.html"

# --- Penang Island bounding box (from project notes) ---------------------
LAT_MIN, LAT_MAX = 5.1, 5.6
LON_MIN, LON_MAX = 100.1, 100.6
PENANG_CENTER = [5.35, 100.30]

# Colour per radio generation
RADIO_COLORS = {
    "GSM": "#9e9e9e",    # 2G  - grey
    "UMTS": "#ffb300",   # 3G  - amber
    "LTE": "#1e88e5",    # 4G  - blue
    "NR": "#e53935",     # 5G  - red (the ones we care about)
}


def load_penang_towers() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    penang = df[
        (df["lat"].between(LAT_MIN, LAT_MAX))
        & (df["lon"].between(LON_MIN, LON_MAX))
    ].copy()
    return penang


def summarise(df: pd.DataFrame) -> None:
    print(f"Penang cells: {len(df):,}")
    print("By radio generation:")
    for radio, n in df["radio"].value_counts().items():
        print(f"  {radio:5s} {n:,}")
    print(f"Coverage range (m):  min={df['range'].min():.0f}  "
          f"median={df['range'].median():.0f}  max={df['range'].max():.0f}")


def build_map(df: pd.DataFrame) -> folium.Map:
    m = folium.Map(location=PENANG_CENTER, zoom_start=12, tiles="cartodbpositron")

    # One toggleable layer per radio generation
    for radio in ["NR", "LTE", "UMTS", "GSM"]:
        sub = df[df["radio"] == radio]
        if sub.empty:
            continue
        color = RADIO_COLORS.get(radio, "#000000")
        label = {"NR": "5G NR", "LTE": "4G LTE", "UMTS": "3G UMTS", "GSM": "2G GSM"}[radio]
        fg = folium.FeatureGroup(name=f"{label} ({len(sub):,})", show=(radio in ("NR", "LTE")))

        cluster = MarkerCluster().add_to(fg)
        for _, r in sub.iterrows():
            # coverage footprint from reported range (metres)
            folium.Circle(
                location=[r["lat"], r["lon"]],
                radius=float(r["range"]) if pd.notna(r["range"]) else 0,
                color=color, weight=1, fill=True, fill_opacity=0.06,
            ).add_to(fg)
            # tower point (clustered so the map stays readable)
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=3, color=color, fill=True, fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>{label}</b><br>range: {r['range']:.0f} m"
                    f"<br>samples (load proxy): {r['samples']:.0f}"
                    f"<br>cell: {r['cell']}",
                    max_width=220,
                ),
            ).add_to(cluster)
        fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend / title box
    title = f"""
    <div style="position: fixed; top: 12px; left: 60px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 8px;
                box-shadow: 0 2px 6px rgba(0,0,0,.3); font-family: sans-serif;">
      <b>SiteSense 5G &mdash; Step 1: See the Network</b><br>
      <span style="font-size:12px">Penang Island &middot; {len(df):,} cells from OpenCelliD (502.csv)</span>
    </div>"""
    m.get_root().html.add_child(folium.Element(title))
    return m


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"Cannot find 502.csv at:\n  {CSV}")
    df = load_penang_towers()
    summarise(df)
    m = build_map(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT))
    print(f"\nSaved map -> {OUT}")


if __name__ == "__main__":
    main()
