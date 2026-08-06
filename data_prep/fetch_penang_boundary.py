"""
Fetch the Penang ISLAND administrative boundary polygon from OSM Nominatim.
No account needed. Saves data/boundaries/penang_island.geojson.

We want the island only (lon ~100.17-100.36), NOT the whole state (which
includes mainland Seberang Perai, lon up to ~100.55).

Run:  py data_prep/fetch_penang_boundary.py
"""
from pathlib import Path
import json
import time
import urllib.parse
import urllib.request

APP_DIR = Path(__file__).resolve().parent.parent
OUT = APP_DIR / "data" / "boundaries" / "penang_island.geojson"

QUERIES = [
    "Penang Island, Malaysia",
    "Timur Laut, Penang, Malaysia",     # NE island district
    "Barat Daya, Penang, Malaysia",     # SW island district
]


def nominatim(q: str):
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({
        "q": q, "format": "jsonv2", "polygon_geojson": 1, "limit": 5,
    })
    req = urllib.request.Request(url, headers={
        "User-Agent": "SiteSense5G/1.0 (hackathon; contact team KH-002)"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for q in QUERIES:
        print(f"\n=== query: {q!r} ===")
        try:
            results = nominatim(q)
        except Exception as e:
            print("  failed:", e); time.sleep(1); continue
        for r in results:
            bb = r.get("boundingbox", [])
            geo = r.get("geojson", {})
            print(f"  - {r.get('display_name','')[:60]!r}")
            print(f"    class={r.get('class')} type={r.get('type')} "
                  f"geom={geo.get('type')} bbox(lat {bb[0]}..{bb[1]}, lon {bb[2]}..{bb[3]})"
                  if len(bb) == 4 else "    (no bbox)")
        # pick first polygon whose lon stays within the island (max lon < 100.42)
        for r in results:
            geo = r.get("geojson", {})
            bb = r.get("boundingbox", [])
            if geo.get("type") in ("Polygon", "MultiPolygon") and len(bb) == 4:
                lon_max = float(bb[3])
                if lon_max < 100.42:
                    fc = {"type": "FeatureCollection", "features": [
                        {"type": "Feature",
                         "properties": {"name": r.get("display_name", ""),
                                        "source_query": q},
                         "geometry": geo}]}
                    OUT.write_text(json.dumps(fc), encoding="utf-8")
                    print(f"\n  SAVED island polygon ({geo['type']}) from {q!r} -> {OUT.name}")
                    return
        time.sleep(1)  # be polite to Nominatim
    print("\nNo island polygon found within expected lon range — inspect output above.")


if __name__ == "__main__":
    main()
