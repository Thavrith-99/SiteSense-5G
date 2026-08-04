"""
Fetch OSM place points (villages / kampungs / towns) inside the Penang bbox
via the Overpass API. No API key required.

Output: data/villages/penang_places.geojson

Run:  py data_prep/fetch_osm_villages.py
"""
from pathlib import Path
import json
import time
import urllib.request

APP_DIR = Path(__file__).resolve().parent.parent
OUT = APP_DIR / "data" / "villages" / "penang_places.geojson"

# Penang bbox as Overpass (south, west, north, east)
S, W, N, E = 5.1, 100.1, 5.6, 100.6

QUERY = f"""
[out:json][timeout:90];
(
  node["place"~"^(city|town|village|hamlet|suburb|neighbourhood)$"]({S},{W},{N},{E});
);
out body;
"""

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def fetch() -> dict:
    last_err = None
    for url in ENDPOINTS:
        try:
            print(f"Querying {url} ...")
            data = urllib.request.urlopen(
                urllib.request.Request(
                    url, data=("data=" + QUERY).encode("utf-8"),
                    headers={"User-Agent": "SiteSense5G/1.0 (hackathon)"},
                ),
                timeout=100,
            ).read()
            return json.loads(data)
        except Exception as e:  # try next mirror
            print(f"  failed: {e}")
            last_err = e
            time.sleep(2)
    raise SystemExit(f"All Overpass endpoints failed: {last_err}")


def to_geojson(osm: dict) -> dict:
    feats = []
    for el in osm.get("elements", []):
        if el.get("type") != "node":
            continue
        tags = el.get("tags", {})
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [el["lon"], el["lat"]]},
            "properties": {
                "name": tags.get("name", ""),
                "place": tags.get("place", ""),
                "osm_id": el.get("id"),
            },
        })
    return {"type": "FeatureCollection", "features": feats}


def main() -> None:
    gj = to_geojson(fetch())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gj), encoding="utf-8")

    n = len(gj["features"])
    by_place = {}
    for f in gj["features"]:
        by_place[f["properties"]["place"]] = by_place.get(f["properties"]["place"], 0) + 1
    print(f"Saved {n} place points -> {OUT.name}")
    print("By type:", dict(sorted(by_place.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
