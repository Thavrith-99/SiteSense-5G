"""
Clip population raster + village points to the Penang ISLAND polygon, so the
dashboard's headline numbers reflect the island pilot (not the mainland spill).

Inputs : data/boundaries/penang_island.geojson
         data/population/penang_ppp_2020.tif
         data/villages/penang_places.geojson
Outputs: data/population/penang_island_ppp_2020.tif
         data/villages/penang_places_island.geojson

Run:  py data_prep/clip_to_island.py
"""
from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, Point

APP = Path(__file__).resolve().parent.parent
BOUNDARY = APP / "data" / "boundaries" / "penang_island.geojson"
POP_IN = APP / "data" / "population" / "penang_ppp_2020.tif"
POP_OUT = APP / "data" / "population" / "penang_island_ppp_2020.tif"
VIL_IN = APP / "data" / "villages" / "penang_places.geojson"
VIL_OUT = APP / "data" / "villages" / "penang_places_island.geojson"


def load_polygon():
    fc = json.loads(BOUNDARY.read_text(encoding="utf-8"))
    geom = fc["features"][0]["geometry"]
    return geom, shape(geom)


def clip_population(geom):
    with rasterio.open(POP_IN) as src:
        nodata = src.nodata if src.nodata is not None else -99999.0
        out_img, out_tr = mask(src, [geom], crop=True, nodata=nodata, filled=True)
        prof = src.profile.copy()
    arr = out_img[0]
    valid = arr[(arr != nodata) & (arr > 0)]
    total = float(valid.sum())
    prof.update(height=arr.shape[0], width=arr.shape[1], transform=out_tr,
                nodata=nodata, compress="deflate")
    with rasterio.open(POP_OUT, "w", **prof) as dst:
        dst.write(arr, 1)
    return total, arr.shape


def clip_villages(poly):
    fc = json.loads(VIL_IN.read_text(encoding="utf-8"))
    kept = [f for f in fc["features"]
            if poly.contains(Point(f["geometry"]["coordinates"]))]
    out = {"type": "FeatureCollection", "features": kept}
    VIL_OUT.write_text(json.dumps(out), encoding="utf-8")
    return len(fc["features"]), len(kept)


def main():
    geom, poly = load_polygon()
    pop, shp = clip_population(geom)
    n_before, n_after = clip_villages(poly)
    print(f"Island raster: {shp[1]}x{shp[0]} px -> {POP_OUT.name}")
    print(f"Penang ISLAND population (WorldPop 2020): {pop:,.0f}")
    print(f"Villages: {n_before} in bbox -> {n_after} on the island -> {VIL_OUT.name}")


if __name__ == "__main__":
    main()
