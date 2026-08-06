"""
Clip population + villages to the full Penang STATE polygon (island + mainland
Seberang Perai). Mirror of clip_to_island.py for the wider scope.

Inputs : data/boundaries/penang_state.geojson
         data/population/penang_ppp_2020.tif
         data/villages/penang_places.geojson
Outputs: data/population/penang_state_ppp_2020.tif
         data/villages/penang_places_state.geojson

Run:  py data_prep/clip_to_state.py
"""
from pathlib import Path
import json
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, Point

APP = Path(__file__).resolve().parent.parent
BOUNDARY = APP / "data" / "boundaries" / "penang_state.geojson"
POP_IN = APP / "data" / "population" / "penang_ppp_2020.tif"
POP_OUT = APP / "data" / "population" / "penang_state_ppp_2020.tif"
VIL_IN = APP / "data" / "villages" / "penang_places.geojson"
VIL_OUT = APP / "data" / "villages" / "penang_places_state.geojson"


def main():
    fc = json.loads(BOUNDARY.read_text(encoding="utf-8"))
    geom = fc["features"][0]["geometry"]
    poly = shape(geom)

    with rasterio.open(POP_IN) as src:
        nodata = src.nodata if src.nodata is not None else -99999.0
        out_img, out_tr = mask(src, [geom], crop=True, nodata=nodata, filled=True)
        prof = src.profile.copy()
    arr = out_img[0]
    valid = arr[(arr != nodata) & (arr > 0)]
    prof.update(height=arr.shape[0], width=arr.shape[1], transform=out_tr,
                nodata=nodata, compress="deflate")
    with rasterio.open(POP_OUT, "w", **prof) as dst:
        dst.write(arr, 1)

    vfc = json.loads(VIL_IN.read_text(encoding="utf-8"))
    kept = [f for f in vfc["features"]
            if poly.contains(Point(f["geometry"]["coordinates"]))]
    VIL_OUT.write_text(json.dumps({"type": "FeatureCollection", "features": kept}),
                       encoding="utf-8")

    print(f"State raster: {arr.shape[1]}x{arr.shape[0]} px -> {POP_OUT.name}")
    print(f"Penang STATE population: {float(valid.sum()):,.0f}")
    print(f"Villages: {len(vfc['features'])} bbox -> {len(kept)} in state -> {VIL_OUT.name}")


if __name__ == "__main__":
    main()
