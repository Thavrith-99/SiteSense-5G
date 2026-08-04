"""
Clip the national WorldPop Malaysia raster to the Penang Island bbox.

Input : data/population/mys_ppp_2020_UNadj.tif  (157 MB, whole country)
Output: data/population/penang_ppp_2020.tif      (small, GitHub-safe)

Run:  py data_prep/clip_worldpop.py
"""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds

APP_DIR = Path(__file__).resolve().parent.parent
SRC = APP_DIR / "data" / "population" / "mys_ppp_2020_UNadj.tif"
OUT = APP_DIR / "data" / "population" / "penang_ppp_2020.tif"

# Penang Island bbox (lon_min, lat_min, lon_max, lat_max)
BBOX = (100.1, 5.1, 100.6, 5.6)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing WorldPop raster: {SRC}")

    with rasterio.open(SRC) as src:
        print(f"Source: {src.width}x{src.height} px, CRS {src.crs}, "
              f"res {src.res[0]:.5f} deg (~100 m)")
        win = from_bounds(*BBOX, transform=src.transform)
        data = src.read(1, window=win)
        transform = src.window_transform(win)

        nodata = src.nodata if src.nodata is not None else -99999.0
        valid = data[(data != nodata) & (data > 0)]
        total_pop = float(valid.sum())

        profile = src.profile.copy()
        profile.update(
            height=data.shape[0], width=data.shape[1], transform=transform,
            compress="deflate",
        )
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(OUT, "w", **profile) as dst:
            dst.write(data, 1)

    size_mb = OUT.stat().st_size / 1e6
    print(f"Clipped: {data.shape[1]}x{data.shape[0]} px -> {OUT.name} ({size_mb:.2f} MB)")
    print(f"Penang population (WorldPop 2020, UN-adj): {total_pop:,.0f}")


if __name__ == "__main__":
    main()
