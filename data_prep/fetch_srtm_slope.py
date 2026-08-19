"""
Fetch real SRTM elevation for Penang and compute a slope-penalty raster for
Function 3's MCDA scoring (Roadmap #6 — terrain/slope).

Source: AWS "Terrain Tiles" open dataset (Mapzen/Tilezen), Skadi/SRTM-HGT
format, public S3, no API key/account needed:
  https://registry.opendata.aws/terrain-tiles/
Penang's whole bbox (5.1-5.6N, 100.1-100.6E) fits inside ONE 1-degree tile,
N05E100 — verified reachable before use (`curl -I` returned 200 OK).

No Google Earth Engine account needed, per the Technical Upgrades sheet's
own note ("SRTM is downloadable directly, e.g. via OpenTopography") — this
uses an even simpler no-auth public mirror of the same SRTM1 (30m) data.

Slope is computed on the native ~30m SRTM grid, then resampled onto EACH
scope's population raster grid (same transform/shape as
data/population/penang_{state,island}_ppp_2020.tif) so recommend.py can
index it with the exact same (row, col) as pop/cov_mask — no separate
lookup system needed.

Honest caveat (state this in the pitch): slope is a simple proxy for
buildability/accessibility, not a full RF propagation/viewshed model.

Run (from SiteSense5G_App/):
    py data_prep/fetch_srtm_slope.py
Outputs:
    data/terrain/penang_state_slope.tif
    data/terrain/penang_island_slope.tif
"""
import gzip
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

APP = Path(__file__).resolve().parent.parent
TERRAIN_DIR = APP / "data" / "terrain"
TERRAIN_DIR.mkdir(parents=True, exist_ok=True)

SRTM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/N05/N05E100.hgt.gz"
HGT_GZ = TERRAIN_DIR / "N05E100.hgt.gz"
HGT = TERRAIN_DIR / "N05E100.hgt"

M_PER_DEG_LAT = 111_320.0

TARGETS = {
    "penang_state": APP / "data" / "population" / "penang_state_ppp_2020.tif",
    "penang_island": APP / "data" / "population" / "penang_island_ppp_2020.tif",
}


def fetch_hgt():
    if not HGT.exists():
        print(f"Downloading {SRTM_URL} ...")
        urllib.request.urlretrieve(SRTM_URL, HGT_GZ)
        with gzip.open(HGT_GZ, "rb") as fin, open(HGT, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        HGT_GZ.unlink()
        print(f"Extracted {HGT} ({HGT.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"{HGT} already present, skipping download")


def compute_slope_degrees(elev: np.ndarray, transform) -> np.ndarray:
    """Slope in degrees from an elevation raster, using local metre-scaling
    (consistent with the project's no-pyproj convention elsewhere)."""
    px = transform.a
    py = -transform.e
    lat0 = transform.f - (elev.shape[0] / 2) * py  # approx centre latitude
    mlon = M_PER_DEG_LAT * np.cos(np.radians(lat0))
    dz_dy, dz_dx = np.gradient(elev.astype("float64"), py * M_PER_DEG_LAT, px * mlon)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    return np.degrees(slope_rad)


def main():
    fetch_hgt()

    with rasterio.open(HGT) as src:
        elev = src.read(1).astype("float64")
        elev[elev <= -32768] = np.nan  # SRTM void/nodata sentinel
        slope = compute_slope_degrees(np.nan_to_num(elev, nan=0.0), src.transform)
        slope[np.isnan(elev)] = np.nan
        src_transform = src.transform
        src_crs = src.crs

    print(f"SRTM tile: {elev.shape[1]}x{elev.shape[0]} px, "
          f"elevation {np.nanmin(elev):.0f}-{np.nanmax(elev):.0f} m, "
          f"slope {np.nanmin(slope):.1f}-{np.nanmax(slope):.1f} deg")

    for scope, pop_path in TARGETS.items():
        with rasterio.open(pop_path) as dst_ref:
            dst_profile = dst_ref.profile.copy()
            dst_transform = dst_ref.transform
            dst_shape = (dst_ref.height, dst_ref.width)
            dst_crs = dst_ref.crs

        out = np.full(dst_shape, np.nan, dtype="float32")
        reproject(
            source=slope.astype("float32"), destination=out,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan,
        )

        out_path = TERRAIN_DIR / f"{scope}_slope.tif"
        dst_profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
        with rasterio.open(out_path, "w", **dst_profile) as dst:
            dst.write(out, 1)
        print(f"{scope}: slope resampled to {dst_shape[1]}x{dst_shape[0]} px -> {out_path.name} "
              f"(mean {np.nanmean(out):.1f} deg)")


if __name__ == "__main__":
    main()
