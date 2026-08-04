"""
Function 1 - coverage-gap computation for SiteSense 5G.

Given a set of towers (lon, lat, coverage-range in metres), determine which
WorldPop population pixels and which OSM place points fall OUTSIDE the union
of all tower coverage footprints -> "people out of coverage" and
"kampungs / villages uncovered".

Distances use a local equirectangular metre-per-degree scaling calibrated to
Penang's latitude (no pyproj, per project constraint).
"""
from pathlib import Path
import json
import numpy as np
import rasterio

M_PER_DEG_LAT = 111_320.0


def m_per_deg_lon(lat_deg: float) -> float:
    return 111_320.0 * np.cos(np.radians(lat_deg))


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_population(tif_path: str | Path):
    """Return (pop_array float64 with NaN nodata, affine transform)."""
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype("float64")
        transform = src.transform
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < 0] = np.nan
    return arr, transform


def load_places(geojson_path: str | Path):
    """Return dict with lon, lat (np arrays), name, place (lists)."""
    d = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    lon, lat, name, place = [], [], [], []
    for f in d["features"]:
        c = f["geometry"]["coordinates"]
        lon.append(c[0]); lat.append(c[1])
        name.append(f["properties"].get("name", ""))
        place.append(f["properties"].get("place", ""))
    return {"lon": np.array(lon), "lat": np.array(lat),
            "name": name, "place": place}


# --------------------------------------------------------------------------
# Coverage tests
# --------------------------------------------------------------------------
def covered_points(plon, plat, tlon, tlat, trange, lat0: float) -> np.ndarray:
    """Boolean mask over points: True if within range of ANY tower."""
    mlon = m_per_deg_lon(lat0)
    tlon = np.asarray(tlon); tlat = np.asarray(tlat); trange = np.asarray(trange)
    ok = np.isfinite(trange) & (trange > 0)
    tlon, tlat, trange = tlon[ok], tlat[ok], trange[ok]
    covered = np.zeros(len(plon), dtype=bool)
    for i in range(len(plon)):
        dx = (tlon - plon[i]) * mlon
        dy = (tlat - plat[i]) * M_PER_DEG_LAT
        covered[i] = np.any((dx * dx + dy * dy) <= trange * trange)
    return covered


def coverage_mask_raster(shape, transform, tlon, tlat, trange, lat0: float) -> np.ndarray:
    """Boolean raster (H×W): True where a pixel is inside any tower footprint.

    Each tower only stamps its own bounding window, so cost scales with total
    covered area, not towers × pixels.
    """
    H, W = shape
    covered = np.zeros((H, W), dtype=bool)
    inv = ~transform
    px = transform.a           # deg per pixel (x, +ve)
    py = -transform.e          # deg per pixel (y, +ve)
    mlon = m_per_deg_lon(lat0)

    for lon, lat, rng in zip(tlon, tlat, trange):
        if not np.isfinite(rng) or rng <= 0:
            continue
        rpx = rng / (px * mlon)          # pixel radius in x
        rpy = rng / (py * M_PER_DEG_LAT)  # pixel radius in y
        col_c, row_c = inv * (lon, lat)
        c0 = max(0, int(np.floor(col_c - rpx))); c1 = min(W, int(np.ceil(col_c + rpx)) + 1)
        r0 = max(0, int(np.floor(row_c - rpy))); r1 = min(H, int(np.ceil(row_c + rpy)) + 1)
        if c0 >= c1 or r0 >= r1:
            continue
        cols = np.arange(c0, c1); rows = np.arange(r0, r1)
        cc, rr = np.meshgrid(cols, rows)
        dx = (cc - col_c) * px * mlon
        dy = (rr - row_c) * py * M_PER_DEG_LAT
        covered[r0:r1, c0:c1] |= (dx * dx + dy * dy) <= (rng * rng)
    return covered


def coverage_gap(pop, transform, places, tlon, tlat, trange, lat0: float) -> dict:
    """Compute the full Function-1 result bundle."""
    cov = coverage_mask_raster(pop.shape, transform, tlon, tlat, trange, lat0)
    valid = np.isfinite(pop)
    total_pop = float(np.nansum(pop))
    covered_pop = float(np.nansum(pop[cov & valid]))
    uncovered_pop = total_pop - covered_pop

    vcov = covered_points(places["lon"], places["lat"], tlon, tlat, trange, lat0)
    n_places = len(places["lon"])
    n_uncov_places = int((~vcov).sum())

    return {
        "coverage_mask": cov,
        "total_pop": total_pop,
        "covered_pop": covered_pop,
        "uncovered_pop": uncovered_pop,
        "pct_covered": (covered_pop / total_pop * 100.0) if total_pop else 0.0,
        "villages_covered_mask": vcov,
        "n_places": n_places,
        "n_uncovered_places": n_uncov_places,
    }
