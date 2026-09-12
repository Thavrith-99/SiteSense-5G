"""
SiteSense 5G — shared data layer (towers, population, villages, coverage gap,
site recommendations). Framework-agnostic: no Streamlit imports here, so both
app.py (Streamlit, wraps these in @st.cache_data) and api/main.py (FastAPI)
call the exact same logic instead of maintaining two copies.

Each vector loader tries PostGIS (db.py) first and falls back to the
CSV/GeoJSON files if DATABASE_URL isn't configured or the DB is unreachable.
Population always stays file-based (WorldPop GeoTIFF) — see
db/schema.sql's header comment for why.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import coverage as cov
import recommend as rec

APP_DIR = Path(__file__).resolve().parent
DATA = APP_DIR / "data"
CSV = DATA / "towers_penang" / "502.csv"

SCOPES = {
    "Penang State": {
        "boundary": DATA / "boundaries" / "penang_state.geojson",
        "pop": DATA / "population" / "penang_state_ppp_2020.tif",
        "places": DATA / "villages" / "penang_places_state.geojson",
        "slope": DATA / "terrain" / "penang_state_slope.tif",
        "center": [5.30, 100.40], "zoom": 11,
        "note": "island + mainland Seberang Perai",
    },
    "Penang Island": {
        "boundary": DATA / "boundaries" / "penang_island.geojson",
        "pop": DATA / "population" / "penang_island_ppp_2020.tif",
        "places": DATA / "villages" / "penang_places_island.geojson",
        "slope": DATA / "terrain" / "penang_island_slope.tif",
        "center": [5.37, 100.27], "zoom": 12,
        "note": "island only",
    },
}

# Maps the display scope names above to db/schema.sql's `boundaries.scope`.
DB_SCOPE = {"Penang State": "penang_state", "Penang Island": "penang_island"}

# Penang bounding box (coarse pre-filter before the polygon clip, file path only)
LAT_MIN, LAT_MAX = 5.1, 5.6
LON_MIN, LON_MAX = 100.1, 100.6

RADIO_META = {
    "NR":   ("5G NR",   "#e53935"),
    "LTE":  ("4G LTE",  "#1e88e5"),
    "UMTS": ("3G UMTS", "#9c27b0"),
    "GSM":  ("2G GSM",  "#9e9e9e"),
}


def using_db() -> bool:
    """True if PostGIS is reachable right now."""
    try:
        from db import get_connection
        get_connection().close()
        return True
    except Exception:
        return False


def _load_towers_db(scope: str):
    try:
        from db import get_connection
        conn = get_connection()
    except Exception:
        return None
    try:
        df = pd.read_sql(
            """
            SELECT t.radio, t.range_m AS range, t.samples, t.cell,
                   ST_X(t.geom) AS lon, ST_Y(t.geom) AS lat
            FROM towers t
            JOIN boundaries b ON ST_Within(t.geom, b.geom)
            WHERE b.scope = %(scope)s
            """,
            conn, params={"scope": DB_SCOPE[scope]},
        )
    except Exception:
        return None
    finally:
        conn.close()
    df["label"] = df["radio"].map(lambda r: RADIO_META.get(r, (r, ""))[0])
    return df


def _load_places_db(scope: str):
    try:
        from db import get_connection
        conn = get_connection()
    except Exception:
        return None
    try:
        df = pd.read_sql(
            """
            SELECT v.name, v.place_type AS place,
                   ST_X(v.geom) AS lon, ST_Y(v.geom) AS lat
            FROM villages v
            JOIN boundaries b ON ST_Within(v.geom, b.geom)
            WHERE b.scope = %(scope)s
            """,
            conn, params={"scope": DB_SCOPE[scope]},
        )
    except Exception:
        return None
    finally:
        conn.close()
    return {"lon": df["lon"].values, "lat": df["lat"].values,
            "name": df["name"].fillna("").tolist(),
            "place": df["place"].fillna("").tolist()}


@lru_cache(maxsize=8)
def load_towers(scope: str) -> pd.DataFrame:
    """OpenCelliD Malaysia cells, clipped to the chosen scope polygon.
    Tries PostGIS first; falls back to the CSV + shapely clip.
    Cached: callers only read / take filtered copies, never mutate the result."""
    db_df = _load_towers_db(scope)
    if db_df is not None:
        return db_df

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


@lru_cache(maxsize=8)
def load_population(scope: str):
    return cov.load_population(SCOPES[scope]["pop"])


@lru_cache(maxsize=8)
def load_places(scope: str):
    """Tries PostGIS first; falls back to the scope's villages GeoJSON."""
    db_places = _load_places_db(scope)
    if db_places is not None:
        return db_places
    return cov.load_places(SCOPES[scope]["places"])


@lru_cache(maxsize=8)
def load_slope(scope: str):
    """Real SRTM-derived slope (degrees), Roadmap #6 — see
    data_prep/fetch_srtm_slope.py. Returns (None, None) if not yet fetched,
    so recommend.py's slope weight is automatically forced to 0."""
    path = SCOPES[scope]["slope"]
    if not path.exists():
        return None, None
    return cov.load_population(path)  # generic single-band float raster loader


@lru_cache(maxsize=16)
def compute_gap(scope: str, picked_key: tuple, max_range: int):
    """Function 1 — coverage gap for the currently-filtered towers.
    Cached (the ~9s hot path): the dashboard's standalone call and the internal
    call inside compute_sites share one result; changing filters that don't
    affect the gap (profile, threshold, new-tower range, #sites) then reruns
    only the fast recommend step."""
    df = load_towers(scope)
    sub = df[df["radio"].isin(picked_key) & (df["range"] <= max_range)]
    pop, transform = load_population(scope)
    places = load_places(scope)
    return cov.coverage_gap(
        pop, transform, places,
        sub["lon"].values, sub["lat"].values, sub["range"].values,
        SCOPES[scope]["center"][0],
    )


def compute_sites(scope: str, picked_key: tuple, max_range: int, load_p: int,
                  new_range: int, n_sites: int, profile: str = "Coverage-first"):
    """Function 3 — weighted multi-criteria (MCDA) new-5G-tower
    recommendations. `profile` selects a preset weighting from
    recommend.WEIGHT_PROFILES ("Coverage-first" reproduces the original
    population-only greedy result exactly)."""
    df = load_towers(scope)
    sub = df[df["radio"].isin(picked_key) & (df["range"] <= max_range)]
    pop, transform = load_population(scope)
    g = compute_gap(scope, picked_key, max_range)
    thr = sub["samples"].quantile(load_p / 100.0)
    ov = sub[sub["samples"] >= thr]
    slope, _ = load_slope(scope)

    # Underserved villages (outside every tower's estimated footprint) so each
    # recommended site can report how many it would newly bring into range (#4).
    places = load_places(scope)
    uncov = ~g["villages_covered_mask"]
    uv_lon = np.asarray(places["lon"])[uncov]
    uv_lat = np.asarray(places["lat"])[uncov]
    uv_names = np.asarray(places["name"])[uncov]

    return rec.recommend_sites(
        pop, g["coverage_mask"], transform, SCOPES[scope]["center"][0],
        new_range, n_sites, ov["lon"].values, ov["lat"].values,
        tower_lon=sub["lon"].values, tower_lat=sub["lat"].values,
        slope=slope, profile=profile,
        uv_lon=uv_lon, uv_lat=uv_lat, uv_names=uv_names,
    )
