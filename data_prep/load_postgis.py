"""
Load the flat-file vector inputs (502.csv, OSM villages, boundaries) into
PostGIS, per db/schema.sql. Population stays file-based — see schema.sql's
header comment for why.

Mirrors app.py's load_towers(): same LAT/LON bbox pre-filter + state-polygon
clip, done once here instead of on every Streamlit rerun. Villages/towers are
stored ONCE against the wider "Penang State" scope; the "Penang Island"
subset is derived at query time via ST_Within against the boundaries table
(see the example query in the module docstring below), so island-clipped
files are no longer needed as separate inputs.

Inputs : data/towers_penang/502.csv
         data/boundaries/penang_state.geojson
         data/boundaries/penang_island.geojson
         data/villages/penang_places_state.geojson
Outputs: rows in towers / villages / boundaries (PostGIS)

Run:
    export DATABASE_URL=postgresql://user:pass@localhost:5432/sitesense5g
    py data_prep/load_postgis.py

Example query the app / FastAPI layer will use afterwards (replaces the
CSV + shapely clip in app.py's load_towers):

    SELECT ST_X(t.geom), ST_Y(t.geom), t.radio, t.range_m, t.samples
    FROM towers t
    JOIN boundaries b ON ST_Within(t.geom, b.geom)
    WHERE b.scope = %(scope)s;   -- 'penang_state' | 'penang_island'
"""
import json
import sys
from pathlib import Path

import pandas as pd
import psycopg2.extras

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))  # db.py lives at the app root, not in data_prep/
from db import get_connection  # noqa: E402
CSV = APP / "data" / "towers_penang" / "502.csv"
BOUNDARIES = {
    "penang_state": (APP / "data" / "boundaries" / "penang_state.geojson", "Penang State"),
    "penang_island": (APP / "data" / "boundaries" / "penang_island.geojson", "Penang Island"),
}
VILLAGES = APP / "data" / "villages" / "penang_places_state.geojson"
SCHEMA_SQL = APP / "db" / "schema.sql"

LAT_MIN, LAT_MAX = 5.1, 5.6
LON_MIN, LON_MAX = 100.1, 100.6


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
    conn.commit()


def load_boundaries(conn):
    with conn.cursor() as cur:
        for scope, (path, label) in BOUNDARIES.items():
            geom = json.loads(path.read_text(encoding="utf-8"))["features"][0]["geometry"]
            cur.execute(
                """
                INSERT INTO boundaries (scope, label, geom)
                VALUES (%s, %s, ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)))
                ON CONFLICT (scope) DO UPDATE
                    SET label = EXCLUDED.label, geom = EXCLUDED.geom
                """,
                (scope, label, json.dumps(geom)),
            )
    conn.commit()
    print(f"boundaries: {len(BOUNDARIES)} polygons loaded")


def load_towers(conn):
    df = pd.read_csv(CSV)
    df = df[df["lat"].between(LAT_MIN, LAT_MAX) & df["lon"].between(LON_MIN, LON_MAX)].copy()

    with conn.cursor() as cur:
        cur.execute("TRUNCATE towers RESTART IDENTITY")
        # keep towers within the STATE polygon (the wider scope); island is
        # derived later via ST_Within against boundaries, not re-clipped here
        cur.execute(
            """
            CREATE TEMP TABLE _state_boundary AS
            SELECT geom FROM boundaries WHERE scope = 'penang_state'
            """
        )
        rows = list(df[["radio", "mcc", "net", "area", "cell", "range", "samples",
                        "changeable", "lon", "lat"]].itertuples(index=False, name=None))
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO towers (radio, mcc, net, area, cell, range_m, samples,
                                 changeable, geom)
            SELECT radio, mcc, net, area, cell, range_m, samples,
                   changeable::boolean, ST_SetSRID(ST_MakePoint(lon, lat), 4326)
            FROM (VALUES %s) AS v(radio, mcc, net, area, cell, range_m, samples,
                                   changeable, lon, lat)
            WHERE ST_Within(ST_SetSRID(ST_MakePoint(lon, lat), 4326),
                             (SELECT geom FROM _state_boundary))
            """,
            rows,
        )
        cur.execute("SELECT count(*) FROM towers")
        n = cur.fetchone()[0]
    conn.commit()
    print(f"towers: {len(df)} in bbox -> {n} in Penang State polygon -> loaded")


def load_villages(conn):
    fc = json.loads(VILLAGES.read_text(encoding="utf-8"))
    rows = [
        (
            f["properties"].get("name", ""),
            f["properties"].get("place", ""),
            f["geometry"]["coordinates"][0],
            f["geometry"]["coordinates"][1],
        )
        for f in fc["features"]
    ]
    with conn.cursor() as cur:
        cur.execute("TRUNCATE villages RESTART IDENTITY")
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO villages (name, place_type, geom)
            SELECT name, place_type, ST_SetSRID(ST_MakePoint(lon, lat), 4326)
            FROM (VALUES %s) AS v(name, place_type, lon, lat)
            """,
            rows,
        )
    conn.commit()
    print(f"villages: {len(rows)} loaded")


def main():
    conn = get_connection()
    try:
        ensure_schema(conn)
        load_boundaries(conn)   # must run before towers (ST_Within needs it)
        load_towers(conn)
        load_villages(conn)
    finally:
        conn.close()
    print("done.")


if __name__ == "__main__":
    main()
