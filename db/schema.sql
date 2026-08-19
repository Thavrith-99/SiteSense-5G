-- SiteSense 5G — PostGIS schema (Physical Finale architecture)
--
-- Migrates the vector inputs currently read from flat files
-- (data/towers_penang/502.csv, data/villages/*.geojson,
-- data/boundaries/*.geojson) into spatial tables. Population stays a
-- WorldPop GeoTIFF (file-based) — coverage.py/recommend.py's raster
-- box-filter search is fast in numpy and PostGIS raster brings no pitch
-- benefit here, so it is intentionally NOT migrated.
--
-- Run once against a fresh database:
--   psql "$DATABASE_URL" -f db/schema.sql

CREATE EXTENSION IF NOT EXISTS postgis;

-- --------------------------------------------------------------------------
-- boundaries — the two study-area polygons the sidebar toggles between.
-- Villages and towers are stored ONCE (state-clipped, the wider scope);
-- the island subset is derived at query time via ST_Within against this
-- table, instead of keeping duplicate island-clipped files/rows.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS boundaries (
    id      SERIAL PRIMARY KEY,
    scope   TEXT NOT NULL UNIQUE,              -- 'penang_state' | 'penang_island'
    label   TEXT NOT NULL,                     -- e.g. 'Penang State'
    geom    GEOMETRY(MULTIPOLYGON, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS boundaries_gix ON boundaries USING GIST (geom);

-- --------------------------------------------------------------------------
-- towers — OpenCelliD Malaysia cells (502.csv), state-clipped.
-- Mirrors app.py's load_towers(): same columns, same LAT/LON bbox +
-- polygon pre-filter, done once here instead of on every Streamlit rerun.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS towers (
    id          SERIAL PRIMARY KEY,
    radio       TEXT NOT NULL,                 -- 'GSM' | 'UMTS' | 'LTE' | 'NR'
    mcc         INT,
    net         INT,
    area        INT,
    cell        BIGINT,
    range_m     DOUBLE PRECISION,               -- coverage.py's `range`
    samples     INT,                            -- load proxy (Function 2 overload)
    changeable  BOOLEAN,
    geom        GEOMETRY(POINT, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS towers_gix ON towers USING GIST (geom);
CREATE INDEX IF NOT EXISTS towers_radio_idx ON towers (radio);

-- --------------------------------------------------------------------------
-- villages — OSM place points (penang_places_state.geojson).
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS villages (
    id          SERIAL PRIMARY KEY,
    name        TEXT,
    place_type  TEXT,                           -- 'village' | 'hamlet' | 'town' | ...
    geom        GEOMETRY(POINT, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS villages_gix ON villages USING GIST (geom);

-- --------------------------------------------------------------------------
-- recommended_sites — persists recommend.py's greedy max-coverage output,
-- one row per site per run, so a FastAPI /recommend-sites call is
-- queryable/auditable afterwards instead of living only in a Streamlit
-- session. recommend.py's algorithm itself is unchanged by this migration.
-- --------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recommended_sites (
    id                  SERIAL PRIMARY KEY,
    run_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    scope               TEXT NOT NULL,          -- 'penang_state' | 'penang_island'
    rank                INT NOT NULL,
    geom                GEOMETRY(POINT, 4326) NOT NULL,
    new_range_m         DOUBLE PRECISION NOT NULL,
    people_gained       DOUBLE PRECISION NOT NULL,
    cumulative_gained   DOUBLE PRECISION NOT NULL,
    overloaded_relieved INT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS recommended_sites_gix ON recommended_sites USING GIST (geom);
CREATE INDEX IF NOT EXISTS recommended_sites_run_idx ON recommended_sites (scope, run_at DESC);
