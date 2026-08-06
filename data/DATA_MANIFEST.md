# SiteSense 5G — Data Manifest

Pilot scope: **Penang Island, Malaysia** (bbox lat 5.1–5.6, lon 100.1–100.6).
Every layer below is tagged by whether it is usable *inside the Penang pilot*.

## ✅ Active Penang layers (the build uses these)

| Folder / file | Source | Rows/extent | Role | Status |
|---|---|---|---|---|
| `towers_penang/502.csv` | OpenCelliD Malaysia (MCC 502) | 5,284 Penang-bbox cells → **2,187 on the island** (1,241 GSM · 944 LTE · 2 UMTS · **0 NR**) | Tower locations + `range` (coverage proxy) + `samples` (load proxy). App filters to the island polygon at load. | ✅ in use (Fn 1 & 2) |
| `boundaries/penang_island.geojson` | OSM Nominatim (island polygon) | Penang Island MultiPolygon | Clips population/villages/towers to the island | ✅ ready via `data_prep/fetch_penang_boundary.py` |
| `population/penang_island_ppp_2020.tif` | WorldPop MYS 2020 clipped to island | 206×272 px, **793,788 people** | People per pixel → uncovered population (**app uses this**) | ✅ ready via `data_prep/clip_to_island.py` |
| `population/penang_ppp_2020.tif` | WorldPop MYS 2020 bbox clip | 600×600 px, 2,200,402 people (island+mainland) | superseded by island clip | ⚠️ kept as intermediate |
| `population/mys_ppp_2020_UNadj.tif` | WorldPop MYS 2020 national | 157 MB | source raster (git-ignore; do NOT commit) | ⚠️ keep local only |
| `villages/penang_places_island.geojson` | OSM places clipped to island | **80 points** (of 536 in bbox) | Kampung/place points → villages out of coverage (**app uses this**) | ✅ ready via `data_prep/clip_to_island.py` |
| `villages/penang_places.geojson` | OSM Overpass (bbox) | 536 points | superseded by island clip | ⚠️ kept as intermediate |
| `terrain/` | SRTM DEM (no GEE needed) or GEE Copernicus | — | Slope/elevation → buildability of candidate sites | ⬜ TODO (Fn 3, optional) |

## ⚠️ Malaysian but NOT Penang (optional / stretch module)

| Folder / file | Source | Extent | Note |
|---|---|---|---|
| `signal_kl/raw_dataset_kl.csv` | Drive-test measurements | **Kuala Lumpur**, lat 3.057–3.08 (≈2 km corridor); 21,032×5G + 9,893×4G; 3 operators; RSRP/SNR/CQI | Real 5G signal KPIs but **KL, 0 points in Penang**. Use ONLY for the optional/transferable **LSTM signal-quality model** (Sept-finale / credibility booster), never as Penang coverage input. |

## ❌ Reference only — DO NOT fuse into the Penang pipeline

| File | Why excluded |
|---|---|
| `reference_indonesia/lokasi_menara_telekomunikasi.csv` | **Indonesia** (Jawa Barat); also has corrupted coordinates (lat = −694265). Kept only as a tower-registry schema example (owner, height, structure, year). |
| `reference_indonesia/menaratelepon_ar_50k.csv` | **Indonesia** (Kalimantan, lat ≈ −0.5); 136 rows. Schema reference only. |

## Open data still to collect
- **WorldPop** — already have MYS 2020 100 m; clip to Penang bbox. (WorldPop has no 2026 raster; 2020 constrained/UN-adj is the current best; a 2026 figure would be a projection.)
- **OSM villages/places** — Overpass/`osmnx`: `place=village|hamlet|town` within Penang bbox.
- **Admin boundaries** — GADM level-2 (districts) or HDX for Penang state + island polygon.
- **Terrain (DEM)** — GEE SRTM 30 m or Copernicus DEM, clipped to Penang → slope for site buildability.
