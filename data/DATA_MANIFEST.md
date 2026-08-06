# SiteSense 5G — Data Manifest

Pilot scope: **Penang Island, Malaysia** (bbox lat 5.1–5.6, lon 100.1–100.6).
Every layer below is tagged by whether it is usable *inside the Penang pilot*.

## ✅ Active Penang layers (the build uses these)

| Folder / file | Source | Rows/extent | Role | Status |
|---|---|---|---|---|
| `towers_penang/502.csv` | OpenCelliD Malaysia (MCC 502) | 5,284 bbox → **4,643 in state** (2,895 GSM · 1,746 LTE · 2 UMTS · 0 NR) / **2,187 on island** (944 LTE) | Tower locations + `range` + `samples`. App filters to the **selected scope** polygon at load. | ✅ in use (Fn 1 & 2) |
| `boundaries/penang_state.geojson` | OSM Nominatim (state admin polygon) | Penang State MultiPolygon (island + Seberang Perai) | Clips to full state (**default scope**) | ✅ via `data_prep/fetch_penang_boundary.py` |
| `boundaries/penang_island.geojson` | OSM Nominatim (island polygon) | Penang Island MultiPolygon | Clips to island (drill-down scope) | ✅ via `data_prep/fetch_penang_boundary.py` |
| `population/penang_state_ppp_2020.tif` | WorldPop 2020 clipped to state | 452×557 px, **1,746,105 people** | Uncovered population — **State scope** | ✅ via `data_prep/clip_to_state.py` |
| `population/penang_island_ppp_2020.tif` | WorldPop 2020 clipped to island | 206×272 px, **793,788 people** | Uncovered population — Island scope | ✅ via `data_prep/clip_to_island.py` |
| `population/penang_ppp_2020.tif` | WorldPop 2020 bbox clip | 600×600 px, 2,200,402 (incl. Kedah spill) | intermediate — source for both scope clips | ⚠️ kept as intermediate |
| `population/mys_ppp_2020_UNadj.tif` | WorldPop MYS 2020 national | 157 MB | source raster (git-ignore; do NOT commit) | ⚠️ keep local only |
| `villages/penang_places_state.geojson` | OSM places clipped to state | **366 points** | Villages out of coverage — State scope | ✅ via `data_prep/clip_to_state.py` |
| `villages/penang_places_island.geojson` | OSM places clipped to island | **80 points** | Villages out of coverage — Island scope | ✅ via `data_prep/clip_to_island.py` |
| `villages/penang_places.geojson` | OSM Overpass (bbox) | 536 points | intermediate — source for both scope clips | ⚠️ kept as intermediate |
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
