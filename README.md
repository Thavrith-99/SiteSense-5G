# SiteSense 5G — GeoAI for Telecom Tower Siting & 5G Coverage-Gap Planning

**Team Neural Shield (KH-002) · ASEAN GeoAI Fusion 2026**

A GeoAI decision-support dashboard that fuses tower locations, network load,
population and place data to answer one question for any district:
**where should the next 5G tower go?**

Pilot area: **Penang Island, Malaysia.**

## What it does

1. **See the network** — every mobile cell on an interactive map.
2. **Find the coverage gap** — population and villages outside 4G/5G coverage.
   *Penang Island: ~56,600 people and 10 kampungs out of broadband coverage.*
3. **Flag overloaded towers** — cells carrying an unusually high load.
4. **Recommend new 5G sites** — a ranked, non-overlapping shortlist from a
   greedy maximum-coverage model, each with an explainable "people gained" score.
   *Top 5 sites (1 km range) close ~62% of the gap.*

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11+ (developed on 3.14).

## Live demo

https://sitesense-5g-mvp-kxs59yiiiouymaaipuyvb7.streamlit.app/

## Project layout

| File | Role |
|------|------|
| `app.py` | Streamlit dashboard (UI + wiring) |
| `coverage.py` | Function 1 — coverage-gap computation |
| `recommend.py` | Function 3 — greedy max-coverage site recommender |
| `data_prep/clip_worldpop.py` | Clip national WorldPop raster to Penang |
| `data_prep/fetch_osm_villages.py` | Fetch OSM place points via Overpass |
| `data/DATA_MANIFEST.md` | Full data provenance & usability notes |

## Data sources

| Layer | Source | License |
|-------|--------|---------|
| Mobile cells (`502.csv`) | [OpenCelliD](https://opencellid.org) Malaysia (MCC 502) | CC BY-SA 4.0 |
| Population (`penang_ppp_2020.tif`) | [WorldPop](https://www.worldpop.org) 2020 UN-adjusted 100 m | CC BY 4.0 |
| Places / villages | [OpenStreetMap](https://www.openstreetmap.org) via Overpass | ODbL |

See `data/DATA_MANIFEST.md` for full provenance, CRS, and the datasets that
were audited and **excluded** (Indonesian tower registries; a Kuala-Lumpur
drive-test set reserved for an optional signal model).

## Method notes

- Coverage uses each cell's reported `range` as its footprint; distances use a
  local metre-per-degree scaling calibrated to Penang's latitude.
- Penang currently has **zero 5G NR cells** in the open data, so the tool frames
  the problem as a **greenfield 5G build-out** on top of the 4G footprint.
- Population and villages are clipped to the **Penang Island boundary** (from
  OpenStreetMap), so counts reflect the island pilot (~793,788 people), not the
  mainland. Towers are likewise filtered to the island.

## Deploy (Streamlit Community Cloud)

1. Push this folder to a GitHub repo (the 157 MB national raster is git-ignored;
   the committed data is < 6 MB).
2. On [share.streamlit.io](https://share.streamlit.io), create an app from the
   repo with `app.py` as the entry point.
3. Streamlit builds from `requirements.txt` and serves a public URL.
