# SiteSense 5G — GeoAI for Telecom Tower Siting & 5G Coverage-Gap Planning

**Team Neural Shield (KH-002) · ASEAN GeoAI Fusion 2026**

A GeoAI decision-support dashboard that fuses tower locations, network load,
population and place data to answer one question for any district:
**where should the next 5G tower go?**

Pilot area: **Penang, Malaysia** — with a live scope switch between the full
**Penang State** (island + mainland) and **Penang Island**.

## What it does

1. **See the network** — every mobile cell on an interactive map.
2. **Find the coverage gap** — population and villages outside 4G/5G coverage.
   *Penang State: ~77,400 people and 31 kampungs out of broadband coverage
   (Penang Island drill-down: ~56,600 people, 10 kampungs).*
3. **Flag overloaded towers** — cells carrying an unusually high load.
4. **Recommend new 5G sites** — a ranked, non-overlapping shortlist from a
   greedy maximum-coverage model, each with an explainable "people gained" score.
   *Top 5 sites (1 km range) close ~48% of the State gap (~62% for the island).*

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
- The open (OpenCelliD) data shows **zero 5G NR cells** in Penang — a data-coverage
  limitation (crowdsourced data under-captures 5G NR), not that Penang lacks 5G
  (real 5G exists via Malaysia's DNB network). The tool therefore plans **greenfield
  5G placement** from the 4G footprint, and works on an operator's real 5G inventory
  when supplied.
- Population, villages and towers are clipped to the selected **scope polygon**
  (from OpenStreetMap): **Penang State** (~1,746,105 people) or **Penang Island**
  (~793,788). The sidebar switches between them and everything recomputes.

## Deploy (Streamlit Community Cloud)

1. Push this folder to a GitHub repo (the 157 MB national raster is git-ignored;
   the committed data is < 6 MB).
2. On [share.streamlit.io](https://share.streamlit.io), create an app from the
   repo with `app.py` as the entry point.
3. Streamlit builds from `requirements.txt` and serves a public URL.
