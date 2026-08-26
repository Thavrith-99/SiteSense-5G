# SiteSense 5G — GeoAI for Telecom Tower Site Selection & 5G Coverage-Gap Planning

**Team Neural Shield (KH-002) · ASEAN GeoAI Fusion 2026 · HACK_KH_102**

A GeoAI decision-support dashboard that fuses tower locations, network load,
population, settlement and terrain data to answer one question with data
instead of guesswork: **where should the next 5G tower go?**

Pilot area: **Penang, Malaysia** — with a live scope switch between the full
**Penang State** (island + mainland Seberang Perai) and **Penang Island**.

**Live:** https://sitesense5g.com

## What it does

1. **See the network** — every mobile cell on an interactive map.
2. **Find the coverage gap** — overlay population + settlements to quantify who is
   outside estimated 4G/5G coverage.
   *Penang State: ~77,362 people and 31 kampungs in estimated underserved areas.*
3. **Spot the strain** — flag existing towers that are overloaded relative to a
   configurable load threshold. *(193 overloaded at the default 90th-percentile.)*
4. **Recommend new 5G sites** — a ranked, non-overlapping shortlist from a
   **weighted multi-criteria (MCDA)** engine, each site with an estimated **capex**,
   a **people-per-$1,000** efficiency figure, and a **3-phase rollout** ordered by ROI.
   Three selectable weighting **profiles**: *Coverage-first*, *Balanced*, *Cost-efficient*.
   *Top-5 sites (1.7 km range, Coverage-first) reach ~50,800 people ≈ 66% of the
   State gap, ~$2.68M estimated capex.*
5. **Forecast network trends (LSTM)** — a trained model predicts next-quarter mobile
   download throughput for a Penang map tile from its last 4 quarters of real
   measurements (Ookla Open Data), served via a separate FastAPI endpoint.

## Architecture (3-tier, per the bootcamp's PDGS Canvas stack)

```
Data tier          Service tier            Presentation tier
PostgreSQL/PostGIS →  FastAPI            →  Streamlit + Leafmap
(towers, villages,   (/towers, /coverage-   (interactive map, KPIs,
 boundaries;          gap, /recommend-       MCDA profile selector,
 spatial ST_Within    sites, /predict-       State⇄Island switch,
 queries)             penang-network, ...)   live LSTM panel)
Trained LSTM (.keras) ┘
```

- **Automatic fallback:** if PostGIS is unreachable, the app and API silently fall
  back to the underlying CSV/GeoJSON files — same data, same results.
- Raster layers (population, slope) are always file-based.

## Run locally

**Full stack (PostGIS + FastAPI + Streamlit) via Docker:**
```bash
cp .env.example .env         # set DB_PASSWORD (letters + numbers only)
docker compose up -d --build
# dashboard → http://localhost:8501   ·   API docs → http://localhost:8001/docs
```

**Dashboard only (file-fallback, no database):**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Python 3.12 recommended (TensorFlow has no 3.14 wheel yet; the dashboard itself
runs on 3.12–3.14).

## Project layout

| Path | Role |
|------|------|
| `app.py` | Streamlit dashboard (UI + wiring) |
| `data_access.py` | Data layer — PostGIS queries with automatic file fallback |
| `coverage.py` | Coverage-gap computation |
| `recommend.py` | Weighted MCDA site-scoring engine (3 profiles, capex, phasing) |
| `db.py`, `db/schema.sql` | PostGIS connection + spatial schema |
| `data_prep/load_postgis.py` | Load towers/villages/boundaries into PostGIS |
| `api/main.py`, `api/requirements.txt` | FastAPI service (Functions 1–3 + LSTMs) |
| `ml/lstm_penang_model.keras` | Trained Penang throughput LSTM |
| `Dockerfile.app`, `Dockerfile.api`, `docker-compose.yml` | Containerized full stack |
| `Caddyfile` | Reverse proxy — automatic HTTPS + social link preview |
| `tests/` | Integration/regression tests (file-mode + PostGIS-mode) |

## Data sources

| Layer | Source | License |
|-------|--------|---------|
| Mobile cells (`502.csv`) | [OpenCelliD](https://opencellid.org) Malaysia (MCC 502) | CC BY-SA 4.0 |
| Population (`*_ppp_2020.tif`) | [WorldPop](https://www.worldpop.org) 2020 UN-adjusted 100 m | CC BY 4.0 |
| Places / villages | [OpenStreetMap](https://www.openstreetmap.org) via Overpass | ODbL |
| Terrain / slope (`*_slope.tif`) | SRTM 30 m via AWS Terrain Tiles | public domain |
| Network throughput (LSTM) | [Ookla Open Data](https://github.com/teamookla/ookla-open-data) quarterly tiles | CC BY-NC-SA 4.0 |

## Method notes

- **Coverage** uses each cell's reported OpenCelliD `range` as a footprint proxy —
  an *estimate*, not verified operator coverage.
- **Site scoring (MCDA)** is a weighted overlay of four normalized factors —
  population reached, backhaul proximity, overload relief, SRTM slope — adapting the
  bootcamp's `GeoAI_Risk.ipynb` scoring technique. Weights are team-set /
  expert-judgement, not learned from data. **No model training** is involved here.
- **Cost** = $150k base + $62.5k/km backhaul fiber (industry benchmark, PatentPC 2026),
  labelled order-of-magnitude, not a site-specific quote.
- **LSTM** is trained on real Penang Ookla history (quarterly, Q1 2019–Q2 2026);
  it forecasts a *trend*, not live per-second signal.
- The open (OpenCelliD) data shows **zero 5G NR cells** in Penang — a data-coverage
  limitation, not that Penang lacks 5G (real 5G exists via Malaysia's DNB network).
- Everything is clipped to the selected **scope polygon**: **Penang State**
  (~1,746,105 people) or **Penang Island** (~793,788); the sidebar recomputes on switch.

## Deploy (DigitalOcean droplet + docker-compose)

The live site runs the full containerized stack on a single DigitalOcean droplet,
behind **Caddy** for automatic HTTPS (Let's Encrypt):

```bash
# on the droplet
git clone <repo> && cd SiteSense-5G
cp .env.example .env         # set a strong DB_PASSWORD
docker compose up -d --build
# Caddy serves https://sitesense5g.com (dashboard) automatically
```

Hardened with a host firewall (SSH/HTTP/HTTPS only), fail2ban, HSTS + security
headers, and no plaintext bypass ports. The stack is self-contained and reproducible
(no external Google Earth Engine dependency — the same scoring runs locally via
rasterio/numpy).
