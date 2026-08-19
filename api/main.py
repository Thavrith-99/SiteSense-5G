"""
SiteSense 5G — FastAPI service layer (Physical Finale architecture).

Two families of endpoints:
  1. /towers, /coverage-gap, /recommend-sites — call data_access.py, the
     exact same functions app.py (Streamlit) uses, so the API and the
     dashboard can never disagree on numbers.
  2. /predict-rsrp (+ /sample-request, /health, /version) — the LSTM signal-
     quality model, adapted from the bootcamp's Session 4 deployment
     notebook (`Session_4_Simple_LSTM_API_Deployment_Dashboard.ipynb`).

IMPORTANT — honesty caveat, keep in the pitch/deck: the LSTM is trained on
`data/signal_kl/raw_dataset_kl.csv`, real Kuala Lumpur drive-test data — NOT
Penang. It demonstrates the deployable ML pipeline on real Malaysian signal
data; it is not fitted on Penang-specific observations. Every /predict-rsrp
response includes a `caveat` field saying so — don't strip it out for demos.

Run (from SiteSense5G_App/, using the isolated ML venv — TensorFlow/rasterio/
psycopg2 all live there, not in the Streamlit app's lighter requirements.txt):
    .venv-ml/Scripts/uvicorn.exe api.main:app --reload --port 8001
Docs: http://127.0.0.1:8001/docs
"""
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))  # coverage.py / recommend.py / data_access.py / db.py live at the app root

import data_access as da  # noqa: E402

app = FastAPI(
    title="SiteSense 5G API",
    description="Tower/coverage-gap/site-recommendation data + LSTM RSRP prediction.",
    version="1.0.0",
)

SCOPE_SLUG = {"state": "Penang State", "island": "Penang Island"}


def _resolve_scope(scope: str) -> str:
    if scope not in SCOPE_SLUG:
        raise HTTPException(status_code=400, detail=f"scope must be one of {list(SCOPE_SLUG)}")
    return SCOPE_SLUG[scope]


def _parse_radios(radio: Optional[str]) -> tuple:
    if not radio:
        return ("NR", "LTE")
    return tuple(r.strip().upper() for r in radio.split(","))


# --------------------------------------------------------------------------
# Health / meta
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok", "db_connected": da.using_db(),
        "kl_lstm_loaded": _lstm.model is not None,
        "penang_lstm_loaded": _penang_lstm.model is not None,
    }


# --------------------------------------------------------------------------
# Function 1/2/3 endpoints — reuse data_access.py (same code app.py uses)
# --------------------------------------------------------------------------
@app.get("/towers")
def towers(
    scope: Literal["state", "island"] = "state",
    radio: Optional[str] = Query(None, description="Comma-separated radio types, e.g. NR,LTE. Default: NR,LTE"),
    max_range: float = Query(100_000, description="Max coverage range (m) to include"),
):
    scope_name = _resolve_scope(scope)
    radios = _parse_radios(radio)
    df = da.load_towers(scope_name)
    sub = df[df["radio"].isin(radios) & (df["range"] <= max_range)]
    return {
        "scope": scope, "count": int(len(sub)),
        "towers": [
            {"radio": r.radio, "range_m": float(r.range), "samples": int(r.samples),
             "cell": int(r.cell) if pd.notna(r.cell) else None,
             "lon": float(r.lon), "lat": float(r.lat)}
            for r in sub.itertuples()
        ],
    }


@app.get("/coverage-gap")
def coverage_gap(
    scope: Literal["state", "island"] = "state",
    radio: Optional[str] = Query(None),
    max_range: float = Query(100_000),
):
    """Function 1 — estimated coverage gap (people + villages)."""
    scope_name = _resolve_scope(scope)
    radios = _parse_radios(radio)
    g = da.compute_gap(scope_name, radios, max_range)
    return {
        "scope": scope,
        "total_pop": round(g["total_pop"]),
        "covered_pop": round(g["covered_pop"]),
        "uncovered_pop": round(g["uncovered_pop"]),
        "pct_covered": round(g["pct_covered"], 2),
        "n_places": g["n_places"],
        "n_uncovered_places": g["n_uncovered_places"],
        "note": "Estimated from OpenCelliD range proxy, not verified operator coverage.",
    }


@app.get("/recommend-sites")
def recommend_sites(
    scope: Literal["state", "island"] = "state",
    radio: Optional[str] = Query(None),
    max_range: float = Query(100_000),
    load_percentile: int = Query(90, ge=0, le=100, description="Overload threshold percentile on `samples`"),
    new_range: float = Query(1000, description="Candidate new-tower coverage range (m)"),
    n_sites: int = Query(5, ge=1, le=20),
    profile: Literal["Coverage-first", "Balanced", "Cost-efficient"] = Query(
        "Coverage-first", description="Weighted multi-criteria scoring profile — "
        "'Coverage-first' reproduces the original population-only ranking."),
):
    """Function 3 — weighted multi-criteria (MCDA) new-5G-site shortlist.
    Weights are team-set/expert-judgement, not learned from data."""
    scope_name = _resolve_scope(scope)
    radios = _parse_radios(radio)
    sites = da.compute_sites(scope_name, radios, max_range, load_percentile, new_range, n_sites, profile)
    return {
        "scope": scope, "new_range_m": new_range, "profile": profile,
        "sites": [
            {"rank": s["rank"], "lon": s["lon"], "lat": s["lat"],
             "people_gained": round(s["people_gained"]),
             "cumulative_gained": round(s["cumulative_gained"]),
             "overloaded_relieved": s["overloaded_relieved"],
             "backhaul_distance_m": round(s["backhaul_distance_m"]) if s.get("backhaul_distance_m") is not None else None,
             "slope_deg": round(s["slope_deg"], 1) if s.get("slope_deg") is not None else None,
             "est_capex_usd": s["est_capex_usd"],
             "people_per_1000usd": s["people_per_1000usd"],
             "phase": s["phase"]}
            for s in sites
        ],
        "note": "Preliminary — weighted multi-criteria siting (population reached, backhaul "
                "proximity to existing towers, overload relief, SRTM slope penalty), not final "
                "engineering placement. Weights are team-set, not learned from data. Capex estimate "
                "uses a cited industry benchmark (PatentPC 2026), not a site-specific quote.",
    }


# --------------------------------------------------------------------------
# LSTM RSRP prediction — adapted from Session 4's deployment notebook
# --------------------------------------------------------------------------
LOOKBACK = 10
MODEL_VERSION = "LSTM-RSRP-2.0-LOCATION-SiteSense5G"
DATA_FILE = APP_DIR / "data" / "signal_kl" / "raw_dataset_kl.csv"
MODEL_FILE = Path(__file__).resolve().parent.parent / "ml" / "lstm_rsrp_model.keras"


class _LSTMService:
    """Lazy-loads the KL-trained LSTM + re-derives the Session-2 scalers
    from the same CSV/split, exactly as the bootcamp's Session 4 notebook
    does — training and serving must stay in lockstep on preprocessing."""

    def __init__(self):
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.features = None
        self.df = None
        self._error = None

    def load(self):
        if self.model is not None or self._error is not None:
            return
        try:
            import tensorflow as tf
            from sklearn.preprocessing import StandardScaler

            if not DATA_FILE.exists():
                raise FileNotFoundError(f"Missing {DATA_FILE}")
            if not MODEL_FILE.exists():
                raise FileNotFoundError(f"Missing {MODEL_FILE} — run ml/train_lstm_rsrp.py first")

            df = pd.read_csv(DATA_FILE)
            required = ["Timestamp", "SessionID", "Level", "Latitude", "Longitude"]
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y.%m.%d_%H.%M.%S", errors="coerce")
            candidates = ["Level", "SNR", "CQI", "DL_bitrate", "UL_bitrate", "Speed", "Latitude", "Longitude"]
            features = [c for c in candidates if c in df.columns]
            df = df.dropna(subset=required).copy()
            df = df.sort_values(["SessionID", "Timestamp"]).reset_index(drop=True)
            df[features] = df.groupby("SessionID")[features].transform(lambda x: x.ffill().bfill())
            df[features] = df[features].fillna(df[features].median(numeric_only=True))

            sessions = df.groupby("SessionID")["Timestamp"].min().sort_values().index.tolist()
            train_end = max(1, int(len(sessions) * 0.70))
            train_mask = df["SessionID"].isin(sessions[:train_end])

            self.feature_scaler = StandardScaler().fit(df.loc[train_mask, features])
            self.target_scaler = StandardScaler().fit(df.loc[train_mask, ["Level"]])
            self.features = features
            self.df = df
            self.model = tf.keras.models.load_model(MODEL_FILE)
        except Exception as e:
            self._error = str(e)

    def predict(self, observations: List[dict]) -> float:
        self.load()
        if self._error:
            raise HTTPException(status_code=503, detail=f"LSTM unavailable: {self._error}")
        if len(observations) != LOOKBACK:
            raise HTTPException(status_code=400, detail=f"Exactly {LOOKBACK} observations are required.")
        frame = pd.DataFrame(observations)
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
        scaled = self.feature_scaler.transform(frame[self.features])
        sequence = scaled.reshape(1, LOOKBACK, len(self.features)).astype(np.float32)
        pred_scaled = self.model.predict(sequence, verbose=0)
        return float(self.target_scaler.inverse_transform(pred_scaled)[0, 0])


_lstm = _LSTMService()


def _signal_class(rsrp: float) -> str:
    if rsrp >= -90:
        return "Good"
    if rsrp >= -105:
        return "Weak"
    return "Poor"


class Observation(BaseModel):
    Level: float
    SNR: float
    CQI: float
    DL_bitrate: float = Field(ge=0)
    UL_bitrate: float = Field(ge=0)
    Speed: float = Field(ge=0)
    Latitude: float = Field(ge=-90, le=90)
    Longitude: float = Field(ge=-180, le=180)


class PredictionRequest(BaseModel):
    observations: List[Observation]


@app.get("/version")
def version():
    _lstm.load()
    return {
        "api_version": app.version,
        "model_version": MODEL_VERSION,
        "lookback": LOOKBACK,
        "features": _lstm.features,
        "trained_on": "Kuala Lumpur drive-test data (raw_dataset_kl.csv) — NOT Penang",
    }


@app.get("/sample-request")
def sample_request():
    _lstm.load()
    if _lstm._error:
        raise HTTPException(status_code=503, detail=f"LSTM unavailable: {_lstm._error}")
    latest_session = _lstm.df["SessionID"].iloc[-1]
    sample = (
        _lstm.df[_lstm.df["SessionID"] == latest_session]
        .sort_values("Timestamp")
        .tail(LOOKBACK)
    )
    if len(sample) < LOOKBACK:
        raise HTTPException(status_code=500, detail="Sample session has fewer than 10 rows.")
    return {"observations": sample[_lstm.features].to_dict(orient="records")}


@app.post("/predict-rsrp")
def predict_rsrp(request: PredictionRequest):
    start = time.perf_counter()
    records = [o.model_dump() for o in request.observations]
    prediction = _lstm.predict(records)
    last = records[-1]
    return {
        "predicted_rsrp_dbm": round(prediction, 2),
        "signal_class": _signal_class(prediction),
        "latitude": last["Latitude"],
        "longitude": last["Longitude"],
        "model_version": MODEL_VERSION,
        "response_time_ms": round((time.perf_counter() - start) * 1000, 2),
        "caveat": "Trained on Kuala Lumpur drive-test data, not Penang — "
                  "demonstrates the deployable pipeline, not a Penang-fitted prediction.",
    }


# --------------------------------------------------------------------------
# Penang network-performance LSTM — genuinely Penang-specific, genuinely
# time-ordered (real Ookla Open Data, quarterly tile history). See
# ml/train_lstm_penang.py's module docstring for the KL-vs-Penang mapping
# (quadkey~SessionID, quarter~Timestamp, avg_d_kbps~RSRP).
# --------------------------------------------------------------------------
LOOKBACK_PENANG = 4  # quarters
TARGET_PENANG = "avg_d_kbps"
PENANG_MODEL_FILE = Path(__file__).resolve().parent.parent / "ml" / "lstm_penang_model.keras"
PENANG_MODEL_VERSION = "LSTM-PENANG-NETWORK-1.0"


class _PenangLSTMService:
    def __init__(self):
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.features = None
        self.df = None
        self._error = None

    def load(self):
        if self.model is not None or self._error is not None:
            return
        try:
            import tensorflow as tf
            from sklearn.preprocessing import StandardScaler
            from ml.train_lstm_penang import load_and_clean, split_tiles, FEATURES, TARGET

            if not PENANG_MODEL_FILE.exists():
                raise FileNotFoundError(f"Missing {PENANG_MODEL_FILE} — run ml/train_lstm_penang.py first")

            df, features = load_and_clean()
            train_tiles, _, _ = split_tiles(df)
            train_mask = df["quadkey"].isin(train_tiles)

            self.feature_scaler = StandardScaler().fit(df.loc[train_mask, features])
            self.target_scaler = StandardScaler().fit(df.loc[train_mask, [TARGET]])
            self.features = features
            self.df = df
            self.model = tf.keras.models.load_model(PENANG_MODEL_FILE)
        except Exception as e:
            self._error = str(e)

    def predict(self, observations: List[dict]) -> float:
        self.load()
        if self._error:
            raise HTTPException(status_code=503, detail=f"Penang LSTM unavailable: {self._error}")
        if len(observations) != LOOKBACK_PENANG:
            raise HTTPException(status_code=400, detail=f"Exactly {LOOKBACK_PENANG} quarterly observations are required.")
        frame = pd.DataFrame(observations)
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
        scaled = self.feature_scaler.transform(frame[self.features])
        sequence = scaled.reshape(1, LOOKBACK_PENANG, len(self.features)).astype(np.float32)
        pred_scaled = self.model.predict(sequence, verbose=0)
        return float(self.target_scaler.inverse_transform(pred_scaled)[0, 0])


_penang_lstm = _PenangLSTMService()


class PenangObservation(BaseModel):
    avg_d_kbps: float = Field(ge=0)
    avg_u_kbps: float = Field(ge=0)
    avg_lat_ms: float = Field(ge=0)
    tests: float = Field(ge=0)


class PenangPredictionRequest(BaseModel):
    observations: List[PenangObservation]


@app.get("/version-penang")
def version_penang():
    _penang_lstm.load()
    return {
        "api_version": app.version,
        "model_version": PENANG_MODEL_VERSION,
        "lookback_quarters": LOOKBACK_PENANG,
        "features": _penang_lstm.features,
        "trained_on": "Real Penang mobile-network tiles, Ookla Open Data, Q1 2019 - Q2 2026 (quarterly)",
    }


@app.get("/sample-request-penang")
def sample_request_penang():
    _penang_lstm.load()
    if _penang_lstm._error:
        raise HTTPException(status_code=503, detail=f"Penang LSTM unavailable: {_penang_lstm._error}")
    counts = _penang_lstm.df.groupby("quadkey").size()
    eligible = counts[counts >= LOOKBACK_PENANG + 1].index
    if len(eligible) == 0:
        raise HTTPException(status_code=500, detail="No Penang tile has enough quarterly history.")
    # Pick a well-observed tile (highest average real test volume) rather than
    # an arbitrary/sparse one — makes for a representative demo, not a
    # mostly-imputed corner case.
    busiest = (_penang_lstm.df[_penang_lstm.df["quadkey"].isin(eligible)]
              .groupby("quadkey")["tests"].mean().idxmax())
    qk = str(busiest)
    g = (_penang_lstm.df[_penang_lstm.df["quadkey"] == qk]
        .sort_values("q_index").tail(LOOKBACK_PENANG + 1))
    obs = [{k: float(v) for k, v in row.items()}
          for row in g.iloc[:-1][_penang_lstm.features].to_dict(orient="records")]
    return {"quadkey": qk, "observations": obs,
            "actual_next_avg_d_kbps": round(float(g.iloc[-1][TARGET_PENANG]))}


@app.post("/predict-penang-network")
def predict_penang_network(request: PenangPredictionRequest):
    """Predicts the NEXT QUARTER's average mobile download throughput for a
    Penang tile from its last 4 quarters. Real Penang data end to end —
    no transfer-learning caveat needed (contrast with /predict-rsrp)."""
    start = time.perf_counter()
    records = [o.model_dump() for o in request.observations]
    prediction = _penang_lstm.predict(records)
    return {
        "predicted_avg_d_kbps": round(prediction),
        "predicted_mbps": round(prediction / 1000, 1),
        "model_version": PENANG_MODEL_VERSION,
        "response_time_ms": round((time.perf_counter() - start) * 1000, 2),
        "note": "Real Penang tile data (Ookla Open Data, quarterly, Q1 2019-Q2 2026). "
                "Predicts next-quarter average mobile download throughput from the "
                "tile's last 4 quarters of real measurements.",
    }
