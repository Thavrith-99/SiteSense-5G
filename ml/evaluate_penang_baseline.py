"""
Mentor #6 — compare the Penang LSTM against a SIMPLE forecasting baseline on
the SAME held-out (unseen) test tiles.

Baseline = naive persistence: "next quarter = last observed quarter"
(predict avg_d_kbps[t] = avg_d_kbps[t-1]). This is the standard yardstick a
forecasting model must beat to be worth its complexity.

This script is TensorFlow-free: it recomputes the naive baseline directly from
the Ookla data using the exact same tile-level split (SEED=42, 70/15/15) and
LOOKBACK as ml/train_lstm_penang.py, so the naive numbers are on identical test
samples. The LSTM's own test metrics are read from RECORDED_LSTM (captured from
the training run); when train_lstm_penang.py is re-run on a TF machine it now
recomputes and overwrites these in the same JSON.

Run:  py ml/evaluate_penang_baseline.py
Out:  ml/lstm_penang_metrics.json  (read by the dashboard's LSTM panel)
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- must match ml/train_lstm_penang.py exactly ---------------------------
SEED = 42
LOOKBACK = 4
MIN_REAL_QUARTERS = 8
FEATURES = ["avg_d_kbps", "avg_u_kbps", "avg_lat_ms", "tests"]
TARGET = "avg_d_kbps"
APP = Path(__file__).resolve().parent.parent
DATA_FILE = APP / "data" / "ookla_penang" / "penang_quarterly.csv"
OUT_JSON = Path(__file__).resolve().parent / "lstm_penang_metrics.json"

# LSTM test metrics recorded from the training run (Session2_LSTM_Training_Penang).
# train_lstm_penang.py now recomputes and overwrites these when re-run with TF.
RECORDED_LSTM = {"mae_kbps": 41847.0, "rmse_kbps": 79011.0, "r2": 0.549,
                 "source": "recorded training run (ml/Session2_LSTM_Training_Penang.ipynb)"}


def load_and_clean():
    df = pd.read_csv(DATA_FILE, dtype={"quadkey": str})
    df["q_index"] = (df["year"] - df["year"].min()) * 4 + (df["quarter"] - 1)
    all_quarters = np.arange(df["q_index"].min(), df["q_index"].max() + 1)
    real_counts = df.groupby("quadkey")["q_index"].nunique()
    keep = real_counts[real_counts >= MIN_REAL_QUARTERS].index
    df = df[df["quadkey"].isin(keep)].copy()
    full = (df.set_index(["quadkey", "q_index"])[FEATURES]
              .reindex(pd.MultiIndex.from_product([df["quadkey"].unique(), all_quarters],
                                                  names=["quadkey", "q_index"])))
    full[FEATURES] = full.groupby(level="quadkey")[FEATURES].transform(lambda x: x.ffill().bfill())
    return full.dropna(subset=FEATURES).reset_index()


def split_tiles(df):
    tiles = sorted(df["quadkey"].unique())
    random.Random(SEED).shuffle(tiles)
    n = len(tiles)
    a = max(1, int(n * .70)); b = max(a + 1, int(n * .85))
    return tiles[:a], tiles[a:b], tiles[b:]


def naive_pairs(df, tile_ids):
    """(actual, persistence-forecast) pairs on the same indices the LSTM tests:
    for each tile, i in [LOOKBACK, len): actual = t[i], naive = t[i-1]."""
    actual, naive = [], []
    for _, g in df[df["quadkey"].isin(tile_ids)].groupby("quadkey"):
        v = g.sort_values("q_index")[TARGET].to_numpy(float)
        for i in range(LOOKBACK, len(v)):
            actual.append(v[i]); naive.append(v[i - 1])
    return np.asarray(actual), np.asarray(naive)


def main():
    df = load_and_clean()
    _, _, test_tiles = split_tiles(df)
    actual, naive = naive_pairs(df, test_tiles)

    n_mae = mean_absolute_error(actual, naive)
    n_rmse = float(np.sqrt(mean_squared_error(actual, naive)))
    n_r2 = r2_score(actual, naive)

    lstm = RECORDED_LSTM
    imp_mae = (n_mae - lstm["mae_kbps"]) / n_mae * 100.0
    imp_rmse = (n_rmse - lstm["rmse_kbps"]) / n_rmse * 100.0

    out = {
        "target": "avg_d_kbps — next-quarter average mobile download (kbps)",
        "test_tiles": len(test_tiles),
        "test_samples": int(actual.size),
        "lstm": lstm,
        "naive_persistence": {"mae_kbps": round(n_mae), "rmse_kbps": round(n_rmse), "r2": round(n_r2, 3)},
        "improvement_vs_naive": {"mae_pct": round(imp_mae, 1), "rmse_pct": round(imp_rmse, 1)},
        "note": ("Naive baseline = persistence (next quarter = last observed quarter). "
                 "Same tile-level 70/15/15 split (SEED=42), LOOKBACK=4 quarters, on unseen test tiles. "
                 "Positive improvement % = LSTM lower error than naive."),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Test tiles: {len(test_tiles)}   test samples: {actual.size}")
    print(f"Naive persistence  — MAE {n_mae:,.0f}  RMSE {n_rmse:,.0f}  R2 {n_r2:.3f}")
    print(f"LSTM (recorded)    — MAE {lstm['mae_kbps']:,.0f}  RMSE {lstm['rmse_kbps']:,.0f}  R2 {lstm['r2']:.3f}")
    print(f"LSTM improvement vs naive — MAE {imp_mae:+.1f}%   RMSE {imp_rmse:+.1f}%")
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
