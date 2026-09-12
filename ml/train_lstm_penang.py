"""
Train the Penang network-performance LSTM — genuinely Penang-specific and
genuinely time-ordered, replacing the earlier KL-transfer framing.

Data: data/ookla_penang/penang_quarterly.csv (data_prep/fetch_ookla_penang.py)
— real Speedtest-by-Ookla measurements, aggregated per ~610m tile per
quarter, Q1 2019 - Q2 2026, filtered to Penang's bbox by decoding each row's
quadkey. Same two properties the KL dataset had (a "session" unit + a time
axis to sequence over) but built from Penang's own real data instead of
borrowing Kuala Lumpur's:

    KL model              Penang model
    ---------              ------------
    SessionID (drive test)  quadkey (map tile)
    Timestamp (per second)  quarter (Q1 2019 .. Q2 2026)
    Level (RSRP, dBm)       avg_d_kbps (download throughput)

Honest tradeoff, not a workaround: quarterly cadence instead of per-second,
network throughput/latency instead of raw RSRP. That's what real, open,
Penang-shaped time-series data actually looks like — no dataset lets you
have Penang AND per-second RSRP AND fully open at the same time.

Run (isolated ML venv):
    .venv-ml/Scripts/python.exe ml/train_lstm_penang.py
Outputs:
    ml/lstm_penang_model.keras
    ml/penang_training_history.png
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
LOOKBACK = 4  # quarters (1 year of history -> predict the next quarter)
MIN_REAL_QUARTERS = 8  # drop tiles with too little real (non-imputed) history
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

APP = Path(__file__).resolve().parent.parent
DATA_FILE = APP / "data" / "ookla_penang" / "penang_quarterly.csv"
MODEL_OUT = Path(__file__).resolve().parent / "lstm_penang_model.keras"
HISTORY_PLOT = Path(__file__).resolve().parent / "penang_training_history.png"

FEATURES = ["avg_d_kbps", "avg_u_kbps", "avg_lat_ms", "tests"]
TARGET = "avg_d_kbps"


def load_and_clean():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {DATA_FILE} — run data_prep/fetch_ookla_penang.py first."
        )
    df = pd.read_csv(DATA_FILE, dtype={"quadkey": str})  # preserve exact digit string, avoid huge-int quirks
    df["q_index"] = (df["year"] - df["year"].min()) * 4 + (df["quarter"] - 1)

    all_quarters = np.arange(df["q_index"].min(), df["q_index"].max() + 1)
    tiles = df["quadkey"].unique()

    # Real-quarter coverage count per tile, BEFORE any imputation — used to
    # drop tiles with too little genuine history (avoids training mostly on
    # forward-filled noise).
    real_counts = df.groupby("quadkey")["q_index"].nunique()
    keep_tiles = real_counts[real_counts >= MIN_REAL_QUARTERS].index
    df = df[df["quadkey"].isin(keep_tiles)].copy()
    print(f"Tiles with >= {MIN_REAL_QUARTERS} real quarters: {len(keep_tiles):,} of {len(tiles):,}")

    # Reindex each kept tile onto the FULL quarter range, then forward/back
    # fill gaps — same technique the KL notebook used per drive-test session,
    # applied here per map tile.
    full = (
        df.set_index(["quadkey", "q_index"])[FEATURES]
        .reindex(pd.MultiIndex.from_product([df["quadkey"].unique(), all_quarters],
                                            names=["quadkey", "q_index"]))
    )
    full[FEATURES] = full.groupby(level="quadkey")[FEATURES].transform(lambda x: x.ffill().bfill())
    full = full.dropna(subset=FEATURES).reset_index()
    return full, FEATURES


def split_tiles(df: pd.DataFrame):
    """Split by TILE (quadkey), the equivalent of the KL model's per-session
    split — a tile's full quarterly history stays together on one side."""
    tiles = sorted(df["quadkey"].unique())
    rng = random.Random(SEED)
    rng.shuffle(tiles)
    n = len(tiles)
    a = max(1, int(n * .70)); b = max(a + 1, int(n * .85))
    return tiles[:a], tiles[a:b], tiles[b:]


def make_sequences(data, tile_ids, features, lookback=LOOKBACK):
    X, y, meta = [], [], []
    for qk, g in data[data["quadkey"].isin(tile_ids)].groupby("quadkey"):
        g = g.sort_values("q_index").reset_index(drop=True)
        xv = g[features].to_numpy(np.float32)
        yv = g["Target_scaled"].to_numpy(np.float32)
        for i in range(lookback, len(g)):
            X.append(xv[i - lookback:i]); y.append(yv[i])
            meta.append({"quadkey": qk, "q_index": g.loc[i, "q_index"],
                        "Actual": g.loc[i, TARGET],
                        "Naive": g.loc[i - 1, TARGET]})  # persistence baseline
    return np.asarray(X, np.float32), np.asarray(y, np.float32), pd.DataFrame(meta)


def main():
    df, features = load_and_clean()
    print(f"Rows after cleaning: {len(df)}  |  features: {features}")

    train_tiles, val_tiles, test_tiles = split_tiles(df)
    print(f"Train tiles: {len(train_tiles)}  Val: {len(val_tiles)}  Test: {len(test_tiles)}")

    train_mask = df["quadkey"].isin(train_tiles)
    feature_scaler = StandardScaler().fit(df.loc[train_mask, features])
    target_scaler = StandardScaler().fit(df.loc[train_mask, [TARGET]])

    ds = df.copy()
    ds[features] = feature_scaler.transform(df[features])
    ds["Target_scaled"] = target_scaler.transform(df[[TARGET]]).ravel()

    X_train, y_train, _ = make_sequences(ds, train_tiles, features)
    X_val, y_val, _ = make_sequences(ds, val_tiles, features)
    X_test, y_test, m_test = make_sequences(ds, test_tiles, features)
    print(f"X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    model = Sequential([
        Input(shape=(LOOKBACK, len(features))),
        LSTM(32, return_sequences=True),
        Dropout(.2),
        LSTM(16),
        Dropout(.2),
        Dense(8, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=50, batch_size=64, callbacks=[early], verbose=1)

    plt.figure(figsize=(9, 4))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("LSTM Training History (Penang, Ookla quarterly)")
    plt.legend(); plt.grid(True)
    plt.savefig(HISTORY_PLOT, dpi=120, bbox_inches="tight")
    print(f"Saved {HISTORY_PLOT}")

    pred_scaled = model.predict(X_test, verbose=0)
    pred = target_scaler.inverse_transform(pred_scaled).ravel()
    actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    print(f"Test MAE:  {mae:,.0f} kbps")
    print(f"Test RMSE: {rmse:,.0f} kbps")
    print(f"Test R^2:  {r2:.3f}")

    # --- Naive persistence baseline (mentor #6): next quarter = last quarter ---
    # The yardstick the model must beat. Same unseen test samples.
    naive = m_test["Naive"].to_numpy(float)
    n_mae = mean_absolute_error(actual, naive)
    n_rmse = float(np.sqrt(mean_squared_error(actual, naive)))
    n_r2 = r2_score(actual, naive)
    print(f"Naive MAE: {n_mae:,.0f} kbps   RMSE: {n_rmse:,.0f}   R^2: {n_r2:.3f}")
    print(f"LSTM vs naive — MAE {(n_mae - mae) / n_mae * 100:+.1f}%   "
          f"RMSE {(n_rmse - rmse) / n_rmse * 100:+.1f}%")

    metrics = {
        "target": "avg_d_kbps — next-quarter average mobile download (kbps)",
        "test_samples": int(actual.size),
        "lstm": {"mae_kbps": round(float(mae)), "rmse_kbps": round(rmse),
                 "r2": round(float(r2), 3), "source": "this training run"},
        "naive_persistence": {"mae_kbps": round(float(n_mae)), "rmse_kbps": round(n_rmse),
                              "r2": round(float(n_r2), 3)},
        "improvement_vs_naive": {"mae_pct": round((n_mae - mae) / n_mae * 100, 1),
                                 "rmse_pct": round((n_rmse - rmse) / n_rmse * 100, 1)},
        "note": ("Naive baseline = persistence (next quarter = last observed quarter). "
                 "Positive improvement % = LSTM lower error than naive."),
    }
    (Path(__file__).resolve().parent / "lstm_penang_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")

    model.save(MODEL_OUT)
    print(f"Saved {MODEL_OUT}")


if __name__ == "__main__":
    main()
