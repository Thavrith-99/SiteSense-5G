"""
Train the Penang UPLOAD-speed LSTM (avg_u_kbps), mirroring train_lstm_penang.py
(which targets download, avg_d_kbps). Same data, features, LSTM architecture and
tile split (SEED=42) — ONLY the target column and the output filenames differ.
This keeps download and upload perfectly comparable.

Run (isolated ML venv):
    .venv-ml/Scripts/python.exe ml/train_lstm_penang_upload.py
Outputs:
    ml/lstm_penang_upload_model.keras
    ml/lstm_penang_upload_metrics.json
    ml/penang_upload_training_history.png
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

# Reuse the download model's exact data pipeline + split so the two are identical.
from train_lstm_penang import load_and_clean, split_tiles, FEATURES, LOOKBACK, SEED

random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

TARGET = "avg_u_kbps"  # upload (the only change vs the download model)
HERE = Path(__file__).resolve().parent
MODEL_OUT = HERE / "lstm_penang_upload_model.keras"
HISTORY_PLOT = HERE / "penang_upload_training_history.png"
METRICS_OUT = HERE / "lstm_penang_upload_metrics.json"


def make_sequences(data, tile_ids, features, lookback=LOOKBACK):
    """Same as the download script's, but the Actual/Naive baseline read the
    UPLOAD target column."""
    X, y, meta = [], [], []
    for qk, g in data[data["quadkey"].isin(tile_ids)].groupby("quadkey"):
        g = g.sort_values("q_index").reset_index(drop=True)
        xv = g[features].to_numpy(np.float32)
        yv = g["Target_scaled"].to_numpy(np.float32)
        for i in range(lookback, len(g)):
            X.append(xv[i - lookback:i]); y.append(yv[i])
            meta.append({"quadkey": qk, "q_index": g.loc[i, "q_index"],
                         "Actual": g.loc[i, TARGET],
                         "Naive": g.loc[i - 1, TARGET]})
    return np.asarray(X, np.float32), np.asarray(y, np.float32), pd.DataFrame(meta)


def main():
    df, features = load_and_clean()
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

    early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=50, batch_size=64, callbacks=[early], verbose=2)

    plt.figure(figsize=(9, 4))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.title("Upload LSTM Training History (Penang, Ookla quarterly)")
    plt.legend(); plt.grid(True)
    plt.savefig(HISTORY_PLOT, dpi=120, bbox_inches="tight")

    pred = target_scaler.inverse_transform(model.predict(X_test, verbose=0)).ravel()
    actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    mae = mean_absolute_error(actual, pred)
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    r2 = r2_score(actual, pred)

    naive = m_test["Naive"].to_numpy(float)
    n_mae = mean_absolute_error(actual, naive)
    n_rmse = float(np.sqrt(mean_squared_error(actual, naive)))
    n_r2 = r2_score(actual, naive)

    print(f"UPLOAD LSTM   MAE {mae:,.0f} kbps  RMSE {rmse:,.0f}  R2 {r2:.3f}")
    print(f"UPLOAD naive  MAE {n_mae:,.0f} kbps  RMSE {n_rmse:,.0f}  R2 {n_r2:.3f}")
    print(f"LSTM vs naive — MAE {(n_mae - mae) / n_mae * 100:+.1f}%   "
          f"RMSE {(n_rmse - rmse) / n_rmse * 100:+.1f}%")

    metrics = {
        "target": "avg_u_kbps — next-quarter average mobile upload (kbps)",
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
    METRICS_OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    model.save(MODEL_OUT)
    print(f"Saved {MODEL_OUT}")
    print(f"Saved {METRICS_OUT}")


if __name__ == "__main__":
    main()
