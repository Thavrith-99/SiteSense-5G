"""
Train the LSTM-RSRP model — SiteSense 5G Physical Finale.

Direct adaptation of the bootcamp's
`Bootcamp 2 - Advanced GeoAI-Day3/Module 7_AD1002_Session 2_Location v1/
Session_2_LSTM_RSRP_GeoAI_With_Location.ipynb`, pointed at our staged
`data/signal_kl/raw_dataset_kl.csv` instead of the notebook's own copy of
the same file. Column names, LOOKBACK, feature list, split logic, and model
architecture are kept IDENTICAL to the notebook on purpose — Session 4's
deployment code (api/main.py's /predict-rsrp) re-derives the same scalers
by re-running this exact preprocessing at API startup, so training and
serving must stay in lockstep.

IMPORTANT — honesty caveat (do not drop from the pitch/deck): this dataset
is Kuala Lumpur drive-test data, not Penang. Our coverage-gap and
site-recommendation logic (Functions 1/3) run on Penang OpenCelliD data.
This model is therefore a transfer-learned proof-of-concept demonstrating
the deployable ML pipeline on real Malaysian signal data — not a model
fitted on Penang-specific observations. Say "trained on real Malaysian
drive-test data (KL)", never "predicts Penang RSRP".

Run (from SiteSense5G_App/, using the isolated ML venv):
    .venv-ml/Scripts/python.exe ml/train_lstm_rsrp.py

Outputs:
    ml/lstm_rsrp_model.keras
    ml/training_history.png
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — no display available when run as a script
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
LOOKBACK = 10
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

APP = Path(__file__).resolve().parent.parent
DATA_FILE = APP / "data" / "signal_kl" / "raw_dataset_kl.csv"
MODEL_OUT = Path(__file__).resolve().parent / "lstm_rsrp_model.keras"
HISTORY_PLOT = Path(__file__).resolve().parent / "training_history.png"

REQUIRED = ["Timestamp", "SessionID", "Level", "Latitude", "Longitude"]
FEATURE_CANDIDATES = ["Level", "SNR", "CQI", "DL_bitrate", "UL_bitrate",
                      "Speed", "Latitude", "Longitude"]
TARGET = "Level"


def load_and_clean() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y.%m.%d_%H.%M.%S", errors="coerce")
    features = [c for c in FEATURE_CANDIDATES if c in df.columns]

    df = df.dropna(subset=REQUIRED).copy()
    df = df.sort_values(["SessionID", "Timestamp"]).reset_index(drop=True)
    df[features] = df.groupby("SessionID")[features].transform(lambda x: x.ffill().bfill())
    df[features] = df[features].fillna(df[features].median(numeric_only=True))
    return df, features


def split_sessions(df: pd.DataFrame):
    sessions = df.groupby("SessionID")["Timestamp"].min().sort_values().index.tolist()
    if len(sessions) < 3:
        raise ValueError("At least 3 sessions are required for train/val/test splitting.")
    n = len(sessions)
    a = max(1, int(n * .70)); b = max(a + 1, int(n * .85))
    return sessions[:a], sessions[a:b], sessions[b:]


def make_sequences(data, session_ids, features, lookback=LOOKBACK):
    X, y, meta = [], [], []
    for sid, g in data[data["SessionID"].isin(session_ids)].groupby("SessionID"):
        g = g.sort_values("Timestamp").reset_index(drop=True)
        xv = g[features].to_numpy(np.float32)
        yv = g["Target_scaled"].to_numpy(np.float32)
        for i in range(lookback, len(g)):
            X.append(xv[i - lookback:i]); y.append(yv[i])
            meta.append({"SessionID": sid, "Timestamp": g.loc[i, "Timestamp"],
                        "Latitude": g.loc[i, "Map_Latitude"], "Longitude": g.loc[i, "Map_Longitude"],
                        "Actual_RSRP": g.loc[i, TARGET]})
    return np.asarray(X, np.float32), np.asarray(y, np.float32), pd.DataFrame(meta)


def main():
    df, features = load_and_clean()
    print(f"Rows after cleaning: {len(df)}  |  features: {features}")

    train_sessions, val_sessions, test_sessions = split_sessions(df)
    print(f"Train sessions: {len(train_sessions)}  Val: {len(val_sessions)}  Test: {len(test_sessions)}")

    train_mask = df["SessionID"].isin(train_sessions)
    feature_scaler = StandardScaler().fit(df.loc[train_mask, features])
    target_scaler = StandardScaler().fit(df.loc[train_mask, [TARGET]])

    ds = df.copy()
    ds["Map_Latitude"] = ds["Latitude"]
    ds["Map_Longitude"] = ds["Longitude"]
    ds[features] = feature_scaler.transform(df[features])
    ds["Target_scaled"] = target_scaler.transform(df[[TARGET]]).ravel()

    X_train, y_train, _ = make_sequences(ds, train_sessions, features)
    X_val, y_val, _ = make_sequences(ds, val_sessions, features)
    X_test, y_test, m_test = make_sequences(ds, test_sessions, features)
    print(f"X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    model = Sequential([
        Input(shape=(LOOKBACK, len(features))),
        LSTM(64, return_sequences=True),
        Dropout(.2),
        LSTM(32),
        Dropout(.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=30, batch_size=64, callbacks=[early], verbose=1)

    plt.figure(figsize=(9, 4))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("LSTM Training History (KL drive-test data)")
    plt.legend(); plt.grid(True)
    plt.savefig(HISTORY_PLOT, dpi=120, bbox_inches="tight")
    print(f"Saved {HISTORY_PLOT}")

    pred_scaled = model.predict(X_test, verbose=0)
    pred = target_scaler.inverse_transform(pred_scaled).ravel()
    actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    print(f"Test MAE:  {mae:.3f} dB")
    print(f"Test RMSE: {rmse:.3f} dB")
    print(f"Test R^2:  {r2:.3f}")

    model.save(MODEL_OUT)
    print(f"Saved {MODEL_OUT}")


if __name__ == "__main__":
    main()
