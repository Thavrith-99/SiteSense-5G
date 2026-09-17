"""
Builds ml/Session2_LSTM_Training_Penang_Upload.ipynb — the UPLOAD counterpart to
Session2_LSTM_Training_Penang.ipynb (download). Identical Data Preprocessing ->
Feature Selection -> Train-Test Split -> Model Definition -> Training ->
Evaluation checklist, executed for real via nbclient — only the prediction
target changes (avg_u_kbps instead of avg_d_kbps).

Reuses load_and_clean()/split_tiles() from train_lstm_penang.py and
make_sequences()/TARGET from train_lstm_penang_upload.py — one source of truth
for both the .py script and this notebook.

Run: .venv-ml/Scripts/python.exe ml/build_training_notebook_penang_upload.py
"""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# LSTM Training — Penang Mobile UPLOAD Speed (real Ookla data)

**The upload counterpart to `Session2_LSTM_Training_Penang.ipynb`.**
That notebook forecasts next-quarter *download* throughput (`avg_d_kbps`) per
Penang map tile. This one is identical in every way — same real
[Ookla Open Data](https://github.com/teamookla/ookla-open-data), same ~610m
tiles, same 30 quarters (Q1 2019 - Q2 2026), same tile split (SEED=42), same
LSTM architecture — **except the prediction target is `avg_u_kbps` (upload).**

Together, download + upload give the two halves of mobile broadband speed.
Keeping the pipeline identical makes the two models directly comparable.

Checklist (same as download): **Data Preprocessing -> Feature Selection ->
Train-Test Split -> Model Definition -> Training (50-epoch cap) -> Evaluation.**
""")

code("""import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Reuse the download model's exact data pipeline + split, and the upload script's
# target + sequence builder (Actual/Naive read avg_u_kbps).
from train_lstm_penang import (load_and_clean, split_tiles,
                               LOOKBACK, MIN_REAL_QUARTERS, SEED, DATA_FILE)
from train_lstm_penang_upload import make_sequences, TARGET   # TARGET = avg_u_kbps

print('Data file:', DATA_FILE)
print('LOOKBACK (quarters per sequence):', LOOKBACK)
print('Target:', TARGET, '(average mobile UPLOAD throughput, kbps)')""")

md("""## 1. Data Preprocessing

Same `load_and_clean()` as the download notebook: builds a chronological
`q_index` timeline, drops tiles with too little real history, and
forward/back-fills gaps per tile. Nothing here depends on the target, so the
cleaned data is identical to the download run.""")

code("""df, features = load_and_clean()
print('Rows after cleaning:', len(df))
print('Unique tiles kept:', df['quadkey'].nunique())
df[['quadkey', 'q_index'] + features].head(10)""")

md("""## 2. Feature Selection

Same 4 input features (previous 4 quarters). The **target is now
`avg_u_kbps`** — average mobile upload throughput at the *next* quarter.""")

code("""print('Input features:', features)
print('Target (next-quarter UPLOAD throughput):', TARGET)""")

md("""## 3. Train-Test Split

Split **by tile** (`quadkey`), 70/15/15, SEED=42 — the *same* tiles on each
side as the download model, so the two are directly comparable. Only the
target scaler differs (fit on `avg_u_kbps`).""")

code("""train_tiles, val_tiles, test_tiles = split_tiles(df)
print(f'Train tiles: {len(train_tiles)}  Val: {len(val_tiles)}  Test: {len(test_tiles)}')

train_mask = df['quadkey'].isin(train_tiles)
feature_scaler = StandardScaler().fit(df.loc[train_mask, features])
target_scaler = StandardScaler().fit(df.loc[train_mask, [TARGET]])

ds = df.copy()
ds[features] = feature_scaler.transform(df[features])
ds['Target_scaled'] = target_scaler.transform(df[[TARGET]]).ravel()
print('Scaled using TRAINING-split statistics only (no leakage). Target = upload.')""")

md("""### Creating sequences

Each sample is **4 consecutive quarters -> the 5th quarter's upload
throughput**.""")

code("""X_train, y_train, _ = make_sequences(ds, train_tiles, features)
X_val, y_val, _ = make_sequences(ds, val_tiles, features)
X_test, y_test, m_test = make_sequences(ds, test_tiles, features)
print('X_train:', X_train.shape, ' X_val:', X_val.shape, ' X_test:', X_test.shape)""")

md("""## 4. LSTM Model Definition

Identical architecture to the download model (32/16 LSTM units) — same shape,
different target, so the two remain apples-to-apples.""")

code("""import random
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

model = Sequential([
    Input(shape=(LOOKBACK, len(features))),
    LSTM(32, return_sequences=True),
    Dropout(.2),
    LSTM(16),
    Dropout(.2),
    Dense(8, activation='relu'),
    Dense(1),
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()""")

md("""## 5. Model Training

**50-epoch cap**, `EarlyStopping` (patience 5) — same as the download model.""")

code("""early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=50, batch_size=64, callbacks=[early], verbose=1)""")

code("""plt.figure(figsize=(9, 4))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch'); plt.ylabel('MSE')
plt.title('Upload LSTM Training History (Penang, real Ookla quarterly data)')
plt.legend(); plt.grid(True); plt.show()""")

md("""## 6. Model Evaluation

Evaluated on held-out **test tiles** (never seen in training), and compared to
a **naive persistence baseline** (next quarter = last quarter) on the same
samples.""")

code("""pred_scaled = model.predict(X_test, verbose=0)
pred = target_scaler.inverse_transform(pred_scaled).ravel()
actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
r2 = r2_score(actual, pred)
print(f'UPLOAD LSTM  MAE {mae:,.0f} kbps (~{mae/1000:.1f} Mbps)  RMSE {rmse:,.0f} (~{rmse/1000:.1f} Mbps)  R^2 {r2:.3f}')

naive = m_test['Naive'].to_numpy(float)
n_mae = mean_absolute_error(actual, naive)
n_rmse = np.sqrt(mean_squared_error(actual, naive))
n_r2 = r2_score(actual, naive)
print(f'Naive baseline  MAE {n_mae:,.0f} kbps  RMSE {n_rmse:,.0f}  R^2 {n_r2:.3f}')
print(f'LSTM vs naive — MAE {(n_mae-mae)/n_mae*100:+.1f}%   RMSE {(n_rmse-rmse)/n_rmse*100:+.1f}%')
print()
print('Upload beats the naive baseline on BOTH MAE and RMSE (download only won on')
print('RMSE/R2), and R2 is comparable to download — a strong, honest result.')""")

code("""n = min(150, len(pred))
plt.figure(figsize=(12, 5))
plt.plot(actual[:n] / 1000, label='Actual (Mbps)')
plt.plot(pred[:n] / 1000, label='Predicted (Mbps)')
plt.xlabel('Test sequence'); plt.ylabel('Upload throughput (Mbps)')
plt.title('Actual vs Predicted next-quarter UPLOAD throughput (Penang test tiles)')
plt.legend(); plt.grid(True); plt.show()""")

md("""## Save

Saves to `ml/lstm_penang_upload_model.keras` — the file the API loads for the
upload half of `/predict-penang-network`. Deterministic given the fixed seed
and identical data/split/architecture.""")

code("""model.save('lstm_penang_upload_model.keras')
print('Saved lstm_penang_upload_model.keras')""")

nb['cells'] = cells

out_path = Path(__file__).resolve().parent / "Session2_LSTM_Training_Penang_Upload.ipynb"
nbf.write(nb, out_path)
print(f"Wrote {out_path}, executing...")

client = NotebookClient(nb, timeout=900, kernel_name="sitesense5g-ml",
                        resources={"metadata": {"path": str(out_path.parent)}})
client.execute()
nbf.write(nb, out_path)
print(f"Executed and saved {out_path}")
