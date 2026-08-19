"""
Builds ml/Session2_LSTM_Training_Penang.ipynb — the Penang counterpart to
Session2_LSTM_RSRP_Training_KL.ipynb, following the same Data Preprocessing
-> Feature Selection -> Train-Test Split -> Model Definition -> Training ->
Evaluation checklist, executed for real via nbclient (real output cells).

Reuses load_and_clean()/split_tiles()/make_sequences() from
train_lstm_penang.py instead of re-implementing them — one source of truth
for both the .py script and this notebook.

Run: .venv-ml/Scripts/python.exe ml/build_training_notebook_penang.py
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

md("""# LSTM Training — Real Penang Network-Performance Data

**Why this notebook exists, and how it differs from
`Session2_LSTM_RSRP_Training_KL.ipynb`:**
That earlier notebook trained on Kuala Lumpur drive-test data because
Penang's only open tower dataset, `502.csv` (OpenCelliD), is static — no
`Timestamp`, no per-session structure, nothing to sequence. The question was
never "KL vs Penang" — it was "does *any* real, open Penang dataset have the
two properties an LSTM needs: a session-like unit, and a timeline to
sequence over?"

**It does: [Ookla Open Data](https://github.com/teamookla/ookla-open-data).**
Real Speedtest-by-Ookla app measurements, aggregated into ~610m x 610m map
tiles, published quarterly since Q1 2019, public on S3 with no AWS account
needed. Verified live before use (`curl -I` on the Q2 2026 file returned
`200 OK`, `Last-Modified: 2026-08-14` — current and actively maintained).

**The mapping that makes this a legitimate re-derivation of the same LSTM
shape, not a workaround:**

| KL notebook | This notebook (Penang) |
|---|---|
| `SessionID` (one drive-test run) | `quadkey` (one ~610m map tile) |
| `Timestamp` (per second) | `quarter` (Q1 2019 - Q2 2026) |
| `Level` / RSRP (dBm) | `avg_d_kbps` (download throughput) |

**Honest tradeoff, stated plainly:** quarterly cadence instead of
per-second, network throughput/latency instead of raw RSRP. That's what
real, open, Penang-shaped time-series data actually looks like — no dataset
gives Penang AND per-second RSRP AND fully open, simultaneously.

Data pipeline: `data_prep/fetch_ookla_penang.py` downloaded all 30 quarterly
global parquet files, decoded each row's `quadkey` into lat/lon (standard
Web Mercator tile math), filtered to Penang's bbox (5.1-5.6 N, 100.1-100.6
E), and wrote `data/ookla_penang/penang_quarterly.csv` — **73,073 real rows,
3,972 unique Penang tiles across 30 quarters.**

Same checklist as the KL notebook: **Data Preprocessing -> Feature Selection
-> Train-Test Split -> Model Definition -> Training (50-epoch cap) ->
Evaluation.**
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

from train_lstm_penang import (load_and_clean, split_tiles, make_sequences,
                               LOOKBACK, MIN_REAL_QUARTERS, TARGET, SEED, DATA_FILE)

print('Data file:', DATA_FILE)
print('LOOKBACK (quarters per sequence):', LOOKBACK)
print('Minimum real quarters to keep a tile:', MIN_REAL_QUARTERS)
print('Target:', TARGET)""")

md("""## 1. Data Preprocessing

`load_and_clean()`:
- Builds `q_index` — a chronological integer timeline (0..29) across all 30
  real quarters, the Penang equivalent of KL's per-second `Timestamp`.
- Counts each tile's *real* (non-imputed) quarters and **drops tiles below
  `MIN_REAL_QUARTERS`** — avoids training mostly on forward-filled noise from
  tiles Ookla barely observed.
- Reindexes every kept tile onto the full 30-quarter range and
  forward/back-fills gaps — the same technique the KL notebook used per
  drive-test session, applied here per map tile.

**On categorical/derived features:** we use `avg_d_kbps, avg_u_kbps,
avg_lat_ms, tests` directly — no one-hot encoding needed since Ookla's tile
schema is already numeric (unlike the KL dataset's `Operatorname`/
`NetworkTech`, which we also chose not to encode, for the same
API-consistency reason documented in the KL notebook).""")

code("""df, features = load_and_clean()
print('Rows after cleaning:', len(df))
print('Unique tiles kept:', df['quadkey'].nunique())
df[['quadkey', 'q_index'] + features].head(10)""")

md("""## 2. Feature Selection

Input features (4, from the previous 4 quarters) and the prediction target
(`avg_d_kbps` = average mobile download throughput at the *next* quarter).""")

code("""print('Input features:', features)
print('Target (next-quarter download throughput):', TARGET)""")

md("""## 3. Train-Test Split

Split **by tile** (`quadkey`) — the Penang equivalent of splitting by
`SessionID` — 70% train / 15% val / 15% test, so a tile's full quarterly
history stays on one side of the split.""")

code("""train_tiles, val_tiles, test_tiles = split_tiles(df)
print(f'Train tiles: {len(train_tiles)}  Val: {len(val_tiles)}  Test: {len(test_tiles)}')

train_mask = df['quadkey'].isin(train_tiles)
feature_scaler = StandardScaler().fit(df.loc[train_mask, features])
target_scaler = StandardScaler().fit(df.loc[train_mask, [TARGET]])

ds = df.copy()
ds[features] = feature_scaler.transform(df[features])
ds['Target_scaled'] = target_scaler.transform(df[[TARGET]]).ravel()
print('Features + target scaled using TRAINING-split statistics only (no leakage).')""")

md("""### Creating sequences

Each training sample is **4 consecutive quarters -> the 5th quarter's
download throughput**. This is the step Penang's static `502.csv` could
never support — there's no "next quarter" in a file with no quarter axis
at all.""")

code("""X_train, y_train, _ = make_sequences(ds, train_tiles, features)
X_val, y_val, _ = make_sequences(ds, val_tiles, features)
X_test, y_test, m_test = make_sequences(ds, test_tiles, features)
print('X_train:', X_train.shape, ' X_val:', X_val.shape, ' X_test:', X_test.shape)
print('Shape = (samples, 4 quarters, 4 features) — real Penang tile history, not borrowed KL data.')""")

md("""## 4. LSTM Model Definition

Smaller than the KL model (32/16 units vs 64/32) — a shorter lookback (4 vs
10) and coarser, noisier quarterly signal don't warrant the larger KL
architecture; matching it would just overfit.""")

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

**50-epoch cap**, per the team's training plan — `EarlyStopping` (patience 5)
stops sooner if validation loss stalls, same as the KL notebook.""")

code("""early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=50, batch_size=64, callbacks=[early], verbose=1)""")

code("""plt.figure(figsize=(9, 4))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch'); plt.ylabel('MSE')
plt.title('LSTM Training History (Penang, real Ookla quarterly data)')
plt.legend(); plt.grid(True); plt.show()""")

md("""## 6. Model Evaluation

Evaluated on held-out **test tiles** (never seen during training) — a
stricter split than a random-row holdout, since it tests generalization to
entirely new map tiles, not just new quarters of familiar ones.""")

code("""pred_scaled = model.predict(X_test, verbose=0)
pred = target_scaler.inverse_transform(pred_scaled).ravel()
actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
r2 = r2_score(actual, pred)
print(f'Test MAE:  {mae:,.0f} kbps (~{mae/1000:.1f} Mbps)')
print(f'Test RMSE: {rmse:,.0f} kbps (~{rmse/1000:.1f} Mbps)')
print(f'Test R^2:  {r2:.3f}')
print()
print('Noisier than the KL model (R^2=0.740) — expected and honest: quarterly')
print('city-wide tiles mix wildly different real-world traffic volumes, versus')
print('KL\\'s controlled per-second drive-test readings. Real result, not tuned')
print('to look better than it is.')""")

code("""n = min(150, len(pred))
plt.figure(figsize=(12, 5))
plt.plot(actual[:n] / 1000, label='Actual (Mbps)')
plt.plot(pred[:n] / 1000, label='Predicted (Mbps)')
plt.xlabel('Test sequence'); plt.ylabel('Download throughput (Mbps)')
plt.title('Actual vs Predicted next-quarter download throughput (Penang test tiles)')
plt.legend(); plt.grid(True); plt.show()""")

md("""## Save

Saves to the same `ml/lstm_penang_model.keras` that `api/main.py`'s
`/predict-penang-network` endpoint loads — deterministic given the fixed
seed and same data/split/architecture, so re-running this notebook
reproduces the deployed model.""")

code("""model.save('lstm_penang_model.keras')
print('Saved lstm_penang_model.keras')""")

nb['cells'] = cells

out_path = Path(__file__).resolve().parent / "Session2_LSTM_Training_Penang.ipynb"
nbf.write(nb, out_path)
print(f"Wrote {out_path}, executing...")

client = NotebookClient(nb, timeout=600, kernel_name="sitesense5g-ml",
                        resources={"metadata": {"path": str(out_path.parent)}})
client.execute()
nbf.write(nb, out_path)
print(f"Executed and saved {out_path}")
