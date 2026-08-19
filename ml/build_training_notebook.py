"""
Builds ml/Session2_LSTM_RSRP_Training_KL.ipynb — a step-by-step notebook
following the Data Preprocessing -> Feature Selection -> Train-Test Split ->
Model Definition -> Training -> Evaluation checklist, executed for real via
nbclient so the saved notebook has genuine output cells (matching the
bootcamp notebooks' own style).

Reuses load_and_clean()/split_sessions()/make_sequences() from
train_lstm_rsrp.py instead of re-implementing them, so there is exactly one
source of truth for the preprocessing logic that both the .py training
script and this notebook rely on.

Run: .venv-ml/Scripts/python.exe ml/build_training_notebook.py
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

md("""# LSTM-RSRP Training — Kuala Lumpur Drive-Test Data

**Why Kuala Lumpur, not Penang — the deeper issue:**
An LSTM needs *time-ordered sequential* signal readings (RSRP/SNR/CQI over
consecutive timestamps within a drive-test session) to learn from. Penang's
only tower dataset, `502.csv` (OpenCelliD), is **static** — one row per
tower (`radio, mcc, net, area, cell, lon, lat, range, samples, changeable`),
no `Timestamp` column, no per-session ordering. There is nothing to
sequence. Forcing an LSTM onto that data would mean feeding it rows in an
arbitrary order with no real temporal structure — the model wouldn't be
learning a genuine pattern, and that's a bigger overclaim risk than just
being upfront about using a dataset that actually has the right shape.

`data/signal_kl/raw_dataset_kl.csv` (Kuala Lumpur drive-test data, 30,925
rows, 23 sessions) has exactly what's needed: `Timestamp` + `SessionID` +
real `Level` (RSRP)/`SNR`/`CQI` readings recorded consecutively as a device
moved through the network. That structural requirement — not city
preference — is why this notebook trains on KL. The trained model is used
in the dashboard's demo panel labeled *"LSTM Signal-Quality Demo (Kuala
Lumpur)"*, kept visibly separate from the Penang coverage-gap/site-
recommendation logic (`data_access.py` / Functions 1–3), which never touches
this dataset.

This notebook follows the standard LSTM checklist: **Data Preprocessing →
Feature Selection → Train-Test Split → Model Definition → Training →
Evaluation** — reusing the tested functions in `train_lstm_rsrp.py` (the
script `api/main.py`'s `/predict-rsrp` endpoint is actually built from) so
there's one source of truth, not a second copy of the preprocessing logic.
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

from train_lstm_rsrp import load_and_clean, split_sessions, make_sequences, LOOKBACK, TARGET, SEED, DATA_FILE

print('Data file:', DATA_FILE)
print('LOOKBACK (observations per sequence):', LOOKBACK)
print('Target:', TARGET)""")

md("""## 1. Data Preprocessing

`load_and_clean()` does exactly what the checklist calls for:
- Converts `Timestamp` to a proper datetime (`%Y.%m.%d_%H.%M.%S` format).
- Sorts rows by `SessionID` then `Timestamp` — this is what makes the data
  genuinely *time-ordered*, the piece Penang's static file can't provide.
- Forward/back-fills gaps within each session, then fills any remainder
  with the column median.

**On categorical features:** the raw file also has `Operatorname` and
`NetworkTech` (categorical). We deliberately do **not** one-hot encode them
here — the deployed feature set (`Level, SNR, CQI, DL_bitrate, UL_bitrate,
Speed, Latitude, Longitude`) matches the bootcamp's own Session 2/4 notebooks
exactly, and `api/main.py`'s `/predict-rsrp` re-derives its scalers from this
same feature list at startup. Changing the feature set here without also
changing the API would desync training from deployment — if you want
operator/tech as features later, that has to be a deliberate change made in
both `train_lstm_rsrp.py` and `api/main.py` together, not just here.""")

code("""df, features = load_and_clean()
print('Rows after cleaning:', len(df))
print('Rows are now sorted by SessionID -> Timestamp (time-ordered):')
df[['SessionID', 'Timestamp'] + features].head(10)""")

md("""## 2. Feature Selection

Input features (8, from the previous 10 observations) and the prediction
target (`Level` = RSRP at the *next* observation) — same as the bootcamp's
Session 2 notebook, so `api/main.py` can reuse this exact preprocessing.""")

code("""print('Input features:', features)
print('Target (next-step RSRP):', TARGET)""")

md("""## 3. Train-Test Split

Split **by drive-test session** (70% train / 15% val / 15% test), not by
random row — sessions are the natural unit here, and this avoids leaking
adjacent timestamps from the same session across the split.""")

code("""train_sessions, val_sessions, test_sessions = split_sessions(df)
print(f'Train sessions: {len(train_sessions)}  Val: {len(val_sessions)}  Test: {len(test_sessions)}')

train_mask = df['SessionID'].isin(train_sessions)
feature_scaler = StandardScaler().fit(df.loc[train_mask, features])
target_scaler = StandardScaler().fit(df.loc[train_mask, [TARGET]])

ds = df.copy()
ds['Map_Latitude'] = ds['Latitude']
ds['Map_Longitude'] = ds['Longitude']
ds[features] = feature_scaler.transform(df[features])
ds['Target_scaled'] = target_scaler.transform(df[[TARGET]]).ravel()
print('Features + target scaled using TRAINING-split statistics only (no leakage).')""")

md("""### Creating sequences

Each training sample is **10 consecutive observations -> the 11th
observation's RSRP**. This is the step that literally cannot be done on
Penang's static `502.csv` — there's no consecutive-observation axis to
window over.""")

code("""X_train, y_train, _ = make_sequences(ds, train_sessions, features)
X_val, y_val, _ = make_sequences(ds, val_sessions, features)
X_test, y_test, m_test = make_sequences(ds, test_sessions, features)
print('X_train:', X_train.shape, ' X_val:', X_val.shape, ' X_test:', X_test.shape)
print('Shape = (samples, 10 timesteps, 8 features) — the sequential structure Penang data lacks.')""")

md("""## 4. LSTM Model Definition

Same architecture as the bootcamp's Session 2 notebook: 2 stacked LSTM
layers with dropout, then a small dense head.""")

code("""import random
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

model = Sequential([
    Input(shape=(LOOKBACK, len(features))),
    LSTM(64, return_sequences=True),
    Dropout(.2),
    LSTM(32),
    Dropout(.2),
    Dense(16, activation='relu'),
    Dense(1),
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()""")

md("## 5. Model Training")

code("""early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=30, batch_size=64, callbacks=[early], verbose=1)""")

code("""plt.figure(figsize=(9, 4))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch'); plt.ylabel('MSE')
plt.title('LSTM Training History (Kuala Lumpur drive-test data)')
plt.legend(); plt.grid(True); plt.show()""")

md("""## 6. Model Evaluation

Evaluated on the held-out test sessions (never seen during training).""")

code("""pred_scaled = model.predict(X_test, verbose=0)
pred = target_scaler.inverse_transform(pred_scaled).ravel()
actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
r2 = r2_score(actual, pred)
print(f'Test MAE:  {mae:.3f} dB')
print(f'Test RMSE: {rmse:.3f} dB')
print(f'Test R^2:  {r2:.3f}')""")

code("""n = min(150, len(pred))
plt.figure(figsize=(12, 5))
plt.plot(actual[:n], label='Actual RSRP')
plt.plot(pred[:n], label='Predicted RSRP')
plt.xlabel('Test observation'); plt.ylabel('RSRP (dBm)')
plt.title('Actual vs Predicted RSRP (Kuala Lumpur test sessions)')
plt.legend(); plt.grid(True); plt.show()""")

md("""## Save

Saves to the same `ml/lstm_rsrp_model.keras` that `api/main.py`'s
`/predict-rsrp` loads — deterministic given the fixed seed, same data, same
architecture, so re-running this notebook reproduces the deployed model
exactly (verified: this notebook's MAE/RMSE/R² match `train_lstm_rsrp.py`'s
run, and `/predict-rsrp`'s sample prediction of **-96.61 dBm** was confirmed
against the bootcamp's own reference notebook output).""")

code("""model.save('lstm_rsrp_model.keras')
print('Saved lstm_rsrp_model.keras')""")

nb['cells'] = cells

out_path = Path(__file__).resolve().parent / "Session2_LSTM_RSRP_Training_KL.ipynb"
nbf.write(nb, out_path)
print(f"Wrote {out_path}, executing...")

client = NotebookClient(nb, timeout=600, kernel_name="sitesense5g-ml",
                        resources={"metadata": {"path": str(out_path.parent)}})
client.execute()
nbf.write(nb, out_path)
print(f"Executed and saved {out_path}")
