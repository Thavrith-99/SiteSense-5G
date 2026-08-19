"""
Fetch real, open, Penang-specific mobile network-performance history from
Ookla's Open Data (https://github.com/teamookla/ookla-open-data) — quarterly
tile aggregates (avg download/upload speed, latency) from real Speedtest
app measurements, publicly hosted on S3, no account/credentials needed.

Why this replaces the KL drive-test dataset for the "Penang LSTM": it is
Penang-specific (filtered by real tile coordinates decoded from each row's
quadkey) AND genuinely time-ordered (one row per tile per quarter, Q1 2019
onward) — the two properties raw_dataset_kl.csv could offer but only for
Kuala Lumpur. The granularity is coarser (quarterly, not per-second) and the
signal is network throughput/latency, not RSRP — that's an honest tradeoff,
not a workaround: this is what real, open, Penang-shaped time-series data
looks like.

Downloads one ~150-200MB global parquet per quarter, keeps only the columns
needed + rows inside Penang's bbox, deletes the temp file, moves to the next
quarter. Total download ~5GB over 30 quarters; disk usage stays low (~200MB
peak) since each file is deleted immediately after filtering.

Run (from SiteSense5G_App/, isolated ML venv — needs pyarrow):
    .venv-ml/Scripts/python.exe data_prep/fetch_ookla_penang.py
Output:
    data/ookla_penang/penang_quarterly.csv
"""
import math
import time
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

APP = Path(__file__).resolve().parent.parent
OUT_DIR = APP / "data" / "ookla_penang"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "penang_quarterly.csv"
TMP = OUT_DIR / "_tmp.parquet"

LAT_MIN, LAT_MAX = 5.1, 5.6
LON_MIN, LON_MAX = 100.1, 100.6

QUARTER_START = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}
COLUMNS = ["quadkey", "avg_d_kbps", "avg_u_kbps", "avg_lat_ms", "tests", "devices"]


def quarters(first_year=2019, first_q=1, last_year=2026, last_q=2):
    y, q = first_year, first_q
    while (y, q) <= (last_year, last_q):
        yield y, q
        q += 1
        if q > 4:
            q = 1
            y += 1


def quadkey_to_latlon(qk: str):
    x = y = 0
    z = len(qk)
    for i, ch in enumerate(qk):
        mask = 1 << (z - i - 1)
        if ch == "1":
            x |= mask
        elif ch == "2":
            y |= mask
        elif ch == "3":
            x |= mask
            y |= mask
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return math.degrees(lat_rad), lon


def fetch_quarter(year: int, q: int) -> pd.DataFrame:
    url = (
        f"https://ookla-open-data.s3.amazonaws.com/parquet/performance/"
        f"type=mobile/year={year}/quarter={q}/{year}-{QUARTER_START[q]}_performance_mobile_tiles.parquet"
    )
    urllib.request.urlretrieve(url, TMP)
    table = pq.read_table(TMP, columns=COLUMNS)
    df = table.to_pandas()
    TMP.unlink()

    df["lat"], df["lon"] = zip(*df["quadkey"].map(quadkey_to_latlon))
    penang = df[df["lat"].between(LAT_MIN, LAT_MAX) & df["lon"].between(LON_MIN, LON_MAX)].copy()
    penang["year"] = year
    penang["quarter"] = q
    return penang


def main():
    frames = []
    for year, q in quarters():
        t0 = time.time()
        try:
            pq_df = fetch_quarter(year, q)
        except Exception as e:
            print(f"{year} Q{q}: SKIPPED ({e})")
            continue
        frames.append(pq_df)
        print(f"{year} Q{q}: {len(pq_df):,} Penang tiles ({time.time()-t0:.0f}s)")

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} — {len(all_df):,} rows across {len(frames)} quarters")
    print(f"Unique Penang tiles ever seen: {all_df['quadkey'].nunique():,}")


if __name__ == "__main__":
    main()
