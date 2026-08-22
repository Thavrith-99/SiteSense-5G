#!/usr/bin/env python
"""SiteSense 5G — integration & regression pass over the data/MCDA layer.

Covers the combinations Roadmap §8.4 asks for: every scope x every MCDA
profile x edge-case filters, checked against the project's CANONICAL numbers
and against structural invariants (monotonic cumulative gain, eligibility
gate, cost/phase fields present and sane). Pure data_access + recommend —
no TensorFlow, runs on the default `py`. Prints a PASS/FAIL table and exits
non-zero if any check fails.

Run:  py tests/integration_regression.py
"""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_access as da
import recommend as rec

# --- canonical ground truth (default NR+LTE view) --------------------------
NRLTE = ("LTE", "NR")            # sorted picked_key the app passes
CANON = {
    "Penang State": {
        "towers_all": 4643, "cells_nrlte": 1746,
        "uncovered_pop": 77362, "pct_covered": 95.57, "uncovered_places": 31,
        "coverage_first_top5": 36788,
    },
    "Penang Island": {
        "towers_all": 2187, "cells_nrlte": 944,
        "uncovered_pop": 56597, "pct_covered": 92.87, "uncovered_places": 10,
        "coverage_first_top5": 35096,
    },
}
PROFILES = list(rec.WEIGHT_PROFILES.keys())

results = []   # (name, ok, detail)
def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))

def approx(a, b, tol):
    return abs(a - b) <= tol

print("=" * 74)
print("SiteSense 5G — Integration / Regression pass")
print(f"Data source: {'PostGIS' if da.using_db() else 'local files'}")
print("=" * 74)

# ===========================================================================
# 1. Per-scope regression against canonical numbers
# ===========================================================================
for scope, c in CANON.items():
    print(f"\n[{scope}] regression vs canonical")
    df = da.load_towers(scope)
    check(f"{scope}: total towers == {c['towers_all']}", len(df) == c["towers_all"], f"got {len(df)}")
    picked = tuple(sorted(r for r in ("NR", "LTE") if r in set(df["radio"])))
    fdf = df[df["radio"].isin(picked)]
    check(f"{scope}: NR+LTE cells == {c['cells_nrlte']}", len(fdf) == c["cells_nrlte"], f"got {len(fdf)}")

    max_range = int(df["range"].max())
    g = da.compute_gap(scope, picked, max_range)
    check(f"{scope}: uncovered_pop == {c['uncovered_pop']}",
          round(g["uncovered_pop"]) == c["uncovered_pop"], f"got {g['uncovered_pop']:.0f}")
    check(f"{scope}: pct_covered ~= {c['pct_covered']}",
          approx(g["pct_covered"], c["pct_covered"], 0.05), f"got {g['pct_covered']:.2f}")
    check(f"{scope}: uncovered_places == {c['uncovered_places']}",
          g["n_uncovered_places"] == c["uncovered_places"], f"got {g['n_uncovered_places']}")

    s = da.compute_sites(scope, picked, max_range, 90, 1000, 5, "Coverage-first")
    top5 = s[-1]["cumulative_gained"] if s else 0
    check(f"{scope}: Coverage-first top-5 == {c['coverage_first_top5']}",
          round(top5) == c["coverage_first_top5"], f"got {top5:.0f}")

# ===========================================================================
# 2. Every scope x profile — structural invariants
# ===========================================================================
for scope in CANON:
    df = da.load_towers(scope)
    picked = tuple(sorted(r for r in ("NR", "LTE") if r in set(df["radio"])))
    mr = int(df["range"].max())
    print(f"\n[{scope}] profile invariants")
    cov_first_top5 = None
    for prof in PROFILES:
        s = da.compute_sites(scope, picked, mr, 90, 1000, 5, prof)
        # a) returns sites
        check(f"{scope}/{prof}: returns >=1 site", len(s) >= 1, f"n={len(s)}")
        if not s:
            continue
        # b) eligibility gate: every returned site adds real coverage
        check(f"{scope}/{prof}: all people_gained > 0",
              all(x["people_gained"] > 0 for x in s),
              f"min={min(x['people_gained'] for x in s)}")
        # c) cumulative non-decreasing
        cums = [x["cumulative_gained"] for x in s]
        check(f"{scope}/{prof}: cumulative non-decreasing", cums == sorted(cums), str(cums))
        # d) cumulative == running sum of people_gained (no double count)
        run = 0; ok_sum = True
        for x in s:
            run += x["people_gained"]
            if round(run) != round(x["cumulative_gained"]):
                ok_sum = False; break
        check(f"{scope}/{prof}: cumulative == sum(gained)", ok_sum)
        # e) cost / phase fields present and sane
        okcost = all(
            x.get("est_capex_usd", 0) > 0
            and x.get("people_per_1000usd", 0) > 0
            and x.get("phase") in (1, 2, 3)
            for x in s)
        check(f"{scope}/{prof}: capex/phase/roi fields sane", okcost,
              "" if okcost else str(s[0].keys()))
        if prof == "Coverage-first":
            cov_first_top5 = cums[-1]
    # f) Coverage-first maximises coverage vs the other profiles
    for prof in PROFILES:
        if prof == "Coverage-first":
            continue
        s = da.compute_sites(scope, picked, mr, 90, 1000, 5, prof)
        top = s[-1]["cumulative_gained"] if s else 0
        check(f"{scope}: Coverage-first >= {prof} coverage",
              cov_first_top5 is not None and cov_first_top5 + 1 >= top,
              f"covfirst={cov_first_top5:.0f} vs {prof}={top:.0f}")

# ===========================================================================
# 3. Edge cases
# ===========================================================================
print("\n[edge cases]")
scope = "Penang State"
df = da.load_towers(scope); mr = int(df["range"].max())
picked = ("LTE", "NR")

# a) n_sites larger than available gap-closing candidates must not crash / not exceed request
s15 = da.compute_sites(scope, picked, mr, 90, 1000, 15, "Coverage-first")
check("n_sites=15 returns <=15 and monotonic", len(s15) <= 15 and
      [x["cumulative_gained"] for x in s15] == sorted(x["cumulative_gained"] for x in s15),
      f"n={len(s15)}")

# b) new tower range extremes
for rng in (300, 3000):
    s = da.compute_sites(scope, picked, mr, 90, rng, 5, "Balanced")
    check(f"new_range={rng} returns sites, all gain>0", len(s) >= 1 and all(x["people_gained"] > 0 for x in s), f"n={len(s)}")

# c) single-generation filter (LTE only) — gap must be >= NR+LTE gap (fewer towers = more gap)
g_lte = da.compute_gap(scope, ("LTE",), mr)
g_all = da.compute_gap(scope, picked, mr)
check("LTE-only gap >= NR+LTE gap", g_lte["uncovered_pop"] >= g_all["uncovered_pop"],
      f"lte={g_lte['uncovered_pop']:.0f} vs nrlte={g_all['uncovered_pop']:.0f}")

# d) load percentile sensitivity — higher pct => fewer/equal overloaded
def n_over(p):
    fdf = df[df["radio"].isin(picked)]
    thr = fdf["samples"].quantile(p/100.0)
    return int((fdf["samples"] >= thr).sum())
check("overloaded(95th) <= overloaded(90th)", n_over(95) <= n_over(90), f"90th={n_over(90)}, 95th={n_over(95)}")

# e) tighter max_range filter reduces (or equals) towers in view
fewer = df[(df["radio"].isin(picked)) & (df["range"] <= 2000)]
allr  = df[df["radio"].isin(picked)]
check("max_range=2000 filters towers", len(fewer) <= len(allr), f"{len(fewer)} <= {len(allr)}")

# ===========================================================================
# summary
# ===========================================================================
n = len(results); passed = sum(1 for _, ok, _ in results if ok)
print("\n" + "=" * 74)
print(f"RESULT: {passed}/{n} checks passed")
fails = [r for r in results if not r[1]]
if fails:
    print("\nFAILURES:")
    for name, _, detail in fails:
        print(f"  - {name}  ({detail})")
print("=" * 74)
sys.exit(0 if passed == n else 1)
