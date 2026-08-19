"""
Function 3 - recommend where to build the next 5G towers (SiteSense 5G).

Weighted multi-criteria site scoring (MCDA), per the Physical Finale
Improvement Roadmap items #1, #6, #7: greedy max-coverage siting generalised
into a weighted combination of

    score = w_population * (uncovered population reached)
          + w_backhaul   * (proximity to nearest EXISTING tower, a
                             backhaul-feasibility proxy — closer is cheaper
                             and easier to connect)
          + w_overload   * (nearby overloaded towers relieved)
          + w_slope      * (−slope penalty, real SRTM terrain — Roadmap #6)

still picked greedily, one site at a time, removing captured population so
later sites are complementary — that part of the algorithm is unchanged
from the original single-criterion version.

Three preset weighting profiles are provided (WEIGHT_PROFILES below); the
old behaviour is exactly the "Coverage-first" profile (population weight 1,
everything else 0) — min-max normalisation is a monotonic transform, so it
picks the identical sites in the identical order as the original
population-only greedy search. Weights are expert-judgement / team-set, not
learned from data — say so plainly, per the Roadmap's own honest-caveat note;
this is not a validated optimisation without real operator ground truth.
Slope is a simple buildability/accessibility proxy, not a full RF
propagation/viewshed model — say so too.

Roadmap #7 (cost/ROI) is layered on AFTER site selection, not as a 5th score
weight: each returned site gets an estimated capex figure
(BASE_CAPEX_USD + backhaul_distance_km * FIBER_COST_PER_KM_USD, both cited
industry benchmarks — see the constants below) and a 3-phase rollout
assignment by cost-efficiency (people reached per dollar). This mirrors the
Technical Upgrades sheet's own spec: "simple arithmetic on the
/recommend-sites output — no new modelling required." Capex figures are
industry benchmarks, not site-specific quotes — label them as
estimated/order-of-magnitude, consistent with the rest of the app's
"estimated/preliminary" wording.

Search uses a fast integral-image box filter to locate the best cell; the
reported gain and the coverage removal use an accurate circular footprint.
Metre-per-degree scaling is local to Penang (no pyproj).
"""
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

M_PER_DEG_LAT = 111_320.0

WEIGHT_PROFILES = {
    "Coverage-first": {"population": 1.00, "backhaul": 0.00, "overload": 0.00, "slope": 0.00},
    "Balanced":       {"population": 0.50, "backhaul": 0.20, "overload": 0.15, "slope": 0.15},
    "Cost-efficient": {"population": 0.30, "backhaul": 0.35, "overload": 0.10, "slope": 0.25},
}

# Cost/ROI benchmarks (Roadmap #7) — cited industry estimates, NOT
# site-specific quotes. Source: "5G Infrastructure Costs: What Telcos Are
# Paying," PatentPC (2026) — average 5G base station $100k-$200k (midpoint
# used below), fiber backhaul $25k-$100k/km (midpoint used below). State
# this source and the "order-of-magnitude estimate" caveat in the pitch.
BASE_CAPEX_USD = 150_000.0
FIBER_COST_PER_KM_USD = 62_500.0
COST_SOURCE = "PatentPC (2026), \"5G Infrastructure Costs: What Telcos Are Paying\" — industry benchmark, not a site-specific quote."


def _m_per_deg_lon(lat_deg: float) -> float:
    return 111_320.0 * np.cos(np.radians(lat_deg))


def _box_sum(arr: np.ndarray, r: int) -> np.ndarray:
    """Sum of arr over a (2r+1) square window centred on every cell."""
    ii = np.pad(arr, ((1, 0), (1, 0))).cumsum(0).cumsum(1)  # (H+1, W+1)
    H, W = arr.shape
    rr = np.arange(H); cc = np.arange(W)
    r0 = np.clip(rr - r, 0, H); r1 = np.clip(rr + r + 1, 0, H)
    c0 = np.clip(cc - r, 0, W); c1 = np.clip(cc + r + 1, 0, W)
    A = ii[r1][:, c1]; B = ii[r0][:, c1]; C = ii[r1][:, c0]; D = ii[r0][:, c0]
    return A - B - C + D


def _minmax_norm(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1] over the valid mask. Flat/degenerate
    input (max == min) normalises to all-zero rather than dividing by zero."""
    out = np.zeros_like(arr, dtype="float64")
    vals = arr[valid]
    if vals.size == 0:
        return out
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    if hi - lo < 1e-9:
        return out
    out[valid] = (arr[valid] - lo) / (hi - lo)
    return out


def _points_to_raster(shape, transform, lon, lat) -> np.ndarray:
    """Count of points falling in each pixel (H, W)."""
    H, W = shape
    counts = np.zeros((H, W), dtype="float64")
    if lon is None or len(lon) == 0:
        return counts
    inv = ~transform
    for x, y in zip(lon, lat):
        col, row = inv * (x, y)
        r, c = int(round(row)), int(round(col))
        if 0 <= r < H and 0 <= c < W:
            counts[r, c] += 1.0
    return counts


def _backhaul_distance_raster(shape, transform, lat0, tower_lon, tower_lat) -> np.ndarray:
    """Distance in metres from every pixel to the nearest EXISTING tower —
    a backhaul-feasibility proxy (closer to existing infrastructure is
    cheaper/easier to connect a new site to)."""
    H, W = shape
    if tower_lon is None or len(tower_lon) == 0:
        return np.full((H, W), np.inf)
    px = transform.a
    py = -transform.e
    mlon = _m_per_deg_lon(lat0)
    background = np.ones((H, W), dtype=bool)  # True = not a tower pixel
    inv = ~transform
    for x, y in zip(tower_lon, tower_lat):
        col, row = inv * (x, y)
        r, c = int(round(row)), int(round(col))
        if 0 <= r < H and 0 <= c < W:
            background[r, c] = False
    return distance_transform_edt(background, sampling=(py * M_PER_DEG_LAT, px * mlon))


def recommend_sites(pop, cov_mask, transform, lat0, new_range_m, n_sites,
                    over_lon=None, over_lat=None,
                    tower_lon=None, tower_lat=None, slope=None,
                    profile="Coverage-first", weights=None) -> list[dict]:
    """Return up to n_sites ranked candidate towers as dicts.

    over_lon/over_lat: OVERLOADED towers only (used for overload-relief score
                       and each site's reported overloaded_relieved count).
    tower_lon/tower_lat: ALL existing towers in the current filter (used for
                       the backhaul-proximity score/report). Optional — pass
                       None to force backhaul weight to 0 regardless of profile.
    slope: optional slope-in-degrees raster, SAME shape/grid as `pop` (see
           data_prep/fetch_srtm_slope.py). Optional — pass None to force
           slope weight to 0 regardless of profile.
    profile: one of WEIGHT_PROFILES' keys; ignored if `weights` is given.
    weights: optional explicit {"population":.., "backhaul":.., "overload":..,
             "slope":..} overriding `profile` — lets a caller set custom
             weights directly.
    """
    w = dict(weights) if weights is not None else dict(WEIGHT_PROFILES[profile])
    if tower_lon is None or len(tower_lon) == 0:
        w["backhaul"] = 0.0
    if slope is None:
        w["slope"] = 0.0

    px = transform.a           # deg per pixel (x, +ve)
    py = -transform.e          # deg per pixel (y, +ve)
    mlon = _m_per_deg_lon(lat0)
    rpx = max(1, int(round(new_range_m / (px * mlon))))
    rpy = max(1, int(round(new_range_m / (py * M_PER_DEG_LAT))))
    r_search = max(rpx, rpy)

    valid = np.isfinite(pop)  # real land data only - never site a tower on a nodata/sea pixel
    work = np.where(valid & ~cov_mask, pop, 0.0).astype("float64")
    H, W = work.shape
    over_lon = np.asarray(over_lon) if over_lon is not None else np.array([])
    over_lat = np.asarray(over_lat) if over_lat is not None else np.array([])

    # --- Static criteria (don't change as sites get picked) — computed once
    backhaul_dist_m = _backhaul_distance_raster(work.shape, transform, lat0, tower_lon, tower_lat)
    backhaul_norm = _minmax_norm(-backhaul_dist_m, valid)  # closer = higher score

    overload_counts = _points_to_raster(work.shape, transform, over_lon, over_lat)
    overload_score = _box_sum(overload_counts, r_search)
    overload_norm = _minmax_norm(overload_score, valid)

    if slope is not None:
        slope_valid = valid & np.isfinite(slope)
        slope_norm = _minmax_norm(-np.nan_to_num(slope, nan=0.0), slope_valid)  # gentler = higher score
    else:
        slope_norm = np.zeros_like(work)

    sites = []
    for rank in range(1, n_sites + 1):
        pop_score = np.where(valid, _box_sum(work, r_search), -np.inf)
        pop_norm = _minmax_norm(pop_score, valid)

        # Eligibility gate: a candidate must reach SOME still-uncovered
        # population to be considered at all, regardless of profile weights
        # — otherwise a heavily backhaul/overload-weighted profile can pick
        # a site right next to an existing tower (great backhaul score) that
        # adds zero new coverage, defeating the point of a coverage-gap tool.
        # Weights RANK eligible candidates; they don't override the gate.
        eligible = valid & (pop_score > 0)
        combined = np.where(
            eligible,
            w["population"] * pop_norm + w["backhaul"] * backhaul_norm
            + w["overload"] * overload_norm + w["slope"] * slope_norm,
            -np.inf,
        )
        if not np.any(eligible):
            break
        i, j = np.unravel_index(int(np.argmax(combined)), (H, W))

        # accurate circular footprint around (i, j) — real captured population,
        # independent of the (possibly multi-criteria) score used to pick it
        r0 = max(0, i - rpy); r1 = min(H, i + rpy + 1)
        c0 = max(0, j - rpx); c1 = min(W, j + rpx + 1)
        rows = np.arange(r0, r1)[:, None]; cols = np.arange(c0, c1)[None, :]
        ell = ((rows - i) / rpy) ** 2 + ((cols - j) / rpx) ** 2 <= 1.0
        gain = float(work[r0:r1, c0:c1][ell].sum())
        if gain <= 0:
            # Rare edge case: the square search window's box-sum was positive
            # (eligibility gate passed) but the accurate CIRCULAR footprint
            # around this exact pixel captures none of it.
            break

        lon, lat = transform * (j + 0.5, i + 0.5)

        # overloaded towers this site could help offload (within footprint)
        relief = 0
        if over_lon.size:
            dx = (over_lon - lon) * mlon
            dy = (over_lat - lat) * M_PER_DEG_LAT
            relief = int(np.count_nonzero(dx * dx + dy * dy <= new_range_m ** 2))

        sites.append({
            "rank": rank, "lon": float(lon), "lat": float(lat),
            "people_gained": gain, "overloaded_relieved": relief,
            "backhaul_distance_m": float(backhaul_dist_m[i, j]) if np.isfinite(backhaul_dist_m[i, j]) else None,
            "slope_deg": float(slope[i, j]) if slope is not None and np.isfinite(slope[i, j]) else None,
            "score": float(combined[i, j]),
        })

        # remove the captured population so the next site is complementary
        block = work[r0:r1, c0:c1]
        block[ell] = 0.0
        work[r0:r1, c0:c1] = block

    cum = 0.0
    for s in sites:
        cum += s["people_gained"]
        s["cumulative_gained"] = cum

    _attach_cost_and_phase(sites)
    return sites


def _attach_cost_and_phase(sites: list[dict]) -> None:
    """Roadmap #7 — estimated capex + people-per-$1000 + 3-phase rollout,
    computed as simple arithmetic on the already-selected sites (no new
    modelling). Phase assignment ranks by cost-efficiency, NOT by the
    original selection rank, so "Phase 1" is genuinely "do these first for
    the best ROI" regardless of which scoring profile chose the sites."""
    for s in sites:
        backhaul_km = (s["backhaul_distance_m"] or 0.0) / 1000.0
        cost_usd = BASE_CAPEX_USD + backhaul_km * FIBER_COST_PER_KM_USD
        s["est_capex_usd"] = round(cost_usd)
        s["people_per_1000usd"] = round(s["people_gained"] / (cost_usd / 1000.0), 2) if cost_usd else 0.0

    by_roi = sorted(sites, key=lambda s: s["people_per_1000usd"], reverse=True)
    phase_size = max(1, math.ceil(len(by_roi) / 3))
    for idx, s in enumerate(by_roi):
        s["phase"] = min(3, idx // phase_size + 1)
