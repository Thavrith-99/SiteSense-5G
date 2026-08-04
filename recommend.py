"""
Function 3 - recommend where to build the next 5G towers (SiteSense 5G).

Greedy maximum-coverage siting: given the current coverage gap (population
pixels that are NOT covered), repeatedly pick the location whose new-tower
footprint would capture the most still-uncovered people, then remove that
population and pick the next site. The result is a ranked, non-overlapping
shortlist, each site carrying an explainable "people gained" score plus how
many overloaded towers it would help offload.

Search uses a fast integral-image box filter to locate the best cell; the
reported gain and the coverage removal use an accurate circular footprint.
Metre-per-degree scaling is local to Penang (no pyproj).
"""
import numpy as np

M_PER_DEG_LAT = 111_320.0


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


def recommend_sites(pop, cov_mask, transform, lat0, new_range_m,
                    n_sites, over_lon=None, over_lat=None) -> list[dict]:
    """Return up to n_sites ranked candidate towers as dicts."""
    px = transform.a           # deg per pixel (x, +ve)
    py = -transform.e          # deg per pixel (y, +ve)
    mlon = _m_per_deg_lon(lat0)
    rpx = max(1, int(round(new_range_m / (px * mlon))))
    rpy = max(1, int(round(new_range_m / (py * M_PER_DEG_LAT))))
    r_search = max(rpx, rpy)

    work = np.where(np.isfinite(pop) & ~cov_mask, pop, 0.0).astype("float64")
    H, W = work.shape
    over_lon = np.asarray(over_lon) if over_lon is not None else np.array([])
    over_lat = np.asarray(over_lat) if over_lat is not None else np.array([])

    sites = []
    for rank in range(1, n_sites + 1):
        score = _box_sum(work, r_search)
        i, j = np.unravel_index(int(np.argmax(score)), (H, W))
        if score[i, j] <= 0:
            break

        # accurate circular footprint around (i, j)
        r0 = max(0, i - rpy); r1 = min(H, i + rpy + 1)
        c0 = max(0, j - rpx); c1 = min(W, j + rpx + 1)
        rows = np.arange(r0, r1)[:, None]; cols = np.arange(c0, c1)[None, :]
        ell = ((rows - i) / rpy) ** 2 + ((cols - j) / rpx) ** 2 <= 1.0
        gain = float(work[r0:r1, c0:c1][ell].sum())
        if gain <= 0:
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
        })

        # remove the captured population so the next site is complementary
        block = work[r0:r1, c0:c1]
        block[ell] = 0.0
        work[r0:r1, c0:c1] = block

    cum = 0.0
    for s in sites:
        cum += s["people_gained"]
        s["cumulative_gained"] = cum
    return sites
