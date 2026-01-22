# Updated GPU region-level probability computation (CuPy, log-space, robust to underflow)
import numpy as np
import pandas as pd
import numpy as cp
from pathlib import Path
from tqdm import tqdm
from scipy.stats import entropy as _entropy

# small alias used in diagnostics
_np = np

# Paths (update if needed)
assigned_path = r"Z:\MINGLE\Data\Esophagus\all_regions_from_h5mu.csv"
probabilities_path = r"Z:\MINGLE\Data\Esophagus\all_regions_esophagus_all_cells_all_neighborhood_probs.csv"
out_probs_path = r"Z:\MINGLE\Data\Esophagus\20251217_all_regions_local_probs.csv"
out_delta_path = r"Z:\MINGLE\Data\Esophagus\20251217_all_regions_delta_probs.csv"
out_folder = Path(r"Z:\MINGLE\Data\Esophagus\mingle_level_comparisons")
out_folder.mkdir(parents=True, exist_ok=True)

# Load dataframes (you can skip loading if already in memory)
assigned_df = pd.read_csv(assigned_path).set_index("cellid")  # must contain neigh_name and region
probabilities_df = pd.read_csv(probabilities_path)  # global probs (has 'cellid' col)

# Basic validation
required_assigned_cols = {"neigh_name", "region"}
if not required_assigned_cols.issubset(set(assigned_df.columns)):
    raise ValueError(f"assigned_df must contain columns: {required_assigned_cols}")

# PARAMETERS YOU REQUESTED
min_count = 10            # neighborhoods with fewer than this are excluded
min_floor_abs = 0.05      # absolute variance floor per feature
use_region_frac = 0.25    # floor as fraction of region std
shrink_alpha = 5.0        # variance shrinkage weight for small-n neighborhoods

# Sanitizer utility (define once)
def sanitize_centroids(means, stds, counts, region_cells,
                       min_floor_abs=min_floor_abs, use_region_frac=use_region_frac, shrink_alpha=shrink_alpha):
    """
    Clean and stabilize neighborhood centroids (means,stds).
    - means: (K, F)
    - stds:  (K, F)
    - counts: (K,)
    - region_cells: DataFrame for region (used to compute region-level std)
    Returns: (means_clean, stds_shrunk)
    """
    K, F = means.shape

    # region-level fallbacks
    region_mean = np.nanmean(means, axis=0) if np.any(np.isfinite(means)) else np.zeros(F, dtype=float)
    region_std  = np.nanstd(region_cells[cell_type_features].values.astype(float), axis=0, ddof=1)
    region_std  = np.where(np.isnan(region_std) | (region_std <= 0), 0.1, region_std)
    min_std_vec = np.maximum(region_std * use_region_frac, min_floor_abs)

    # 1) Replace inf/nan means with region_mean (broadcast-safe)
    means_clean = np.where(np.isfinite(means), means, region_mean[None, :])

    # 2) Replace nan/inf stds with per-feature floor and enforce floor
    stds_clean = np.where(np.isfinite(stds), stds, min_std_vec[None, :])
    stds_clean = np.maximum(stds_clean, min_std_vec[None, :])

    # 3) Shrink small-n variances toward region variance
    n = counts.astype(float) + 1e-8
    neigh_var = stds_clean ** 2
    region_var = (region_std ** 2)[None, :]
    combined_var = (n[:, None] * neigh_var + shrink_alpha * region_var) / (n[:, None] + shrink_alpha)
    stds_shrunk = np.sqrt(np.maximum(combined_var, min_std_vec[None, :]**2))
    stds_shrunk = np.maximum(stds_shrunk, min_std_vec[None, :])

    return means_clean, stds_shrunk

# Helper: stable logsumexp on cupy
def cp_logsumexp(a, axis=1, keepdims=True):
    a_max = cp.max(a, axis=axis, keepdims=True)
    a_max = cp.where(cp.isfinite(a_max), a_max, -1e300)
    s = cp.sum(cp.exp(a - a_max), axis=axis, keepdims=True)
    s = cp.where(s == 0, cp.nan, s)
    return a_max + cp.log(s)

# Prepare storage
all_region_probs = []
all_region_deltas = []

# NOTE: script expects these variables to be defined in the surrounding scope:
# windows2: DataFrame with all cells and region column
# neighborhoods: list (ordered) of all neighborhood labels (full set)
# cell_type_features: list of feature/marker column names (F features)
# copy_cells: DataFrame mapping cellid->neigh_name (or with a 'cellid' column)
# If not present, load/define them before running the loop.

unique_regions = windows2['region'].unique()
print("Processing regions:", len(unique_regions))

for region in tqdm(unique_regions, desc="Processing regions (log-space GPU)"):
    # 1) Filter cells for this region
    region_cells = windows2[windows2['region'] == region].copy()
    region_cell_ids = region_cells['cellid'].values
    cell_data = region_cells[cell_type_features].copy()
    C = cell_data.shape[0]
    if C == 0:
        print(f"  - Region {region}: no cells, skipping")
        continue

    # Replace infinities in raw cell_data (prevents inf in means/stds)
    cell_data = cell_data.replace([np.inf, -np.inf], np.nan)

    # 2) Add assigned neighborhood (from your copy_cells; keep index alignment)
    try:
        region_cells['neigh_name'] = copy_cells.loc[region_cells.index, 'neigh_name'].values
    except Exception:
        if 'cellid' in copy_cells.columns:
            mapping = copy_cells.set_index('cellid')['neigh_name'].to_dict()
            region_cells['neigh_name'] = region_cells['cellid'].map(mapping).values
        else:
            raise

    # 3) Build region-specific centroids for each neighborhood
    region_results = []
    neigh_counts = []
    for neighborhood in neighborhoods:
        neighborhood_cells = region_cells[region_cells['neigh_name'] == neighborhood]
        matching_cell_ids = neighborhood_cells.index
        neigh_counts.append(len(matching_cell_ids))
        stats = {"Neighborhood": neighborhood}
        if len(matching_cell_ids) <= 1:
            for col in cell_type_features:
                stats[f"{col}_mean"] = np.nan
                stats[f"{col}_std"] = np.nan
        else:
            for col in cell_type_features:
                stats[f"{col}_mean"] = neighborhood_cells[col].mean()
                std_val = neighborhood_cells[col].std(ddof=0)
                stats[f"{col}_std"] = np.nan if pd.isna(std_val) else std_val
        region_results.append(stats)

    df_region_centroids = pd.DataFrame(region_results)  # K x (2F + 1)
    K = df_region_centroids.shape[0]

    # Extract raw arrays (now defined)
    centroid_means = df_region_centroids[[f"{c}_mean" for c in cell_type_features]].values.astype(float)  # (K,F)
    centroid_stds  = df_region_centroids[[f"{c}_std"  for c in cell_type_features]].values.astype(float)  # (K,F)
    counts = np.array(neigh_counts, dtype=int)  # (K,)

    # DIAGNOSTICS: show counts and centroid std summary
    neigh_counts_arr = _np.array(neigh_counts)
    flat_stds = centroid_stds.flatten()
    pct = lambda q: float(_np.nanpercentile(flat_stds, q))
    print(f"  Region '{region}': cell_count={C}, neighborhoods={K}")
    print(f"   neigh_counts: min={neigh_counts_arr.min()}, median={_np.median(neigh_counts_arr)}, max={neigh_counts_arr.max()}")
    print("   centroid std percentiles (1,5,25,50,75,95,99,100):",
          pct(1), pct(5), pct(25), pct(50), pct(75), pct(95), pct(99), _np.nanmax(flat_stds))
    small_frac = float(_np.nanmean(flat_stds <= 1e-3))
    print(f"   fraction of centroid stds <= 1e-3: {small_frac:.3f}")

    # SANITIZE centroids (fill inf/nan, floor stds, shrink small-n variances)
    means_clean, stds_shrunk = sanitize_centroids(centroid_means.copy(), centroid_stds.copy(), counts, region_cells,
                                                  min_floor_abs=min_floor_abs, use_region_frac=use_region_frac, shrink_alpha=shrink_alpha)

    # Mark neighborhoods with too few cells as excluded (keep NaNs so we can set probs to NaN later)
    low_mask = counts < min_count
    if low_mask.any():
        # keep NaNs to indicate excluded neighborhoods; not forcing fill here
        means_clean[low_mask, :] = np.nan
        stds_shrunk[low_mask, :] = np.nan

    # ---------- Build compute arrays but DO NOT force-fill excluded neighborhoods ----------
    # region-level fallbacks for feature-wise missingness only (do NOT force-fill excluded neigh rows)
    region_mean  = np.nanmean(means_clean, axis=0)
    # if some features are NaN across all neighborhood-means, fallback to region cell-level mean
    alt_region_mean = np.nanmean(cell_data.values.astype(float), axis=0)
    region_mean = np.where(np.isfinite(region_mean), region_mean, np.where(np.isfinite(alt_region_mean), alt_region_mean, 0.0))

    region_std  = np.nanstd(cell_data.values.astype(float), axis=0, ddof=1)
    region_std  = np.where(np.isnan(region_std) | (region_std <= 0), 0.1, region_std)
    min_std_vec = np.maximum(region_std * use_region_frac, min_floor_abs)

    compute_means = np.where(np.isfinite(means_clean), means_clean, region_mean[None, :])   # (K, F) - but some rows may be NaN
    compute_stds  = np.where(np.isfinite(stds_shrunk), stds_shrunk, min_std_vec[None, :])   # (K, F)

    # Determine which neighborhoods are truly valid for THIS REGION:
    # require 1) counts >= min_count, and 2) fully-finite mean and std across all features
    counts = np.array(neigh_counts, dtype=int)
    count_mask = counts >= min_count
    finite_mask = np.all(np.isfinite(compute_means) & np.isfinite(compute_stds), axis=1)
    valid_mask = count_mask & finite_mask

    valid_idx = np.where(valid_mask)[0]      # indices in `neighborhoods` that we'll compute for
    invalid_idx = np.where(~valid_mask)[0]   # columns we will leave as NaN

    K_full = len(neighborhoods)   # full neighborhood count (keeps final matrices compatible)

    # If there are no valid neighborhoods, create an all-NaN local_probs array and skip GPU
    if valid_idx.size == 0:
        local_probs_np = np.full((C, K_full), np.nan, dtype=float)
        print(f"  Region {region}: NO valid neighborhoods (all excluded) -> local_probs all NaN")
    else:
        # Build small compute arrays for only the valid neighborhoods
        means_valid = compute_means[valid_idx, :]   # (K_valid, F)
        stds_valid  = compute_stds[valid_idx, :]    # (K_valid, F)

        # Sanity: ensure finite for GPU subset
        if not (np.isfinite(means_valid).all() and np.isfinite(stds_valid).all()):
            # This shouldn't happen, but if it does, mark all NaN and skip
            local_probs_np = np.full((C, K_full), np.nan, dtype=float)
            print(f"  Region {region}: unexpected non-finite in valid subset -> skipping GPU for safety")
        else:
            # Move subset to GPU
            region_means_cp = cp.asarray(means_valid, dtype=cp.float64)   # (K_valid, F)
            region_stds_cp  = cp.asarray(stds_valid, dtype=cp.float64)    # (K_valid, F)
            cell_array_cp   = cp.asarray(cell_data.values.astype(np.float64), dtype=cp.float64)  # (C, F)

            # Broadcast: X: C x 1 x F, M: 1 x K_valid x F, S: 1 x K_valid x F
            X = cell_array_cp[:, None, :]       # C x 1 x F
            M = region_means_cp[None, :, :]     # 1 x K_valid x F
            S = region_stds_cp[None, :, :]      # 1 x K_valid x F

            # log-pdf per feature (stable, avoids underflow)
            log_coeff = -0.5 * cp.log(2.0 * cp.pi * (S ** 2))
            exponent = -0.5 * ((X - M) / S) ** 2
            log_pdf = log_coeff + exponent     # C x K_valid x F

            # sum across features -> C x K_valid
            log_total = cp.sum(log_pdf, axis=2)

            # stable normalization across K_valid neighborhoods (axis=1)
            row_logsum = cp_logsumexp(log_total, axis=1, keepdims=True)   # (C,1)
            log_prob_valid = log_total - row_logsum                       # (C, K_valid)
            probs_valid_cp = cp.exp(log_prob_valid)                       # (C, K_valid)

            # bring back to cpu
            try:
                probs_valid = probs_valid_cp.get()
            except Exception:
                probs_valid = np.array(probs_valid_cp)

            # create full-sized C x K matrix with NaN, and fill valid columns with computed probs
            local_probs_np = np.full((C, K_full), np.nan, dtype=float)
            local_probs_np[:, valid_idx] = probs_valid

            # ensure explicitly that invalid columns are NaN (keeps semantics)
            if invalid_idx.size:
                local_probs_np[:, invalid_idx] = np.nan

            # cleanup GPU arrays to free memory
            del region_means_cp, region_stds_cp, cell_array_cp, X, M, S, log_pdf, log_total, row_logsum, log_prob_valid, probs_valid_cp
            try:
                cp._default_memory_pool.free_all_blocks()  # release CuPy memory pool (optional)
            except Exception:
                pass

    # Diagnostics: rows with NaN-only or rows that have finite columns summing to ~0
    rows_all_nan = np.all(np.isnan(local_probs_np), axis=1)
    row_sums = np.nansum(local_probs_np, axis=1)   # sums across finite columns only
    n_all_nan_rows = int(rows_all_nan.sum())
    n_zero_rows = int(((~rows_all_nan) & np.isclose(row_sums, 0.0, atol=1e-12)).sum())
    print(f"  Region {region}: cells={C}, rows_all_nan={n_all_nan_rows}, zero-sum-rows={n_zero_rows}")

    # assigned neighborhood index mapping (neigh_name -> column index)
    col_to_idx = {col: i for i, col in enumerate(neighborhoods)}
    assigned_idx = [col_to_idx.get(n, None) for n in region_cells['neigh_name'].values]
    assigned_idx_arr = _np.array([i if i is not None else -1 for i in assigned_idx], dtype=int)

    # compute assigned probs per-row where mapping exists (propagates NaN if assigned neigh excluded)
    assigned_probs = _np.full(len(local_probs_np), _np.nan)
    valid_rows = assigned_idx_arr >= 0
    if valid_rows.any():
        assigned_probs[valid_rows] = local_probs_np[valid_rows, assigned_idx_arr[valid_rows]]

    # Diagnostics prints for assigned probs (NaNs propagate)
    print("  assigned-prob: mean, median, min, max:",
          _np.nanmean(assigned_probs), _np.nanmedian(assigned_probs),
          _np.nanmin(assigned_probs), _np.nanmax(assigned_probs))

    # Entropy per-row, ignoring NaNs (base 2). If row has all NaNs -> entropy = NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        p = local_probs_np.copy()
        # set any non-finite or negative values to zero (negatives shouldn't occur)
        p = np.where((p >= 0) & np.isfinite(p), p, 0.0)
        mask = p > 0.0
        ent = -np.nansum(p * np.log2(np.where(mask, p, 1.0)), axis=1)
        ent[rows_all_nan] = np.nan

    print("  entropy (bits) mean/median/min/max:",
          _np.nanmean(ent), _np.nanmedian(ent), _np.nanmin(ent), _np.nanmax(ent))

    # optionally show neighborhoods with <=1 cell
    low_count_neighs = [neigh for neigh, cnt in zip(neighborhoods, neigh_counts) if cnt <= 1]
    if len(low_count_neighs) > 0:
        print(f"  neighborhoods with <=1 cell in region: {len(low_count_neighs)} (examples):", low_count_neighs[:6])

    # 6) Retrieve global probabilities for the same cells (align by cellid)
    global_idxed = probabilities_df.set_index('cellid')
    common_ids = [cid for cid in region_cell_ids if cid in global_idxed.index]
    if len(common_ids) != len(region_cell_ids):
        global_probs = np.full((len(region_cell_ids), K_full), np.nan, dtype=float)
        present_mask = [cid in global_idxed.index for cid in region_cell_ids]
        if any(present_mask):
            for i, cid in enumerate(region_cell_ids):
                if cid in global_idxed.index:
                    global_probs[i, :] = global_idxed.loc[cid, neighborhoods].values
    else:
        global_probs = global_idxed.loc[region_cell_ids, neighborhoods].values

    # Prepare x/y for output (case-insensitive: try 'x'/'y' then 'X'/'Y', else NaN)
    x_vals = region_cells.get('x', region_cells.get('X', np.nan))
    y_vals = region_cells.get('y', region_cells.get('Y', np.nan))

    # 7) Build local_probs_df and save to list (include x,y)
    local_probs_df = pd.DataFrame(local_probs_np, columns=neighborhoods)
    local_probs_df['cellid'] = region_cells['cellid'].values
    local_probs_df['region'] = region
    local_probs_df['neigh_name'] = region_cells['neigh_name'].values
    # attach x,y columns (align by index)
    local_probs_df['x'] = np.array(x_vals)
    local_probs_df['y'] = np.array(y_vals)
    all_region_probs.append(local_probs_df)

    # 8) Compute delta (local - global) (NaNs propagate) and include x,y
    delta_values = local_probs_np - global_probs
    delta_df = pd.DataFrame(delta_values, columns=[f"{n}_delta" for n in neighborhoods])
    delta_df["cellid"] = region_cells['cellid'].values
    delta_df["region"] = region
    delta_df["neigh_name"] = region_cells["neigh_name"].values
    delta_df['x'] = np.array(x_vals)
    delta_df['y'] = np.array(y_vals)
    all_region_deltas.append(delta_df)

# After loop: combine and save
final_probs_df = pd.concat(all_region_probs, ignore_index=True)
final_deltas_df = pd.concat(all_region_deltas, ignore_index=True)

final_probs_df.to_csv(out_probs_path, index=False)
final_deltas_df.to_csv(out_delta_path, index=False)

print("✅ Done. Saved region-level local probs to:", out_probs_path)
print("✅ Done. Saved region-level delta probs to:", out_delta_path)
