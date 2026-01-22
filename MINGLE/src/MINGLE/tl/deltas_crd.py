#Import Packages
import pandas as pd
import numpy as np

import time
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Specify the path to your CSV file
file_path = r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_from_h5mu.csv"#r"/Volumes/data/MINGLE/Esophagus/all_regions_from_h5mu.csv

# Read the CSV file
df = pd.read_csv(file_path)
cells = df#[df['donor'] == 'B004']
cells = cells#[cells['unique_region'] == 'B004_Ascending']

copy_cells = cells.copy()

# Print the shape of the DataFrame (rows, columns)
print(cells.shape)

# Optionally, to view the first few rows of the file
print(cells.head())

cells.reset_index(inplace=True, drop=True)
print(cells)

# Define column names that will be used for neighborhood analysis
X = 'x'                  # Variable for the X coordinate
Y = 'y'                  # Variable for the Y coordinate
reg = 'region'         # Variable for the filename or region identifier associated with coordinates
cluster_col = 'Cell Type'  # Variable for cell type/subtype classification
cellid = 'cellid'
neigh = 'neigh_name'
path = "Path_report"

# List of columns to keep for analysis
keep_cols = [X, Y, reg, cluster_col, cellid, neigh, path]

# Concatenate the original 'cells' DataFrame with dummy variables created from 'cluster_col'
# pd.get_dummies() converts categorical variable(s) into dummy/indicator variables
cells = pd.concat([cells, pd.get_dummies(cells[cluster_col])], axis=1)

# Get unique values from the 'cluster_col' column to use for summarization
sum_cols = cells[cluster_col].unique()

# Retrieve the values for these unique categories as a NumPy array
# This array can be used for further analysis or operations later for calculating the neighborhoods
values = cells[sum_cols].values

#We can choose a range of nearest neighbors to calculate the neighborhoods
ks = [5,10,20] # k=5 means it collects 5 nearest neighbors for each center cell
n_neighbors = max(ks) #sets n_neighbors to max of the list that is set

# Group the cell data by region
# 'cells' is a DataFrame containing cell data
# 'tissue_group' will be a GroupBy object with cells grouped by the 'reg' column (representing regions)
tissue_group = cells[[X, Y, reg]].groupby(reg)

# Get a list of unique regions (filenames)
# 'exps' will contain all unique region names found in the 'reg' column of the 'cells' DataFrame
exps = list(cells[reg].unique())

# Prepare chunks of data for processing
# 'tissue_chunks' is a list of tuples, each tuple representing a job for processing
# Each tuple contains the current time, index of the region in 'exps', the region name, and a subset of indices
# 'np.array_split(indices, 1)' splits the indices for each group into chunks (1 chunk in this case)
# This loop goes through each group in 'tissue_group', and for each group, it creates a job tuple
tissue_chunks = [(time.time(), exps.index(t), t, a) for t, indices in tissue_group.groups.items() for a in np.array_split(indices, 1)]

#Function for getting neighborhood windows
def get_windows(job, n_neighbors):

    # Unpack the job tuple containing start_time, idx, tissue_name, and indices
    start_time, idx, tissue_name, indices = job

    # Record the current time to measure the duration of the job
    job_start = time.time()

    # Print a message indicating the start of the job
    print("Starting:", str(idx+1)+'/'+str(len(exps)), ': ' + exps[idx])

    # Get the subset of the dataset for the specific tissue
    tissue = tissue_group.get_group(tissue_name)

    # Extract the coordinates (X, Y) for the points to be fitted from the tissue subset
    to_fit = tissue.loc[indices][[X, Y]].values

    # Fit the NearestNeighbors model on the tissue's X, Y coordinates
    fit = NearestNeighbors(n_neighbors=n_neighbors).fit(tissue[[X, Y]].values)

    # Find the nearest neighbors for the points in 'to_fit'
    m = fit.kneighbors(to_fit)

    # Sort the neighbors
    # 'args' are the indices that would sort the distances
    args = m[0].argsort(axis=1)

    # 'add' is used to adjust indices for flattened array
    add = np.arange(m[1].shape[0]) * m[1].shape[1]

    # Calculate sorted indices for neighbors
    sorted_indices = m[1].flatten()[args + add[:, None]]

    # Retrieve the neighbor indices from the tissue dataset
    neighbors = tissue.index.values[sorted_indices]

    # Record the end time of the job
    end_time = time.time()

    # Print a message indicating the end of the job and the duration
    print("Finishing:", str(idx+1)+"/"+str(len(exps)), ": "+ exps[idx], end_time - job_start, end_time - start_time)

    # Return the neighbor indices as an array of integers
    return neighbors.astype(np.int32)

# Process each job to get the windows (neighbors of the cells)
# 'tissues' is a list of results from the 'get_windows' function
# The 'get_windows' function is applied to each job in 'tissue_chunks'
# 'n_neighbors' is a parameter for the 'get_windows' function, defining the number of neighbors to consider
tissues = [get_windows(job, n_neighbors) for job in tissue_chunks]



# Initialize a dictionary to store the output
out_dict = {}

# Loop over a list of values 'ks' (different numbers of neighbors to consider)
for k in ks:
    # Iterate over each tissue's neighbors and the corresponding job information
    for neighbors, job in zip(tissues, tissue_chunks):

        # Create an array of indices for the current chunk of data
        chunk = np.arange(len(neighbors))  # equivalent to 0, 1, 2, ..., len(neighbors)-1

        # Extract the tissue name and indices from the job tuple
        tissue_name = job[2]  # Region/filename from the job tuple
        indices = job[3]      # Indices from the job tuple

        # Compute the 'window' - a summary measure for the neighborhood of each cell up to the k-th neighbor
        # Reshape and sum values to get a compact representation of neighborhood information
        window = values[neighbors[chunk, :k].flatten()].reshape(len(chunk), k, len(sum_cols)).sum(axis=1)

        # Store the computed window and indices in the output dictionary
        # Keyed by a tuple of (tissue_name, k)
        out_dict[(tissue_name, k)] = (window.astype(np.float16), indices)

# Initialize a dictionary to store the final windows data
windows = {}

# Iterate over each value of k again to process the stored information
for k in ks:

    # Concatenate data for each experiment ('exp') into a DataFrame
    # This DataFrame contains the window data for each cell, indexed by cell indices, for the current value of k
    window = pd.concat([pd.DataFrame(out_dict[(exp, k)][0], index=out_dict[(exp, k)][1].astype(int), columns=sum_cols) for exp in exps], axis=0)

    # Ensure the window data is in the same order as the original cells DataFrame
    window = window.loc[cells.index.values]

    # Concatenate the window data with the original columns specified in 'keep_cols'
    window = pd.concat([cells[keep_cols], window], axis=1)

    # Store the concatenated DataFrame in the 'windows' dictionary, keyed by the current value of k
    windows[k] = window

#Choose k value to analyze and pull out from dictionary of stored results of vectors
k = 10
windows2 = windows[k]
#Add cell type column to output windows dataframe
windows2[cluster_col] = cells[cluster_col]

# List of cell type features
cell_type_features = [
    'Squamous Annexin A1+', 'Squamous p63+', 'Endothelial', 'Chief',
       'Squamou p63+ EGFRhi', 'Neutrophil', 'M1 Macrophage', 'Epithelial',
       'Stroma', 'Epithelial Ki67+ p53+', 'Endothelial CD36hi',
       'CD4+ Treg', 'CD4+ T cell PD1+', 'CD4+ T cell', 'Nerve',
       'CD8+ T cell', 'Epithelial MUC1+ Ki67+', 'B cell',
       'CD8+ T cell PD1+', 'M2 Macrophage', 'Foveloar', 'Neuroendocrine',
       'Epithelial CK7+ p53+', 'Smooth Muscle', 'Epithelial pH2AX+', 'DC',
       'Plasma', 'Epithelial p53+', 'Endothelial aSMAhi', 'Lymphatic',
       'Neck', 'Goblet', 'Parietal', 'Epithelial CD73hi',
       'Foveloar Ki67+ p53+', 'Stroma CD73+', 'Goblet p53+', 'Neck p53+',
       'Lymphatic CD73+', 'Epithelial CK7+', 'Foveloar p53+',
       'Goblet Ki67+ p53+', 'Paneth', 'Epithelial HLADR+',
       'Neck Ki67+ p53+'
]

probabilities_df = pd.read_csv(r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_esophagus_all_cells_all_neighborhood_probs.csv")

# List of all neighborhoods (column names from probabilities_df)
neighborhoods = probabilities_df.columns.tolist()
neighborhoods.remove('cellid')  # If 'cellid' is a column in there
neighborhoods.remove('Unnamed: 0')

# Updated GPU region-level probability computation (CuPy, log-space, robust to underflow)


# Paths (update if needed)
assigned_path = r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_from_h5mu.csv"
probabilities_path = r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_esophagus_all_cells_all_neighborhood_probs.csv"
out_probs_path = "local_probs.csv"
out_delta_path = "delta_probs.csv"
out_folder = Path("mingle_level_comparisons")
out_folder.mkdir(parents=True, exist_ok=True)

# Load dataframes (you can skip loading if already in memory)
assigned_df = pd.read_csv(assigned_path).set_index("cellid")  # must contain neigh_name and region
probabilities_df = pd.read_csv(probabilities_path)  # global probs (has 'cellid' col)
# If you already have windows2, copy_cells, cell_type_features, neighborhoods in memory, they will be used.
# Otherwise, ensure windows2 and copy_cells are loaded in the environment prior to running this block.

# --- User: ensure these variables exist in your session ---
# windows2: DataFrame containing cell rows with 'cellid' and 'region' columns and the feature columns in cell_type_features
# copy_cells: DataFrame indexed the same as windows2 (or with matching index) containing 'neigh_name'
# cell_type_features: list of feature column names (the same list you provided earlier)
# neighborhoods: list of neighborhood names, in the same order as used in probabilities_df columns (excluding 'cellid' column)

# If these are not present, try to load or recreate them before running this block.
# Example (uncomment if you want to construct neighborhoods from probabilities_df):
# all_cols = probabilities_df.columns.tolist()
# if 'cellid' in all_cols: all_cols.remove('cellid')
# neighborhoods = all_cols

# Basic validation
required_assigned_cols = {"neigh_name", "region"}
if not required_assigned_cols.issubset(set(assigned_df.columns)):
    raise ValueError(f"assigned_df must contain columns: {required_assigned_cols}")

# Prepare storage
all_region_probs = []
all_region_deltas = []

# Helper: stable logsumexp on cupy (fallback manual)
def cp_logsumexp(a, axis=1, keepdims=True):
    # a : cupy array
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, -1e300)
    s = np.sum(np.exp(a - a_max), axis=axis, keepdims=True)
    s = np.where(s == 0, np.nan, s)  # avoid log(0)
    return a_max + np.log(s)

# Precompute global fallback stats for centroids (used when neighborhood has <=1 cell in region)
# We'll compute these per-region inside the loop from df_region_centroids as in your original code,
# but having a global fallback computed once is helpful:
# (not strictly necessary - will be computed in loop)

# Iterate regions
unique_regions = windows2['region'].unique()
print("Processing regions:", len(unique_regions))

for region in tqdm(unique_regions, desc="Processing regions (log-space GPU)"):
    # 1) Filter cells for this region
    region_cells = windows2[windows2['region'] == region].copy()
    region_cell_ids = region_cells['cellid'].values
    # Extract feature matrix for this region (rows in same order as region_cells)
    cell_data = region_cells[cell_type_features].copy()
    C = cell_data.shape[0]
    if C == 0:
        print(f"  - Region {region}: no cells, skipping")
        continue

    # 2) Add assigned neighborhood (from your copy_cells; keep index alignment)
    # ensure index alignment: region_cells.index corresponds to copy_cells' index in your original pipeline
    try:
        region_cells['neigh_name'] = copy_cells.loc[region_cells.index, 'neigh_name'].values
    except Exception:
        # fallback: if copy_cells indexed by cellid, map by cellid
        if 'cellid' in copy_cells.columns:
            mapping = copy_cells.set_index('cellid')['neigh_name'].to_dict()
            region_cells['neigh_name'] = region_cells['cellid'].map(mapping).values
        else:
            raise

    # 3) Build region-specific centroids for each neighborhood (as you had)
    region_results = []
    neigh_counts = []
    for neighborhood in neighborhoods:
        neighborhood_cells = region_cells[region_cells['neigh_name'] == neighborhood]
        matching_cell_ids = neighborhood_cells.index
        neigh_counts.append(len(matching_cell_ids))
        stats = {"Neighborhood": neighborhood}
        if len(matching_cell_ids) <= 1:
            # mark zeros for now; we'll replace low-count neighborhoods with fallback after computing centroids
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

    # 3a) Compute fallback mean/std across neighborhoods in this region (ignore NaNs)
    centroid_means = df_region_centroids[[f"{c}_mean" for c in cell_type_features]].values.astype(float)  # (K,F)
    centroid_stds = df_region_centroids[[f"{c}_std" for c in cell_type_features]].values.astype(float)    # (K,F)

    # Fallback: per-feature global mean (across neighborhoods where available) and median std (so not tiny)
    # If a feature's median std is zero or nan, fall back to a small positive constant.
    global_mean_fallback = np.nanmean(centroid_means, axis=0)
    global_std_fallback = np.nanmedian(np.where(np.isnan(centroid_stds), np.nan, centroid_stds), axis=0)
    # Safety for fallback std
    global_std_fallback = np.where(np.isnan(global_std_fallback) | (global_std_fallback <= 0), 1e-1, global_std_fallback)

    # Replace NaN means/stds for neighborhoods with <=1 cell with fallback values
    neigh_counts = np.array(neigh_counts)
    low_mask = neigh_counts <= 1
    if low_mask.any():
        centroid_means[low_mask, :] = global_mean_fallback[None, :]
        centroid_stds[low_mask, :] = global_std_fallback[None, :]

    # Final safety: ensure no zeros in stds (avoid divide by zero)
    centroid_stds = np.where(centroid_stds <= 0, 1e-6, centroid_stds)

    # 4) Move centroids and cell data to GPU (use float64 for numerical stability)
    region_means_cp = np.array(centroid_means, dtype=np.float64)   # (K, F)
    region_stds_cp = np.array(centroid_stds, dtype=np.float64)     # (K, F)
    cell_array_cp = np.array(cell_data.values.astype(np.float64), dtype=np.float64)  # (C, F)

    # 5) Compute log-pdf per feature on GPU using broadcasting (avoid underflow)
    # Expand dims: X (C,1,F), M (1,K,F), S (1,K,F)
    X = cell_array_cp[:, None, :]       # C x 1 x F
    M = region_means_cp[None, :, :]     # 1 x K x F
    S = region_stds_cp[None, :, :]      # 1 x K x F

    # log coefficient and exponent
    log_coeff = -0.5 * np.log(2.0 * np.pi * (S ** 2))   # C x K x F (broadcasted)
    exponent = -0.5 * ((X - M) / S) ** 2
    log_pdf = log_coeff + exponent   # C x K x F

    # Sum log-pdf across features -> log_total (C x K)
    log_total = np.sum(log_pdf, axis=2)  # shape (C, K)

    # Normalize in log-space using stable logsumexp
    row_logsum = cp_logsumexp(log_total, axis=1, keepdims=True)  # C x 1
    log_prob = log_total - row_logsum  # C x K
    probs_cp = np.exp(log_prob)        # C x K, rows should sum to 1 (or NaN if row_logsum was NaN)

    # move back to numpy
    try:
        local_probs_np = probs_cp#.get()     # shape (C, K)
    except Exception as e:
        # fallback: copy in chunks if memory issue
        local_probs_np = np.array(probs_cp)  # may still raise; usually .get() works

    # Diagnostics: rows with NaN or zero sum
    row_sums = np.nansum(local_probs_np, axis=1)
    n_nan_rows = int(np.sum(np.isnan(row_sums)))
    n_zero_rows = int(np.sum(np.isclose(row_sums, 0.0, atol=1e-12)))
    print(f"Region {region}: cells={C}, NaN-sum-rows={n_nan_rows}, zero-sum-rows={n_zero_rows}")

    # 6) Retrieve global probabilities for the same cells (align by cellid)
    # Ensure probabilities_df has 'cellid' column and neighborhoods in same order
    global_idxed = probabilities_df.set_index('cellid')
    # Some region_cell_ids might not be present in probabilities_df; select intersection
    common_ids = [cid for cid in region_cell_ids if cid in global_idxed.index]
    if len(common_ids) != len(region_cell_ids):
        # create global_probs array aligned to region_cell_ids: missing -> NaN
        global_probs = np.full((len(region_cell_ids), len(neighborhoods)), np.nan, dtype=float)
        present_mask = [cid in global_idxed.index for cid in region_cell_ids]
        if any(present_mask):
            present_ids = [cid for cid, present in zip(region_cell_ids, present_mask) if present]
            global_present_vals = global_idxed.loc[present_ids, neighborhoods].values
            # fill into proper rows
            for i, cid in enumerate(region_cell_ids):
                if cid in global_idxed.index:
                    global_probs[i, :] = global_idxed.loc[cid, neighborhoods].values
    else:
        global_probs = global_idxed.loc[region_cell_ids, neighborhoods].values

    # 7) Build local_probs_df and save to list
    local_probs_df = pd.DataFrame(local_probs_np, columns=neighborhoods)
    local_probs_df['cellid'] = region_cells['cellid'].values
    local_probs_df['region'] = region
    local_probs_df['neigh_name'] = region_cells['neigh_name'].values
    all_region_probs.append(local_probs_df)

    # 8) Compute delta (local - global) (per-row subtraction, NaNs propagate)
    if global_probs.shape != local_probs_np.shape:
        # Align shapes (global may contain NaNs for missing rows)
        # global_probs should be same rows as region_cell_ids order
        # if local_probs_np rows match region_cell_ids order, we can proceed; missing global rows are NaN
        # Already handled above to create global_probs with NaNs where missing
        pass

    delta_values = local_probs_np - global_probs  # this will produce NaNs where either side is NaN
    delta_df = pd.DataFrame(delta_values, columns=[f"{n}_delta" for n in neighborhoods])
    delta_df["cellid"] = region_cells['cellid'].values
    delta_df["region"] = region
    delta_df["neigh_name"] = region_cells["neigh_name"].values
    all_region_deltas.append(delta_df)

# After loop: combine and save
final_probs_df = pd.concat(all_region_probs, ignore_index=True)
final_deltas_df = pd.concat(all_region_deltas, ignore_index=True)

final_probs_df.to_csv(out_probs_path, index=False)
final_deltas_df.to_csv(out_delta_path, index=False)

print("✅ Done. Saved region-level local probs to:", out_probs_path)
print("✅ Done. Saved region-level delta probs to:", out_delta_path)

exit()
# ABOVE IS FIRST PART

