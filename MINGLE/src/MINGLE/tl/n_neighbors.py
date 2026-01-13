import os
import sys
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from matplotlib.patches import Patch
#from MINGLE.tl.knn import KNN
#from MINGLE.pp.preprocessing import read_file

df = pd.read_csv(r"/Volumes/data/MINGLE/Data/Intestine/05_25_huBMAP_tunit.csv")
X = 'x'
Y = 'y'
reg = 'unique_region'
cluster_col = 'Cell Type'  # or your categorical column for dummy encoding
sum_cols = list(df[cluster_col].unique())
keep_cols = [X, Y, reg, cluster_col]

cells = df

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
cells.reset_index(inplace=True, drop=True)
# Concatenate the original 'cells' DataFrame with dummy variables created from 'cluster_col'
# pd.get_dummies() converts categorical variable(s) into dummy/indicator variables
cells = pd.concat([cells, pd.get_dummies(cells[cluster_col])], axis=1)

# Get unique values from the 'cluster_col' column to use for summarization
sum_cols = cells[cluster_col].unique()

# Retrieve the values for these unique categories as a NumPy array
# This array can be used for further analysis or operations later for calculating the neighborhoods
values = cells[sum_cols].values#We can choose a range of nearest neighbors to calculate the neighborhoods
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

knn_feature_cols = ['NK', 'Enterocyte',
       'MUC1+ Enterocyte', 'TA', 'CD66+ Enterocyte', 'Paneth', 'Smooth muscle',
       'M1 Macrophage', 'Goblet', 'Neuroendocrine', 'CD57+ Enterocyte',
       'Lymphatic', 'CD8+ T', 'DC', 'M2 Macrophage', 'B', 'Neutrophil',
       'Endothelial', 'Cycling TA', 'Plasma', 'CD4+ T cell', 'Stroma', 'Nerve',
       'ICC', 'CD7+ Immune']

def run_mingle_over_n_clusters(
    df,
    knn_feature_cols,
    n_range=range(1, 51),
    output_col_template='Neighborhood_{}',
    return_per_cell=True,
    plot_summary=True
):
    """
    CPU-compatible version of your CuPy/GPU function.
    - Keeps the SAME computation logic (including the product-of-pdfs approach + nan_to_num guards).
    - Replaces CuPy arrays/ops with NumPy.
    - Removes .get() calls (GPU->CPU transfers) since everything is already on CPU.

    Returns:
        summary_df (pd.DataFrame)
        per_cell_df (pd.DataFrame, optional)
    """
    import pandas as pd
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    df = df.copy()
    df[knn_feature_cols] = df[knn_feature_cols].astype('float32')

    per_cell_df = df[['x', 'y', 'unique_region']].copy() if return_per_cell else None
    summary_rows = []

    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(df[knn_feature_cols].values)

    for n in tqdm(n_range, desc="Running KMeans + MINGLE (CPU)"):
        km = MiniBatchKMeans(n_clusters=n, random_state=0)
        labels = km.fit_predict(X_scaled_all)
        cluster_col = output_col_template.format(n)
        df[cluster_col] = labels.astype(str)

        # Centroid computation (same logic as your version)
        centroids = []
        for label in map(str, range(n)):
            group = df[df[cluster_col] == label][knn_feature_cols]
            centroids.append({
                'Neighborhood': label,
                **{f'{c}_mean': group[c].mean() for c in knn_feature_cols},
                **{f'{c}_std': group[c].std(ddof=0) for c in knn_feature_cols}
            })
        df_centroids = pd.DataFrame(centroids).set_index("Neighborhood")

        # CPU arrays (NumPy) in place of cp.array
        means = np.array(df_centroids[[f"{c}_mean" for c in knn_feature_cols]].values, dtype=np.float32)
        stds  = np.array(df_centroids[[f"{c}_std"  for c in knn_feature_cols]].values, dtype=np.float32)

        # same safeguard
        stds = np.where(stds < 1e-2, 1e-2, stds)

        label_map = {name: i for i, name in enumerate(df_centroids.index)}
        assigned_labels = df[cluster_col].values
        assigned_indices = [label_map[label] for label in assigned_labels]

        # Evaluation loop (same batching + same math)
        batch_size = 20000
        num_cells = len(df)
        num_batches = (num_cells + batch_size - 1) // batch_size
        log_likelihoods = []
        assigned_probs = []

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_cells)
            batch_df = df.iloc[start:end]

            # CPU float32 batch
            batch_data = np.array(batch_df[knn_feature_cols].values, dtype=np.float32)
            batch_indices = np.array(assigned_indices[start:end], dtype=np.int32)

            batch_means = means[batch_indices]  # (B, F)
            batch_stds  = stds[batch_indices]   # (B, F)

            # log-likelihood for assigned cluster (same formula)
            log_coeff = -np.log(batch_stds) - 0.5 * np.log(2 * np.pi)
            squared_diff = ((batch_data - batch_means) / batch_stds) ** 2
            log_pdf_vals = log_coeff - 0.5 * squared_diff
            log_probs = np.sum(log_pdf_vals, axis=1)
            log_likelihoods.extend(log_probs.tolist())

            # Full PDF matrix for posteriors (same logic: pdf then product across features)
            coeffs = 1.0 / (stds[np.newaxis, :, :] * np.sqrt(2 * np.pi))
            exponents = -0.5 * ((batch_data[:, np.newaxis, :] - means[np.newaxis, :, :]) / stds[np.newaxis, :, :]) ** 2
            pdf_vals = coeffs * np.exp(exponents)

            total_probs = np.prod(pdf_vals, axis=2)
            total_probs = np.nan_to_num(total_probs, nan=1e-300, posinf=1e-300, neginf=1e-300)

            prob_sums = np.sum(total_probs, axis=1, keepdims=True)
            prob_sums = np.where(prob_sums == 0, 1e-300, prob_sums)

            normalized_probs = total_probs / prob_sums
            normalized_probs = np.nan_to_num(normalized_probs, nan=0.0, posinf=0.0, neginf=0.0)

            row_indices = np.arange(end - start)
            batch_probs = normalized_probs[row_indices, batch_indices]
            assigned_probs.extend(batch_probs.tolist())

        # Save summary stats (same)
        avg_log = float(np.mean(log_likelihoods))
        avg_prob = float(np.mean(assigned_probs))
        summary_rows.append({
            'n_clusters': n,
            'avg_log_likelihood': avg_log,
            'avg_assigned_probability': avg_prob
        })

        if return_per_cell:
            per_cell_df[f'log_likelihood_n{n}'] = log_likelihoods
            per_cell_df[f'assigned_prob_n{n}'] = assigned_probs
            per_cell_df[f'Neighborhood_{n}'] = labels

    summary_df = pd.DataFrame(summary_rows)

    if plot_summary:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel("Number of Neighborhoods")
        ax1.set_ylabel("Avg Log-Likelihood", color="tab:blue")
        ax1.plot(summary_df['n_clusters'], summary_df['avg_log_likelihood'], color="tab:blue", marker='o')
        ax1.tick_params(axis='y', labelcolor="tab:blue")
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Avg Assigned Probability", color="tab:green")
        ax2.plot(summary_df['n_clusters'], summary_df['avg_assigned_probability'], color="tab:green", marker='s')
        ax2.tick_params(axis='y', labelcolor="tab:green")

        plt.title("Log-Likelihood and Assigned Probability vs. Neighborhood Count")
        fig.tight_layout()
        plt.show()

    return summary_df, per_cell_df if return_per_cell else summary_df


summary_df, per_cell_df = run_mingle_over_n_clusters(
    df=windows2.copy(),           # preserve your input
    knn_feature_cols=sum_cols,         # your KNN cell type features
    n_range=range(1, 51),
    return_per_cell=True,
    plot_summary=True
)
print(per_cell_df)
print()
print(summary_df)


def find_elbow_point(y_values, x_values=None, threshold=0.01, window=9, polyorder=3):
    """
    Finds the 'elbow' point in a curve using smoothed first derivative.

    Args:
        y_values (array-like): Curve to analyze.
        x_values (array-like or None): X-axis values. If None, use 1..N.
        threshold (float): Gradient threshold to define "elbow".
        window (int): Window size for Savitzky-Golay smoothing.
        polyorder (int): Polynomial order for smoothing.

    Returns:
        elbow_idx (int): Index of the elbow.
    """
    import numpy as np
    from scipy.signal import savgol_filter

    y = np.array(y_values)
    x = np.arange(1, len(y) + 1) if x_values is None else np.array(x_values)

    smoothed = savgol_filter(y, window_length=window, polyorder=polyorder)
    slope = np.gradient(smoothed, x)

    # Find first point where slope flattens below threshold
    flat_idx = np.where(np.abs(slope) < threshold)[0]
    elbow_idx = flat_idx[0] if len(flat_idx) > 0 else np.argmin(slope)

    return elbow_idx, x[elbow_idx], slope


def find_best_unsupervised_plateau(
    log_likelihoods,
    assigned_probs,
    method="harmonic",
    slope_threshold=0.02,
    score_threshold=0.9,
    window_length=9,
    polyorder=3,
    elbow_min=None,
    elbow_max=None
):
    """
    Finds the most biologically meaningful plateau in the composite score curve.

    Restricts search between elbow_min and elbow_max if provided.
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from scipy.signal import savgol_filter
    from itertools import groupby
    from operator import itemgetter

    log_y = np.array(log_likelihoods).reshape(-1, 1)
    prob_y = np.array(assigned_probs).reshape(-1, 1)
    n_clusters = np.arange(1, len(log_y) + 1)

    scaler = MinMaxScaler()
    norm_log = scaler.fit_transform(log_y).flatten()
    norm_prob = scaler.fit_transform(prob_y).flatten()

    if method == "harmonic":
        composite = 2 * (norm_log * norm_prob) / (norm_log + norm_prob + 1e-8)
    elif method == "weighted":
        composite = 0.5 * norm_log + 0.5 * norm_prob
    else:
        raise ValueError("Invalid method")

    composite_smooth = savgol_filter(composite, window_length, polyorder)
    slope = np.gradient(composite_smooth)
    max_score = composite_smooth.max()

    score_mask = composite_smooth >= (score_threshold * max_score)
    slope_mask = np.abs(slope) < slope_threshold
    plateau_mask = score_mask & slope_mask

    # Restrict to elbow range
    if elbow_min is not None and elbow_max is not None:
        elbow_mask = (n_clusters > elbow_min) & (n_clusters < elbow_max)
        plateau_mask = plateau_mask & elbow_mask

    runs = []
    for k, g in groupby(enumerate(plateau_mask), key=lambda x: x[1]):
        if k:
            idxs = list(map(itemgetter(0), g))
            run = {
                'start_idx': idxs[0],
                'length': len(idxs),
                'mean_score': np.mean(composite_smooth[idxs]),
                'mean_slope': np.mean(np.abs(slope[idxs])),
                'idxs': idxs
            }
            runs.append(run)

    if not runs:
        best_idx = int(np.argmax(composite_smooth))
    else:
        runs.sort(key=lambda r: (
            r['mean_slope'],
            -r['mean_score'],
            -r['length'],
            -r['start_idx']
        ))
        best_idx = runs[0]['start_idx']

    best_n = n_clusters[best_idx]

    composite_df = pd.DataFrame({
        'n_clusters': n_clusters,
        'norm_log_likelihood': norm_log,
        'norm_assigned_probability': norm_prob,
        'composite_score': composite,
        'composite_smooth': composite_smooth,
        'composite_slope': slope
    })

    ranked_plateaus = pd.DataFrame(runs)
    if not ranked_plateaus.empty:
        ranked_plateaus['start_n'] = [n_clusters[r['start_idx']] for r in runs]
        ranked_plateaus['rank'] = np.arange(1, len(ranked_plateaus) + 1)
        ranked_plateaus = ranked_plateaus[['rank', 'start_n', 'length', 'mean_score', 'mean_slope', 'start_idx']]
    else:
        ranked_plateaus = pd.DataFrame(columns=['rank', 'start_n', 'length', 'mean_score', 'mean_slope', 'start_idx'])

    print(f"📍 Best plateau starts at n = {best_n} (score = {composite_smooth[best_idx]:.4f})")

    return composite_df, best_n, ranked_plateaus

def plot_stable_composite(df, best_n, ll_n=None, prob_n=None):
    """
    Plots the composite score + slope and places the legend in a separate figure.
    Removes the top and right spines from the main plot.
    """
    import matplotlib.pyplot as plt

    # Set global font size (you can still override locally if wanted)
    plt.rcParams.update({'font.size': 25})

    # Create high-res figure for the main plot
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Plot curves (capture handles for legend)
    h1, = ax.plot(df['n_clusters'], df['composite_score'],
                  label='Composite Score', color='black', marker='o',
                  linewidth=3, markersize=10)
    h2, = ax.plot(df['n_clusters'], df['composite_slope'],
                  label='Slope', linestyle='--', color='gray',
                  linewidth=3)
    # vertical lines (also capture handles)
    h3 = ax.axvline(best_n, color='red', linestyle=':', label=f"Selected n = {best_n}", linewidth=5)
    h4 = None
    h5 = None
    if ll_n is not None:
        h4 = ax.axvline(ll_n, color='blue', linestyle='--', label=f"Log Likelihood Elbow n = {ll_n}", linewidth=5)
    if prob_n is not None:
        h5 = ax.axvline(prob_n, color='orange', linestyle='--', label=f"Probability Elbow n = {prob_n}", linewidth=5)

    # Labels & styling
    ax.set_xlabel("Number of Neighborhoods", fontsize=25)
    ax.set_ylabel("", fontsize=35)
    #ax.set_title("Stable Composite Score Selection", fontsize=35)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.grid(False)

    # Remove top and right spines (keep left & bottom)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Collect handles & labels for legend in defined order
    handles = [h1, h2, h3]
    labels = ['Composite Score', 'Slope', f"Selected n = {best_n}"]
    if h4 is not None:
        handles.append(h4); labels.append(f"Log Likelihood Elbow n = {ll_n}")
    if h5 is not None:
        handles.append(h5); labels.append(f"Probability Elbow n = {prob_n}")

    # Draw the main plot without a legend
    # (we purposely do NOT call ax.legend here)

    plt.tight_layout()
    plt.show()

    # --- Create a separate figure that only contains the legend ---
    # Size of legend figure can be adjusted
    legend_fig = plt.figure(figsize=(8, 2.5), dpi=300)
    legend_ax = legend_fig.add_subplot(111)

    # Turn off axes for the legend figure
    legend_ax.axis('off')

    # Create the legend centered in the figure
    legend = legend_ax.legend(handles=handles, labels=labels,
                              loc='center', frameon=False,
                              fontsize=28, ncol=1)

    # Ensure no box/spine around the legend figure (axis is off above)
    for spine in legend_ax.spines.values():
        spine.set_visible(False)

    legend_fig.tight_layout()
    plt.show()

    # Return the two figure objects in case the caller wants to save them
    return fig, legend_fig

# Step 1: Find elbows
ll_idx, ll_n, _ = find_elbow_point(summary_df['avg_log_likelihood'], summary_df['n_clusters'])
prob_idx, prob_n, _ = find_elbow_point(summary_df['avg_assigned_probability'], summary_df['n_clusters'])

# Step 2: Constrained plateau search
composite_df, best_n, ranked_plateaus = find_best_unsupervised_plateau(
    log_likelihoods=summary_df['avg_log_likelihood'],
    assigned_probs=summary_df['avg_assigned_probability'],
    elbow_min=min(ll_n, prob_n),
    elbow_max=max(ll_n, prob_n)
)

# Step 3: Plot
plot_stable_composite(composite_df, best_n, ll_n, prob_n)

# Step 4: (Optional) View or save ranked plateau table
print(ranked_plateaus)
# ranked_plateaus.to_csv("plateaus_ranked.csv", index=False)
