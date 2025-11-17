#GMM File Includes:
#1. Read-in the file (merged df)
#2. Eliminated hardcoded items 
#3. KNN & get_windows function
#4. GPU based GMM probability function

#TODO:add plotting functions, clean up code,  
import anndata as ad
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm

import time
import sys
import matplotlib.pyplot as plt
import math
import os

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

#For users running GPU accelerated version
import cupy as cp

# Coordinates
X_COL = "x"          # column name for X coordinate
Y_COL = "y"          # column name for Y coordinate

# Region or filename to group images
REGION_COL = "unique_region"

# Cell type annotation
CLUSTER_COL = "Cell Type"

# Neighborhood annotation
NEIGHBORHOOD_COL = "Neighborhood"


# Reads cell CSV file and returns a pandas DataFrame; in this case cella
def read_cell_file(filePath):
    df = pd.read_csv(filePath)
    adata = ad.AnnData(X=np.zeros((len(df), 1)), obs=df)
    return adata


def KNN(cells):

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
        to_fit = tissue.loc[indices][[X_COL, Y_COL]].values

        # Fit the NearestNeighbors model on the tissue's X, Y coordinates
        fit = NearestNeighbors(n_neighbors=n_neighbors).fit(tissue[[X_COL, Y_COL]].values)

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
         

    # Convert AnnData obs to working DataFrame
    obs = cells.obs.copy()
    obs.reset_index(inplace=True, drop=True)

    # Define column names that will be used for neighborhood analysis
    X = X_COL
    Y = Y_COL
    reg = REGION_COL
    cluster_col = CLUSTER_COL

    # List of columns to keep for analysis
    keep_cols = [X, Y, reg, cluster_col]

    # Concatenate with dummy variables for the cell type
    obs = pd.concat([obs, pd.get_dummies(obs[cluster_col])], axis=1)

    # Store dummy variable matrix inside AnnData
    cells.obsm["celltype_matrix"] = obs[pd.get_dummies(obs[cluster_col]).columns].values
    cells.uns["cell_type_features"] = list(pd.get_dummies(obs[cluster_col]).columns)

    # Get unique values from the cluster column
    sum_cols = obs[cluster_col].unique()

    # Retrieve these dummy values
    values = obs[sum_cols].values

    #KNN Set-up
    ks = [5,10,20]
    n_neighbors = max(ks)

    # Group by region
    tissue_group = obs[[X, Y, reg]].groupby(reg)

    # Get list of unique regions
    exps = list(obs[reg].unique())

    # Prepare job chunks for parallelization
    tissue_chunks = [
        (time.time(), exps.index(t), t, a)
        for t, indices in tissue_group.groups.items()
        for a in np.array_split(indices, 1)
    ]

    # Compute neighbors
    tissues = [get_windows(job, n_neighbors) for job in tissue_chunks]

    # Store output windows
    out_dict = {}

    for k in ks:
        for neighbors, job in zip(tissues, tissue_chunks):

            chunk = np.arange(len(neighbors))
            tissue_name = job[2]
            indices = job[3]

            window = values[neighbors[chunk, :k].flatten()].reshape(
                len(chunk), k, len(sum_cols)
            ).sum(axis=1)

            out_dict[(tissue_name, k)] = (window.astype(np.float16), indices)

    windows = {}

    for k in ks:
        window = pd.concat(
            [
                pd.DataFrame(
                    out_dict[(exp, k)][0],
                    index=out_dict[(exp, k)][1].astype(int),
                    columns=sum_cols
                )
                for exp in exps
            ],
            axis=0
        )

        window = window.loc[obs.index.values]
        window = pd.concat([obs[keep_cols], window], axis=1)
        windows[k] = window

    return windows



# Reads  Centroids file and returns a pandas DataFrame; in this case centroid data
def read_centroids_file(filePath):
    df_centroids = pd.read_csv(filePath)
    return df_centroids



#Parameters are the cells dataframe and the centroids dataframe
def gpu_gmm_probability(cells, df_centroids):

    cluster_col = CLUSTER_COL

    # Parameter for cell analysis
    windows = KNN(cells)

    #Choose k value to analyze and pull out from dictionary of stored results of vector
    k = 10
    windows2 = windows[k]

    #Add cell type column to output windows dataframe
    windows2[cluster_col] = cells.obs[cluster_col].values

    # Your neighborhoods and cell types (unchanged)
    neighborhoods_to_loop = cells.obs[NEIGHBORHOOD_COL].unique()
    cell_type_features = cells.obs[cluster_col].unique()

    # Adjust batch size according to your GPU memory (~8GB)
    batch_size = 20000

    num_cells = len(windows2)
    num_batches = (num_cells + batch_size - 1) // batch_size  # ceiling division

    # Extract neighborhood names and centroids
    neighborhood_names = df_centroids['Neighborhood'].values
    mean_cols = [f"{ct}_mean" for ct in cell_type_features]
    std_cols = [f"{ct}_std" for ct in cell_type_features]

    # Convert means and stds once to GPU arrays
    means = cp.array(df_centroids[mean_cols].values)  # shape (num_neighborhoods, num_cell_types)
    stds = cp.array(df_centroids[std_cols].values)    # same shape

    def compute_batch_probs(batch_df):
        # Convert cell data to GPU array
        cell_data = cp.array(batch_df[cell_type_features].values)  # (batch_size, num_cell_types)
        
        # Broadcast dims for probability calc
        cell_data_exp = cell_data[:, cp.newaxis, :]  # (batch_size, 1, num_cell_types)
        means_exp = means[cp.newaxis, :, :]          # (1, num_neighborhoods, num_cell_types)
        stds_exp = stds[cp.newaxis, :, :]            # (1, num_neighborhoods, num_cell_types)
        
        # Avoid division by zero
        stds_exp_safe = cp.where(stds_exp == 0, 1e-10, stds_exp)
        
        # Calculate Gaussian PDF
        coeff = 1.0 / (stds_exp_safe * cp.sqrt(2 * cp.pi))
        exponent = -0.5 * ((cell_data_exp - means_exp) / stds_exp_safe) ** 2
        pdf_vals = coeff * cp.exp(exponent)
        
        # Handle std=0 cases explicitly
        zero_std_mask = (stds_exp == 0)
        equal_mask = (cell_data_exp == means_exp)
        pdf_vals = cp.where(zero_std_mask & equal_mask, 1, pdf_vals)
        pdf_vals = cp.where(zero_std_mask & (~equal_mask), 0, pdf_vals)
        
        # Product over cell types axis to get total probability per cell per neighborhood
        total_probs = cp.prod(pdf_vals, axis=2)  # shape: (batch_size, num_neighborhoods)
        
        # Normalize probabilities per cell
        prob_sums = cp.sum(total_probs, axis=1, keepdims=True)
        normalized_probs = total_probs / prob_sums
        
        # Return to CPU as numpy array
        return normalized_probs.get()

    results = []

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_cells)
        batch_df = windows2.iloc[start:end]
        
        batch_probs = compute_batch_probs(batch_df)
        batch_df_probs = pd.DataFrame(batch_probs, index=batch_df.index, columns=neighborhood_names)
        results.append(batch_df_probs)
        
        print(f"Processed batch {i + 1}/{num_batches}")

    # Concatenate all batches to get final result
    probabilities_df = pd.concat(results).sort_index()

    # Store inside AnnData
    cells.obsm["neighborhood_probabilities"] = probabilities_df.values
    cells.uns["neighborhood_probability_neighborhoods"] = list(probabilities_df.columns)

    return probabilities_df