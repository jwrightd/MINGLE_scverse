#LLM conversion to Scverse

import anndata as ad
import seaborn as sns
import numpy as np
from scipy.stats import norm
import time
import sys
import matplotlib.pyplot as plt
import math
import os
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# @param filePath: filePath to .csv or .h5ad containing cell information
def read_file(filePath):
    # For .h5ad files
    #TODO: .csv compatibility
    return ad.read(filePath)

# @param adata: AnnData object containing cell information
def KNN(adata):
    def get_windows(job, n_neighbors):
        # Unpack the job tuple containing start_time, idx, tissue_name, and indices
        start_time, idx, tissue_name, indices = job

        # Record the current time to measure the duration of the job
        job_start = time.time()

        # Get the subset of the dataset for the specific tissue
        tissue = adata[adata.obs['filename'] == tissue_name]

        # Extract the coordinates (X, Y) for the points to be fitted from the tissue subset
        to_fit = tissue.obs[['x', 'y']].values

        # Fit the NearestNeighbors model on the tissue's X, Y coordinates
        fit = NearestNeighbors(n_neighbors=n_neighbors).fit(tissue.obs[['x', 'y']].values)

        # Find the nearest neighbors for the points in 'to_fit'
        m = fit.kneighbors(to_fit)

        # Sort the neighbors
        args = m[0].argsort(axis=1)
        add = np.arange(m[1].shape[0]) * m[1].shape[1]
        sorted_indices = m[1].flatten()[args + add[:, None]]

        # Retrieve the neighbor indices from the tissue dataset
        neighbors = tissue.obs.index.values[sorted_indices]

        # Return the neighbor indices as an array of integers
        return neighbors.astype(np.int32)
    
    # Extract relevant columns for KNN analysis
    X = 'x'
    Y = 'y'
    reg = 'filename'
    cluster_col = 'Cell_Type'

    # Create dummy variables from cluster_col and add to AnnData
    adata = adata.copy()  # Avoid modifying original AnnData object
    adata.obs = pd.concat([adata.obs, pd.get_dummies(adata.obs[cluster_col])], axis=1)

    sum_cols = adata.obs[cluster_col].unique()
    values = adata.obs[sum_cols].values

    ks = [5, 10, 20]  # Range of neighbors to analyze
    n_neighbors = max(ks)  # Set max k as the number of neighbors

    # Group by region (filename)
    tissue_group = adata.obs.groupby(reg)

    # Prepare chunks for processing
    exps = list(adata.obs[reg].unique())
    tissue_chunks = [(time.time(), exps.index(t), t, a) for t, indices in tissue_group.groups.items() for a in np.array_split(indices, 1)]

    # Process each chunk to find nearest neighbors
    tissues = [get_windows(job, n_neighbors) for job in tissue_chunks]

    # Store results in a dictionary
    out_dict = {}

    for k in ks:
        for neighbors, job in zip(tissues, tissue_chunks):
            chunk = np.arange(len(neighbors))
            tissue_name = job[2]
            indices = job[3]
            window = values[neighbors[chunk, :k].flatten()].reshape(len(chunk), k, len(sum_cols)).sum(axis=1)
            out_dict[(tissue_name, k)] = (window.astype(np.float16), indices)

    windows = {}
    for k in ks:
        window = pd.concat([pd.DataFrame(out_dict[(exp, k)][0], index=out_dict[(exp, k)][1].astype(int), columns=sum_cols) for exp in exps], axis=0)
        window = window.loc[adata.obs.index.values]
        window = pd.concat([adata.obs[['filename', 'Cell_Type']], window], axis=1)
        windows[k] = window
    return windows

# @param adata: AnnData object containing cell data
def centroid_Calculation(adata): 
    # Get neighborhood windows from KNN function
    windows = KNN(adata)

    k = 10  # Choose the number of neighbors to analyze
    windows2 = windows[k]
    cluster_col = 'Cell_Type'
    windows2[cluster_col] = adata.obs[cluster_col]
    copy_cells = adata.obs.copy()

    filtered_cells = copy_cells
    filtered_cells.reset_index(inplace=True, drop=True)

    cell_type_columns = adata.obs['Cell_Type'].unique()
    windows2[cell_type_columns] = windows2[cell_type_columns].astype('float32')

    neighborhoods_to_loop = adata.obs['Neighborhood'].unique()
    all_results = []

    for neighborhood in neighborhoods_to_loop:
        filtered_neighborhood_df = filtered_cells[filtered_cells['Neighborhood'] == neighborhood]
        cell_numbers_in_neighborhood = filtered_neighborhood_df.index.values
        matching_cells_df = windows2[windows2.index.isin(cell_numbers_in_neighborhood)]

        mean_std_results = {'Neighborhood': neighborhood}

        for column in cell_type_columns:
            if column in matching_cells_df.columns:
                column_mean = matching_cells_df[column].mean()
                column_std = matching_cells_df[column].std()
                mean_std_results[f'{column}_mean'] = column_mean
                mean_std_results[f'{column}_std'] = column_std

        all_results.append(mean_std_results)

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(all_results)
    return results_df
