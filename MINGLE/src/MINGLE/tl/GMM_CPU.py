#GMM File Includes:
#1. Read-in the file (merged df)
#2. Eliminated hardcoded items 
#3. KNN & get_windows function
#4. CPU based probability function

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

# For users running CPU parallel processing version
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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
def read_file(filePath):
    df = pd.read_csv(filePath)
    adata = ad.AnnData(X=np.zeros((len(df), 1)), obs=df)
    return adata

def KNN(cells_adata):
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
         

    # cells is now expected to be an AnnData object
    obs = cells.obs.copy()

    # Define column names that will be used for neighborhood analysis
    X = X_COL                 # Variable for the X coordinate
    Y = Y_COL                 # Variable for the Y coordinate
    reg = REGION_COL          # Variable for the filename or region identifier associated with coordinates
    cluster_col = CLUSTER_COL  # Variable for cell type/subtype classification

    # List of columns to keep for analysis
    keep_cols = [X, Y, reg, cluster_col]

    # Concatenate the original 'cells' DataFrame with dummy variables created from 'cluster_col'
    # pd.get_dummies() converts categorical variable(s) into dummy/indicator variables
    obs = pd.concat([obs, pd.get_dummies(obs[cluster_col])], axis=1)

    # Save cell type dummy names to AnnData for later use
    cells.obsm["celltype_matrix"] = obs[pd.get_dummies(obs[cluster_col]).columns].values
    cells.uns["cell_type_features"] = list(pd.get_dummies(obs[cluster_col]).columns)

    # Get unique values from the 'cluster_col' column to use for summarization
    sum_cols = obs[cluster_col].unique()

    # Retrieve the values for these unique categories as a NumPy array
    # This array can be used for further analysis or operations later for calculating the neighborhoods
    values = obs[sum_cols].values

    #KNN Set-up
    #We can choose a range of nearest neighbors to calculate the neighborhoods
    ks = [5,10,20] # k=5 means it collects 5 nearest neighbors for each center cell
    n_neighbors = max(ks) #sets n_neighbors to max of the list that is set

    # Group the cell data by region
    # 'cells' is a DataFrame containing cell data
    # 'tissue_group' will be a GroupBy object with cells grouped by the 'reg' column (representing regions)
    tissue_group = obs[[X, Y, reg]].groupby(reg)

    # Get a list of unique regions (filenames)
    # 'exps' will contain all unique region names found in the 'reg' column of the 'cells' DataFrame
    exps = list(obs[reg].unique())

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
        window = window.loc[obs.index.values]

        # Concatenate the window data with the original columns specified in 'keep_cols'
        window = pd.concat([obs[keep_cols], window], axis=1)

        # Store the concatenated DataFrame in the 'windows' dictionary, keyed by the current value of k
        windows[k] = window
    return windows


# Reads  Centroids file and returns a pandas DataFrame; in this case centroid data
def read_centroids_file(filePath):
    df_centroids = pd.read_csv(filePath)
    return df_centroids


#Parameters are the cells dataframe and the centroids dataframe
def cpu_gmm_probability(cells, df_centroids):
    cluster_col = CLUSTER_COL
    # Parameter for cell analysis
    windows = KNN(cells)
    #Choose k value to analyze and pull out from dictionary of stored results of vector
    k = 10
    windows2 = windows[k]
    #Add cell type column to output windows dataframe
    windows2[cluster_col] = cells.obs[cluster_col].values

    # List of neighborhoods to loop through, update this for other datasets
    neighborhoods_to_loop = cells.obs[NEIGHBORHOOD_COL].unique()

    # List of cell types, update this for other datasets
    cell_type_features =  cells.obs[cluster_col].unique()

    # Function to calculate probabilities for a single cell
    def calculate_probabilities_for_cell(args):
        cell_data, df_centroids, cell_type_features = args  # Unpack the arguments

        neighborhood_probs = {}

        # Loop through each neighborhood (each row in df_centroids)
        for _, centroid_row in df_centroids.iterrows():
            neighborhood_name = centroid_row['Neighborhood']
            total_prob = 1

            # For each cell type in the list, calculate the probability
            for cell_type in cell_type_features:
                mean_col = f'{cell_type}_mean'
                std_col = f'{cell_type}_std'

                if mean_col in centroid_row and std_col in centroid_row:
                    mean = centroid_row[mean_col]
                    std = centroid_row[std_col]

                    # Get the value of the current cell for this cell type (nearest neighbor count)
                    cell_value = cell_data.get(cell_type, np.nan)

                    # If std is zero, check if the cell value matches the mean
                    if std == 0:
                        if cell_value == mean:
                            cell_prob = 1
                        else:
                            cell_prob = 0
                    else:
                        # Otherwise, calculate the probability using the normal distribution (PDF)
                        cell_prob = norm.pdf(cell_value, loc=mean, scale=std)

                    # Multiply the probability for this cell type to the total probability
                    total_prob *= cell_prob

            # Store the total probability for the current neighborhood
            neighborhood_probs[neighborhood_name] = total_prob

        # Normalize the probabilities across all neighborhoods so they sum to 1
        total_prob_sum = sum(neighborhood_probs.values())
        for neighborhood in neighborhood_probs:
            neighborhood_probs[neighborhood] /= total_prob_sum

        return neighborhood_probs

    # Function to parallelize the calculations across all cells
    def parallelize_probability_calculations(windows2, df_centroids, cell_type_features):
        # Use all available CPUs for parallel processing
        num_processes = cpu_count()
        # Create a pool of workers
        with Pool(num_processes) as pool:
            # Use tqdm to display progress bar
            results = list(tqdm(pool.imap(
                calculate_probabilities_for_cell,  # Pass the function directly
                [(windows2.loc[cell_index], df_centroids, cell_type_features) for cell_index in windows2.index]), total=len(windows2)))

        return results

    # Call the parallelization function
    #probabilities_list = []
    #for cell_index in windows2.index:
    #    probabilities_list.append(calculate_probabilities_for_cell((windows2.loc[cell_index], df_centroids, cell_type_features)))
    #    print(cell_index)
    probabilities_list = parallelize_probability_calculations(windows2, df_centroids, cell_type_features)

    # Convert the results into a DataFrame
    probabilities_df = pd.DataFrame(probabilities_list, index=windows2.index)

    # Attach to AnnData for scverse compatibility
    cells.obsm["neighborhood_probabilities"] = probabilities_df.values
    cells.uns["neighborhood_probability_neighborhoods"] = list(probabilities_df.columns)

    return probabilities_df