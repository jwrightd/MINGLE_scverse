#GMM File Includes:
#1. Read-in the file (merged df)
#2. Eliminated hardcoded items 
#3. KNN & get_windows function
#4. CPU based probability function

#TODO:add plotting functions, clean up code,  
import anndata as ad
import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Optional, Union, Dict, Sequence

from sklearn.neighbors import NearestNeighbors

# For users running CPU parallel processing version
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from .knn import KNN

"""
# Coordinates
X_COL = "x"          # column name for X coordinate
Y_COL = "y"          # column name for Y coordinate

# Region or filename to group images
REGION_COL = "unique_region"

# Cell type annotation
CLUSTER_COL = "cell_type"

# Neighborhood annotation
NEIGHBORHOOD_COL = "neighborhood"

# Global variables to hold loaded data
CELLS_ADATA = None
CENTROIDS_ADATA = None


def read_file(cells_path, centroids_path):

    global CELLS_ADATA, CENTROIDS_ADATA

    # Load cells 
    df_cells = pd.read_csv(cells_path).reset_index(drop=True)
    CELLS_ADATA = ad.AnnData(
        X=np.zeros((len(df_cells), 1)),
        obs=df_cells
    )

    # Load centroids 
    df_centroids = pd.read_csv(centroids_path).reset_index(drop=True)
    CENTROIDS_ADATA = ad.AnnData(
        X=np.zeros((len(df_centroids), 1)),
        obs=df_centroids
    )

    return CELLS_ADATA, CENTROIDS_ADATA
"""

#Parameters are the cells dataframe and the centroids dataframe
def cpu_gmm_probability(
    CELLS_ADATA: ad.AnnData,
    CENTROIDS_ADATA: ad.AnnData,
    *,
    cluster_col: str = "cell_type",  # Default cluster column in obs
    neighborhood_col: str = "neighborhood",  # Default neighborhood column in obs
    ks: Sequence[int] = (5, 10, 20),  # List of k values for neighbors
    threshold: float = 0.25,  # Probability threshold for counting
    num_processes: Optional[int] = None,  # Optional: number of processes for parallelism (defaults to max CPUs)
) -> pd.DataFrame:
    """
    Calculate GMM probabilities for each cell's assigned neighborhood.

    Parameters
    ----------
    CELLS_ADATA
        AnnData object containing cell-level data (with cluster labels and coordinates).
    CENTROIDS_ADATA
        AnnData object containing centroid data with neighborhood means and standard deviations for each cell type.
    cluster_col
        Column in `CELLS_ADATA.obs` representing the cluster or cell type (default is 'cell_type').
    neighborhood_col
        Column in `CELLS_ADATA.obs` representing the neighborhood assignment (default is 'neighborhood').
    ks
        Sequence of values for k (the number of neighbors) to compute neighborhood summaries.
    threshold
        Threshold probability value to count neighborhoods (default is 0.25).
    num_processes
        Number of parallel processes to use for computation (default is None, which uses all available CPUs).

    Returns
    -------
    probabilities_df
        DataFrame with probabilities for each cell and neighborhood.
    """

    # Ensure neighborhood columns exist in obs
    if neighborhood_col not in CELLS_ADATA.obs or cluster_col not in CELLS_ADATA.obs:
        raise KeyError(f"One or more required columns ({neighborhood_col}, {cluster_col}) are missing in obs.")

    # Step 1: Get KNN neighborhood windows
    windows = KNN(CELLS_ADATA, ks=ks)
    k = 10  # You can change this if needed, default is 10
    windows2 = windows[k]
    windows2[cluster_col] = CELLS_ADATA.obs[cluster_col].values

    # Step 2: List of neighborhoods and cell types to loop through
    neighborhoods_to_loop = CELLS_ADATA.obs[neighborhood_col].unique().tolist()
    cell_type_features = CELLS_ADATA.obs[cluster_col].unique()

    # Function to calculate probabilities for a single cell
    def calculate_probabilities_for_cell(args):
        cell_index, windows2, CELLS_ADATA, CENTROIDS_ADATA, cell_type_features = args
        neighborhood_probs = {}

        # Iterate through each centroid (neighborhood)
        for _, centroid_row in CENTROIDS_ADATA.obs.iterrows():
            neighborhood_name = centroid_row[neighborhood_col]
            total_prob = 1

            # For each cell type, calculate the probability
            for cell_type in cell_type_features:
                mean_col = f'{cell_type}_mean'
                std_col = f'{cell_type}_std'

                if mean_col in centroid_row and std_col in centroid_row:
                    mean = centroid_row[mean_col]
                    std = centroid_row[std_col]

                    # Get the value of the current cell for this cell type
                    cell_value = windows2.loc[cell_index, cell_type] if cell_type in windows2.columns else np.nan

                    # Check if std is zero, calculate probability
                    if std == 0:
                        cell_prob = 1 if cell_value == mean else 0
                    else:
                        cell_prob = norm.pdf(cell_value, loc=mean, scale=std)

                    total_prob *= cell_prob  # Multiply for each cell type

            # Store the neighborhood probability for this cell
            neighborhood_probs[neighborhood_name] = total_prob

        # Normalize the probabilities to sum to 1
        total_prob_sum = sum(neighborhood_probs.values())
        for neighborhood in neighborhood_probs:
            neighborhood_probs[neighborhood] /= total_prob_sum

        return neighborhood_probs

    # Function to parallelize calculations across all cells
    def parallelize_probability_calculations(windows2, CELLS_ADATA, CENTROIDS_ADATA, cell_type_features, num_processes):
        # Use specified number of processes or default to all available CPUs
        if num_processes is None:
            num_processes = cpu_count()

        with Pool(num_processes) as pool:
            # Use tqdm for progress bar
            results = list(tqdm(pool.imap(
                calculate_probabilities_for_cell,
                [(cell_index, windows2, CELLS_ADATA, CENTROIDS_ADATA, cell_type_features) for cell_index in windows2.index]
            ), total=len(windows2)))
        
        return results

    # Parallelize the calculations
    probabilities_list = parallelize_probability_calculations(windows2, CELLS_ADATA, CENTROIDS_ADATA, cell_type_features, num_processes)

    # Convert the results into a DataFrame
    probabilities_df = pd.DataFrame(probabilities_list, index=windows2.index)

    # Attach to AnnData for scverse compatibility
    CELLS_ADATA.obsm["neighborhood_probabilities"] = probabilities_df.values
    CELLS_ADATA.uns["neighborhood_probability_neighborhoods"] = list(probabilities_df.columns)

    return probabilities_df
