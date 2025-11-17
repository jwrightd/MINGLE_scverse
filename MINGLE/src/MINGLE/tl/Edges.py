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

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from itertools import combinations
from tqdm import tqdm

# @param filePath: filePath to .csv or .h5ad containing cell information
def read_file(filePath):
    # For CSV:
    # df = pd.read_csv(filePath)
    # For h5ad (AnnData):
    return ad.read(filePath)

# @param GMM_adata: AnnData object containing GMM results, 
#        cell_adata: AnnData object containing raw annotated dataset
def mergeGMM(GMM_adata, cell_adata):
    # Merge AnnData objects based on their observations (cells)
    return GMM_adata.concatenate(cell_adata, join='outer', axis=1)

# @param adata: AnnData object containing cell data (cell annotations in .obs)
def findPositives(adata):
    # List of columns to use from .obs
    neighborhoods_to_loop = adata.obs['Neighborhood'].unique().tolist()
    neighborhoods_to_loop.extend(['Neighborhood', 'Cell_Type', 'filename', 'x', 'y'])

    # Subset the obs dataframe with relevant columns
    df_probabilities = adata.obs[neighborhoods_to_loop]

    # Set threshold
    threshold = 0.25

    # Initialize tqdm progress bar
    with tqdm(total=df_probabilities.shape[0]) as pbar:
        counts = []
        for _, row in df_probabilities.iterrows():
            # Count how many neighborhoods have a probability above the threshold
            count = (row > threshold).sum()
            counts.append(count)
            pbar.update(1)  # Update the progress bar by 1 step after processing each row

    # Add the 'Count_Above_Threshold' as a new column in .obs
    adata.obs['Count_Above_Threshold'] = counts

    return adata