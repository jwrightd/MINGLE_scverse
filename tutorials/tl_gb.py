#Import Packages
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

df = pd.read_csv(r"Z:\MINGLE\Data\Intestine\intestine_all_information_2.csv")
df_probabilities = pd.read_csv(r"Z:\MINGLE\Data\Intestine\all_cells_all_neighborhood_probs")
total_df = pd.read_csv(r"Z:\MINGLE\Data\Intestine\20251217_intestine_inner_outerfollicle_probabilitybincluster_2probbins.csv")
loaded_palette_hex = {0: '#e41a1c',
 1: '#377eb8',
 2: '#4daf4a',
 3: '#984ea3',
 4: '#ff7f00',
 5: '#ffff33',
 6: '#a65628',
 7: '#f781bf',
 8: '#999999',
 9: '#e41a1c'}

