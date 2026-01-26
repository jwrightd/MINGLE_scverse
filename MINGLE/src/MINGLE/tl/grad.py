#Import Packages
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import sys



class Neighborhoods(object):
    def __init__(self, cells,ks,cluster_col,sum_cols,keep_cols,neigh,X='X:X',Y = 'Y:Y',reg = 'Exp',add_dummies = True):
        self.cells_nodumz = cells
        self.X = X
        self.Y = Y
        self.reg = reg
        self.keep_cols = keep_cols
        self.sum_cols = sum_cols
        self.ks = ks
        self.cluster_col = cluster_col
        self.n_neighbors = max(ks)
        self.exps = list(self.cells_nodumz[self.reg].unique())
        self.bool_add_dummies = add_dummies
        self.neigh = neigh
        
    def add_dummies(self):
        
        c = self.cells_nodumz
        dumz = pd.get_dummies(c[self.cluster_col], dtype=int)
        keep = c[self.keep_cols]
        
        self.cells = pd.concat([keep,dumz],axis = 1)
        
        
        
    def get_tissue_chunks(self):
        self.tissue_group = self.cells[[self.X,self.Y,self.reg]].groupby(self.reg)
        
        tissue_chunks = [(time.time(),self.exps.index(t),t,a) for t,indices in self.tissue_group.groups.items() for a in np.array_split(indices,1)] 
        return tissue_chunks
    
    def make_windows(self,job):
        

        start_time,idx,tissue_name,indices = job
        job_start = time.time()

        print ("Starting:", str(idx+1)+'/'+str(len(self.exps)),': ' + self.exps[idx])

        tissue = self.tissue_group.get_group(tissue_name)
        to_fit = tissue.loc[indices][[self.X,self.Y]].values

        fit = NearestNeighbors(n_neighbors=self.n_neighbors+1).fit(tissue[[self.X,self.Y]].values)
        m = fit.kneighbors(to_fit)
        
        #don't include index cell in window (can always easily add to windows again later by just adding 1)
        m = m[0][:,1:], m[1][:,1:]


        #sort_neighbors
        args = m[0].argsort(axis = 1)
        add = np.arange(m[1].shape[0])*m[1].shape[1]
        sorted_indices = m[1].flatten()[args+add[:,None]]
        sorted_dists = m[0].flatten()[args+add[:,None]]
        neighbors = tissue.index.values[sorted_indices].astype(np.int32)
        end_time = time.time()

        print ("Finishing:", str(idx+1)+"/"+str(len(self.exps)),": "+ self.exps[idx],end_time-job_start,end_time-start_time)
        return neighbors,sorted_dists
    
    def k_windows(self,distance_max = 'none'):
        if self.bool_add_dummies:
            self.add_dummies()
        else:
            self.cells =self.cells_nodumz
        sum_cols = list(self.sum_cols)
        for col in sum_cols:
            if col in self.keep_cols:
                self.cells[col+'_sum'] = self.cells[col]
                self.sum_cols.remove(col)
                self.sum_cols+=[col+'_sum']

        values = self.cells[self.sum_cols].values
        tissue_chunks = self.get_tissue_chunks()
        tissues = [self.make_windows(job) for job in tissue_chunks]
        
        out_dict = {}
        if distance_max == 'none':
            k_dists = {k:np.inf for k in self.ks}
        else:
            k_dists = {k: distance_max for k in self.ks}

        print("k_dists:", k_dists, "Type:", type(k_dists))
        print("distance_max:", distance_max, "Type:", type(distance_max))

        for k in self.ks:
            for (neighbors,sorted_dists),job in zip(tissues,tissue_chunks):
                chunk = np.arange(len(neighbors))#indices
                tissue_name = job[2]
                indices = job[3]
                window = values[neighbors[chunk,:k].flatten()].reshape(len(chunk),k,len(self.sum_cols))

                mask = sorted_dists>k_dists[k]
                no_cells_masked = mask.sum(1)
                avg_cells_excluded = no_cells_masked[no_cells_masked!=0]
                if len(avg_cells_excluded)>0:
                    avg_cells_excluded = avg_cells_excluded.mean()
                else:
                    avg_cells_excluded = 0
                print ('{}:{}--{}/{} cells had cells excluded.  Avg excluded={}'.format(tissue_name,k,(no_cells_masked>0).sum(),len(no_cells_masked),avg_cells_excluded))

                mask = np.repeat(mask[:, :k, np.newaxis], len(sum_cols), axis=2)# don't sum cells that are outside max_distance
                masked_window = np.ma.array(window,mask = mask)
                summed = masked_window.sum(1).data

                out_dict[(tissue_name,k)] = (summed.astype(np.float16),indices)
        
        windows = {}
        for k in self.ks:

            window = pd.concat([pd.DataFrame(out_dict[(exp,k)][0],index = out_dict[(exp,k)][1].astype(int),columns = self.sum_cols) for exp in self.exps],axis=0)
            window = window.loc[self.cells.index.values]
            window = pd.concat([self.cells[self.keep_cols],window],axis=1)
            windows[k] = window
        return windows



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

probabilities_df = pd.read_csv(r"/Volumes/data/MINGLE/Data/Intestine/intestine_tissueunit_all_cells_all_neighborhood_probs.csv")
allinfo_df = pd.read_csv(r"/Volumes/data/MINGLE/Data/Intestine/intestine_all_information_2.csv")

# Define units of interest
tu1 = "Inner Follicle"
tu2 = "Outer Follicle"

# Additional columns needed downstream
extra_cols = ['x','y','unique_region','Neighborhood','Cell Type','Tissue Unit','Community']

# Create new dataframe
df2 = allinfo_df[[tu1, tu2] + extra_cols].copy()

import numpy as np

# ratio with small constant to avoid division by zero
eps = 1e-9
df2["ratio"] = (df2[tu1] + eps) / (df2[tu2] + eps)

# log ratio
df2["log_ratio"] = np.log(df2["ratio"])

# max probability
df2["max_prob"] = df2[[tu1, tu2]].max(axis=1)

# final score
df2["Score"] = df2["log_ratio"] * df2["max_prob"]
import numpy as np
import pandas as pd

def assign_probability_level_with_edges(series,
                                        n_bins=5,
                                        labels=None,
                                        use_quantiles=True):
    """
    Same as before, but now RETURNS:
      - categories (pd.Categorical)
      - bin edges actually used (np.ndarray)
    and PRINTS them clearly.
    """
    if labels is None:
        labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    labels = list(labels)

    s = series.dropna()
    vmin = float(s.min())
    vmax = float(s.max())

    # Case: constant series
    if np.isclose(vmin, vmax):
        print("All Score values are identical → single bin.")
        print("Bin edges = [vmin, vmax] =", [vmin, vmax])
        cat = pd.Series(["Medium"] * len(series), index=series.index)
        return cat.astype(pd.CategoricalDtype(categories=labels, ordered=True)), np.array([vmin, vmax])

    # === Try quantiles first ===
    if use_quantiles:
        try:
            cat = pd.qcut(series, q=n_bins, labels=labels, duplicates="raise")
            # If this succeeded, we can extract quantile edges
            edges = series.quantile(np.linspace(0,1,n_bins+1)).values
            print("Used quantile bins:")
            print("Quantile edges:", edges)
            return cat, edges
        except ValueError:
            print("Quantile binning failed (duplicate edges). Falling back to equal-width bins.")

    # === Fallback: equal-width bins ===
    edges = np.linspace(vmin, vmax, n_bins + 1)

    # Ensure unique edges (should be unless precision issues)
    unique_edges = np.unique(edges)
    if len(unique_edges) < len(edges):
        print("Warning: equal-width bin edges collapsed due to precision. Unique edges =", unique_edges)
    else:
        print("Using equal-width bins:")
        print("Equal-width edges:", unique_edges)

    # Assign bins
    labels_use = labels[: len(unique_edges) - 1]
    cat = pd.cut(series, bins=unique_edges, labels=labels_use, include_lowest=True)

    return cat, unique_edges


# ------------------ Example on df2 ------------------
df2 = df2.copy()
df2["Probability_Level"], edges_used = assign_probability_level_with_edges(df2["Score"], n_bins=5)

print("\nFinal bin edges actually used:\n", edges_used)
print("\nCounts per level:\n", df2["Probability_Level"].value_counts(dropna=False))

df = df2.copy()
df.reset_index(inplace=True, drop=True)

# Define column names that will be used for neighborhood analysis
ks = [10,20,30,50,100]  # k=5 means it collects 5 nearest neighbors for each center cell
X = 'x'                  # Variable for the X coordinate
Y = 'y'                  # Variable for the Y coordinate
reg = 'unique_region'         # Variable for the filename or region identifier associated with coordinates
neigh = 'Neighborhood'      # Variable for the neighborhood assignemnt of the cell
cluster_col = 'Probability_Level'  # Variable for cell type/subtype classification
celltype = 'Cell Type'
tiss = 'Tissue Unit'
comm = 'Community'
sum_cols = list(df[cluster_col].unique())
# List of columns to keep for analysis
keep_cols = [X, Y, reg, neigh, celltype, comm, tiss, cluster_col]

#Run neighborhood analysis function with radial distance threshold
Neigh = Neighborhoods(df,ks = ks,cluster_col = cluster_col,
                      sum_cols=sum_cols,reg=reg,
                      keep_cols=keep_cols,neigh=neigh, X = X,Y=Y, add_dummies=True)
cluster_name_windows = Neigh.k_windows(distance_max=1000) #Distance threshold for cell neighborhoods in terms of pixels conservative of 100 um

# --- CONCATENATE DUMMIES (unchanged) ---
# Concatenate the original 'cells' DataFrame with dummy variables created from 'cluster_col'
# pd.get_dummies() converts categorical variable(s) into dummy/indicator variables
df = pd.concat([df, pd.get_dummies(df[cluster_col], dtype=int)], axis=1)

# --- FIX: use the dummy column names as the summary columns (was previously using unique category values) ---
sum_cols2 = pd.get_dummies(df[cluster_col], dtype=int).columns.tolist()

# Retrieve the values for these dummy categories as a NumPy array
values = df[sum_cols2].values

#Choose k value to analyze and pull out from dictionary of stored results of vectors
k = 20
clusters = 5
windows2 = cluster_name_windows[k]
#Add cell type column to output windows dataframe
windows2[cluster_col] = df[cluster_col]

#Fill in based on above for the number of clusters you want
n_neighborhoods = clusters

#return a name of the column for storing the clusters
neighborhood_name = "neighborhood"+str(k)

# Initialize a dictionary to store the centroids for each value of 'k'
k_centroids = {}

# ----------------- NEW: filter to target neighborhoods BEFORE clustering -----------------
target_neighborhoods = ['Inner Follicle','Outer Follicle']

# windows2 and df should be index-aligned; if not but same length, reset indices to align
if not windows2.index.equals(df.index):
    if len(windows2) == len(df):
        windows2 = windows2.reset_index(drop=True)
        df = df.reset_index(drop=True)
    else:
        # If lengths differ, user must align windows2 and df appropriately; raise explicit error
        raise ValueError("windows2 and df indices differ and lengths differ; align them before filtering.")

# create a boolean mask for the target neighborhoods
mask = df['Neighborhood'].isin(target_neighborhoods)

# filter windows2 and df to only the rows corresponding to the target neighborhoods
windows2_sub = windows2.loc[mask].copy()
df_sub = df.loc[mask].copy()

# Initialize a MiniBatchKMeans clustering model
# 'n_clusters' is set to 'n_neighborhoods', which is the desired number of clusters
# 'random_state=0' ensures reproducibility of the results
km = MiniBatchKMeans(n_clusters=n_neighborhoods, random_state=0)

# Prepare features to feed into KMeans. Prefer windows2_sub one-hot columns if available; otherwise use df_sub dummies
if set(sum_cols2).issubset(windows2_sub.columns):
    feature_matrix = windows2_sub[sum_cols2].values
else:
    feature_matrix = df_sub[sum_cols2].values

# Perform clustering on the data in the filtered windows2_sub using the columns specified in 'sum_cols2'
# '.values' converts the DataFrame to a NumPy array, which is the input format for KMeans
labels_sub = km.fit_predict(feature_matrix)

# Store the centroids of the clusters in the 'k_centroids' dictionary, keyed by 'k'
k_centroids[k] = km.cluster_centers_

# Add the cluster labels to the filtered dataframe (df_sub) only
df_sub[neighborhood_name] = labels_sub

# If you want to keep a full-length column in df, but avoid mismatches, fill with -1 for non-subset rows
df[neighborhood_name] = -1
df.loc[mask, neighborhood_name] = labels_sub

# Select the centroids for a specific value of 'k' for plotting
k_to_plot = k
niche_clusters = (k_centroids[k_to_plot])

# Calculate the average cell types across the subset values array (use subset so baseline is Inner+Outer Follicle)
values_sub = df_sub[sum_cols2].values
tissue_avgs = values_sub.mean(axis=0)

# Compute fold change (fc) of cell types abundance within a neighborhood versus that in the tissue
# This involves a log2 transformation of the ratio of (niche_clusters + tissue_avgs) to tissue_avgs
# The ratio is normalized by the sum across each row (axis=1), ensuring that the sum of ratios for each row is 1
eps = 1e-12
tissue_row = tissue_avgs.reshape(1, -1)
niche_plus = niche_clusters + tissue_row
row_sums = niche_plus.sum(axis=1, keepdims=True)
norm = niche_plus / (row_sums + eps)
fc_array = np.log2((norm + eps) / (tissue_row + eps))

# Convert the fold change array into a pandas DataFrame for each cell type
fc = pd.DataFrame(fc_array, columns=sum_cols2)

# --------------------------
# *** MINIMAL CHANGE: ORDER ROWS (by high-bin enrichment) AND COLUMNS (highest->lowest) DYNAMICALLY ***
# --------------------------

import re

def col_score(lbl):
    s = str(lbl).strip().lower()

    # specific exact/phrase matches first (VERY important: very-low/high before low/high)
    if s == 'very high' or 'very high' in s:
        return 10000.0
    if s == 'high' or s.endswith(' high') or s.startswith('high'):
        return 8000.0
    if s == 'medium' or 'medium' in s or s == 'med':
        return 5000.0
    if s == 'low' and 'very' not in s:   # exact low (not 'very low')
        return 2000.0
    if s == 'very low' or 'very low' in s:
        return 100.0

    # Bin_# or Bin #
    m = re.search(r'bin[_\s]*(\d+)', s)
    if m:
        return float(m.group(1))

    # numeric range like 0.0-0.1 or 0.0 – 0.1
    m2 = re.search(r'([0-9]*\.?[0-9]+)\s*[-–—]\s*([0-9]*\.?[0-9]+)', s)
    if m2:
        a = float(m2.group(1)); b = float(m2.group(2))
        return (a + b) / 2.0

    # single float like '0.75'
    m3 = re.search(r'^\d*\.?\d+$', s)
    if m3:
        return float(s)

    # trailing digit fallback
    m4 = re.search(r'(\d+)$', s)
    if m4:
        return float(m4.group(1))

    # final fallback: very low (lowest)
    return -9999.0

# compute desired column order: sort fc.columns by col_score descending -> highest prob left
desired_col_order = sorted(list(fc.columns), key=lambda c: -col_score(c))

# fallback: if all scores identical, reverse sum_cols2
if len(set(col_score(c) for c in fc.columns)) == 1:
    desired_col_order = list(sum_cols2[::-1])

# pick the high bin automatically as the first column in the desired_col_order
high_bin = desired_col_order[0]

# compute row order by descending enrichment for the highest-probability bin
ordered_rows = fc[high_bin].sort_values(ascending=False).index.tolist()

# reindex fc to enforce row and column ordering
fc = fc.reindex(index=ordered_rows, columns=desired_col_order)


n_conversion_20 = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',

}
df_sub['Probability_Bin_Cluster']=df_sub[neighborhood_name].map(n_conversion_20)
print(df_sub)
print(df_sub['Probability_Bin_Cluster'])
print(df_sub['neighborhood20'])