# --- scverse-compatible imports ---
import numpy as np
import pandas as pd
import anndata as ad

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans


# -----------------------------
# 1) Build AnnData from your df
# -----------------------------
probabilities_df = pd.read_csv(r"/Volumes/data/MINGLE/Data/Intestine/intestine_tissueunit_all_cells_all_neighborhood_probs.csv")
allinfo_df = pd.read_csv(r"/Volumes/data/MINGLE/Data/Intestine/intestine_all_information_2.csv")

tu1 = "Inner Follicle"
tu2 = "Outer Follicle"

extra_cols = ['x','y','unique_region','Neighborhood','Cell Type','Tissue Unit','Community']
df2 = allinfo_df[[tu1, tu2] + extra_cols].copy()

# Score construction (unchanged)
eps = 1e-9
df2["ratio"] = (df2[tu1] + eps) / (df2[tu2] + eps)
df2["log_ratio"] = np.log(df2["ratio"])
df2["max_prob"] = df2[[tu1, tu2]].max(axis=1)
df2["Score"] = df2["log_ratio"] * df2["max_prob"]

def assign_probability_level_with_edges(series, n_bins=5, labels=None, use_quantiles=True):
    if labels is None:
        labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    labels = list(labels)

    s = series.dropna()
    vmin = float(s.min())
    vmax = float(s.max())

    if np.isclose(vmin, vmax):
        cat = pd.Series(["Medium"] * len(series), index=series.index)
        return cat.astype(pd.CategoricalDtype(categories=labels, ordered=True)), np.array([vmin, vmax])

    if use_quantiles:
        try:
            cat = pd.qcut(series, q=n_bins, labels=labels, duplicates="raise")
            edges = series.quantile(np.linspace(0,1,n_bins+1)).values
            return cat, edges
        except ValueError:
            pass

    edges = np.linspace(vmin, vmax, n_bins + 1)
    unique_edges = np.unique(edges)
    labels_use = labels[: len(unique_edges) - 1]
    cat = pd.cut(series, bins=unique_edges, labels=labels_use, include_lowest=True)
    return cat, unique_edges

df2["Probability_Level"], edges_used = assign_probability_level_with_edges(df2["Score"], n_bins=5)

# Create AnnData (X can be empty; we’re using obs + obsm)
adata = ad.AnnData(X=np.zeros((df2.shape[0], 1), dtype=np.float32))
adata.obs = df2.drop(columns=["x","y"]).copy()
adata.obsm["spatial"] = df2[["x","y"]].to_numpy(dtype=np.float32)

# Make obs_names stable/unique (recommended)
adata.obs_names = pd.Index([f"cell_{i}" for i in range(adata.n_obs)], dtype=str)


# ---------------------------------------------------------
# 2) AnnData-native Neighborhood windows (scverse-friendly)
# ---------------------------------------------------------
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class NeighborhoodsAnnData:
    def __init__(self,
                 adata,
                 ks,
                 cluster_col,
                 keep_obs_cols,
                 reg_key="unique_region",
                 spatial_key="spatial",
                 add_dummies=True,
                 out_obsm_prefix="windows_k"):

        self.adata = adata
        self.ks = list(ks)
        self.cluster_col = cluster_col
        self.keep_obs_cols = list(keep_obs_cols)
        self.reg_key = reg_key
        self.spatial_key = spatial_key
        self.out_obsm_prefix = out_obsm_prefix

        self.n_neighbors = max(self.ks)
        self.exps = list(self.adata.obs[self.reg_key].astype(str).unique())

        # ✅ rename flag so it doesn't shadow the method
        self.bool_add_dummies = bool(add_dummies)

    def add_dummies(self):
        dumz = pd.get_dummies(self.adata.obs[self.cluster_col], dtype=int)
        keep = self.adata.obs[self.keep_obs_cols]
        self.cells = pd.concat([keep, dumz], axis=1)

    def get_tissue_chunks(self):
        reg = self.adata.obs[self.reg_key].astype(str)
        tissue_groups = {t: np.where(reg.values == t)[0] for t in self.exps}
        tissue_chunks = [(time.time(), self.exps.index(t), t, tissue_groups[t]) for t in self.exps]
        return tissue_chunks

    def make_windows(self, job):
        start_time, idx, tissue_name, indices = job
        job_start = time.time()

        print("Starting:", str(idx+1)+'/'+str(len(self.exps)), ': ' + self.exps[idx])

        coords = self.adata.obsm[self.spatial_key][indices]
        fit = NearestNeighbors(n_neighbors=self.n_neighbors + 1).fit(coords)
        dists, inds = fit.kneighbors(coords)

        # drop self
        dists = dists[:, 1:]
        inds = inds[:, 1:]

        # sort neighbors (keep your original intent)
        args = dists.argsort(axis=1)
        add = np.arange(inds.shape[0])[:, None]
        sorted_indices = inds[add, args]
        sorted_dists = dists[add, args]

        neighbors = indices[sorted_indices].astype(np.int32)

        end_time = time.time()
        print("Finishing:", str(idx+1)+'/'+str(len(self.exps)), ": " + self.exps[idx],
              end_time - job_start, end_time - start_time)

        return neighbors, sorted_dists

    def k_windows(self, distance_max="none"):
        if self.bool_add_dummies:
            self.add_dummies()
        else:
            self.cells = self.adata.obs.copy()

        sum_cols = list(pd.get_dummies(self.adata.obs[self.cluster_col]).columns)
        values = self.cells[sum_cols].values

        tissue_chunks = self.get_tissue_chunks()
        tissues = [self.make_windows(job) for job in tissue_chunks]

        out_dict = {}
        if distance_max == "none":
            k_dists = {k: np.inf for k in self.ks}
        else:
            k_dists = {k: distance_max for k in self.ks}

        # original prints
        print("k_dists:", k_dists, "Type:", type(k_dists))
        print("distance_max:", distance_max, "Type:", type(distance_max))

        for k in self.ks:
            for (neighbors, sorted_dists), job in zip(tissues, tissue_chunks):
                chunk = np.arange(len(neighbors))
                tissue_name = job[2]
                indices = job[3]

                window = values[neighbors[:, :k].flatten()].reshape(len(chunk), k, len(sum_cols))

                mask = sorted_dists[:, :k] > k_dists[k]
                no_cells_masked = mask.sum(1)
                avg_cells_excluded = no_cells_masked[no_cells_masked != 0]
                avg_cells_excluded = avg_cells_excluded.mean() if len(avg_cells_excluded) > 0 else 0

                print('{}:{}--{}/{} cells had cells excluded.  Avg excluded={}'.format(
                    tissue_name, k, (no_cells_masked > 0).sum(), len(no_cells_masked), avg_cells_excluded
                ))

                mask3 = np.repeat(mask[:, :, np.newaxis], len(sum_cols), axis=2)
                masked_window = np.ma.array(window, mask=mask3)
                summed = masked_window.sum(1).data

                out_dict[(tissue_name, k)] = (summed.astype(np.float16), indices)

        windows = {}
        n_total = self.adata.n_obs

        for k in self.ks:
            # Build a full matrix in the ORIGINAL cell order (0..n-1 positions)
            full = np.zeros((n_total, len(sum_cols)), dtype=np.float16)

            for exp in self.exps:
                mat, idxs = out_dict[(exp, k)]          # mat shape: (#cells_in_exp, #sum_cols)
                full[idxs, :] = mat                     # put back into global positions

            # Make DF indexed by obs_names so it is scverse-safe
            win_counts = pd.DataFrame(full, index=self.adata.obs_names, columns=sum_cols)

            # Add keep columns (also indexed by obs_names)
            win = pd.concat([self.adata.obs[self.keep_obs_cols], win_counts], axis=1)

            windows[k] = win

            # Store into AnnData
            self.adata.obsm[f"{self.out_obsm_prefix}{k}"] = full.astype(np.float32)
            self.adata.uns[f"{self.out_obsm_prefix}{k}_cols"] = sum_cols

        return windows





# ------------------------------------------
# 3) Run windows + subset clustering (same)
# ------------------------------------------
ks = [10,20,30,50,100]
k = 20
clusters = 5
distance_max = 1000

# columns to keep in obs alongside windows
keep_obs_cols = ['Neighborhood','Cell Type','Community','Tissue Unit','Probability_Level','unique_region']

Neigh = NeighborhoodsAnnData(
    adata=adata,
    ks=ks,
    cluster_col="Probability_Level",
    keep_obs_cols=keep_obs_cols,
    reg_key="unique_region",
    spatial_key="spatial",
    add_dummies=True,
    out_obsm_prefix="windows_k"
)

cluster_name_windows = Neigh.k_windows(distance_max=distance_max)
windows2 = cluster_name_windows[k]  # DataFrame: obs cols + dummy-count sums

# dummy feature columns (exactly the dummies used inside windows)
sum_cols2 = adata.uns[f"windows_k{k}_cols"]

# subset BEFORE clustering (same logic, now on adata.obs)
target_neighborhoods = ['Inner Follicle','Outer Follicle']
mask = adata.obs['Neighborhood'].isin(target_neighborhoods)

# feature matrix from windows (preferred)
feature_matrix = windows2.loc[mask, sum_cols2].values

km = MiniBatchKMeans(n_clusters=clusters, random_state=0)
labels_sub = km.fit_predict(feature_matrix)

neighborhood_name = f"neighborhood{k}"

# store labels in adata.obs
adata.obs[neighborhood_name] = -1
adata.obs.loc[mask, neighborhood_name] = labels_sub

# centroids
niche_clusters = km.cluster_centers_

# baseline avgs (subset baseline like you did)
values_sub = adata.obs.loc[mask, "Probability_Level"]
# but we need dummy baseline: use windows2 subset dummy columns as baseline
tissue_avgs = windows2.loc[mask, sum_cols2].values.mean(axis=0)

# fold change (same math)
eps = 1e-12
tissue_row = tissue_avgs.reshape(1, -1)
niche_plus = niche_clusters + tissue_row
row_sums = niche_plus.sum(axis=1, keepdims=True)
norm = niche_plus / (row_sums + eps)
fc_array = np.log2((norm + eps) / (tissue_row + eps))
fc = pd.DataFrame(fc_array, columns=sum_cols2)

# map cluster ids to strings (your mapping)
n_conversion_20 = {0:'0',1:'1',2:'2',3:'3',4:'4'}
adata.obs["Probability_Bin_Cluster"] = adata.obs[neighborhood_name].map(n_conversion_20).astype("category")

print(adata)
print(adata.obs[[neighborhood_name, "Probability_Bin_Cluster"]].head())
