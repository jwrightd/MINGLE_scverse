import MINGLE as mg
import pandas as pd
import anndata as ad   
file_path = r"/Volumes/data/MINGLE/Data/Intestine/05_25_huBMAP_tunit.csv"
    
adata = mg.pp.read_file(file_path)

X = "x"
Y = "y"
reg = "unique_region"
cluster_col = "Cell Type"

sum_cols = list(adata.obs[cluster_col].unique())
keep_cols = [X, Y, reg, cluster_col]

windows = mg.tl.KNN2(adata, cluster_col=cluster_col, keep_obs_cols=keep_cols)
k = 10
windows2 = windows[k]
windows2[cluster_col] = adata.obs[cluster_col].values

adata_windows = ad.AnnData(X=None, obs=windows2.copy())
adata_windows.obs_names = adata_windows.obs.index.astype(str)

adata_windows.obsm[f"knn_windows_k{k}"] = adata_windows.obs[sum_cols].astype("float32").values
adata_windows.uns["knn_windows"] = {
    "k": k,
    "cols": list(sum_cols),
    "source": "dummy-count neighborhood windows",
}

summary_df, per_cell_df = mg.tl.run_mingle_over_n_clusters(
    adata=adata_windows,          # <-- AnnData in, AnnData updated in-place
    knn_feature_cols=sum_cols,    # same as your call
    n_range=range(1, 51),
    return_per_cell=True,
    plot_summary=True,
    x_key="x",
    y_key="y",
    region_key="unique_region",
)

ll_idx, ll_n, _ = mg.tl.find_elbow_point(
    y_values=None,
    x_values=None,
    adata=adata_windows,
    uns_key="mingle_n_clusters",
    y_key="avg_log_likelihood",
    x_key="n_clusters",
)

prob_idx, prob_n, _ = mg.tl.find_elbow_point(
    y_values=None,
    x_values=None,
    adata=adata_windows,
    uns_key="mingle_n_clusters",
    y_key="avg_assigned_probability",
    x_key="n_clusters",
)

# Step 2: Constrained plateau search (pulled from adata.uns)
composite_df, best_n, ranked_plateaus = mg.tl.find_best_unsupervised_plateau(
    log_likelihoods=None,
    assigned_probs=None,
    elbow_min=min(ll_n, prob_n),
    elbow_max=max(ll_n, prob_n),
    adata=adata_windows,
    uns_key="mingle_n_clusters",
    ll_key="avg_log_likelihood",
    prob_key="avg_assigned_probability",
    out_uns_key="mingle_plateau_selection",  # optional; remove if you don't want storage
)

# Step 3: Plot (unchanged)
mg.tl.plot_stable_composite(composite_df, best_n, ll_n, prob_n)

# Step 4: View ranked plateau table
print(ranked_plateaus)