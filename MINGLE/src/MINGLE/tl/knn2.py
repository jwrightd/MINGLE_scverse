import anndata as ad
from typing import Dict, Sequence, Optional, List
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

def KNN2(
    adata: ad.AnnData,
    *,
    x_key: str = "x",
    y_key: str = "y",
    region_key: str = "unique_region",
    cluster_col: str = "cell_type",
    ks: Sequence[int] = (5, 10, 20),
    keep_obs_cols: Optional[Sequence[str]] = None,   # <-- NEW
) -> Dict[int, pd.DataFrame]:
    """
    Compute cell-type neighborhood windows using a k-NN in (x,y) per region.

    NEW:
    ----
    keep_obs_cols:
        Optional list/tuple of columns from adata.obs to carry through into each
        output window dataframe (in addition to region_key and cluster_col).

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping k -> DataFrame of neighborhood summaries.
    """
    ks = list(ks)
    n_neighbors = max(ks)

    adata = adata.copy()

    # Default keep set: keep x/y if present (without changing your logic elsewhere)
    if keep_obs_cols is None:
        keep_obs_cols = []

    # Build the final "kept" columns list (preserve user order, de-dup)
    keep_cols: List[str] = []
    for c in list(keep_obs_cols) + [x_key, y_key, region_key, cluster_col]:
        if c not in keep_cols:
            keep_cols.append(c)

    # Filter to only those that exist in obs (avoid KeyError if user passes extras)
    keep_cols_existing = [c for c in keep_cols if c in adata.obs.columns]

    # Dummy encode cell types (same logic)
    dummies = pd.get_dummies(adata.obs[cluster_col])
    adata.obs = pd.concat([adata.obs, dummies], axis=1)

    sum_cols = adata.obs[cluster_col].unique()
    values = adata.obs[sum_cols].values

    tissue_group = adata.obs.groupby(region_key)
    exps = list(adata.obs[region_key].unique())

    def get_windows(job, n_neighbors: int) -> np.ndarray:
        _start_time, _idx, tissue_name, indices = job
        tissue = adata[adata.obs[region_key] == tissue_name]
        coords = tissue.obs[[x_key, y_key]].values
        fit = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
        distances, neighbor_idx = fit.kneighbors(coords)

        args = distances.argsort(axis=1)
        add = np.arange(neighbor_idx.shape[0]) * neighbor_idx.shape[1]
        sorted_indices = neighbor_idx.flatten()[args + add[:, None]]

        neighbors = tissue.obs.index.values[sorted_indices]
        return neighbors.astype(np.int32)

    tissue_chunks = [
        (time.time(), exps.index(tissue_name), tissue_name, indices)
        for tissue_name, indices in tissue_group.groups.items()
        for indices in np.array_split(indices, 1)
    ]

    tissues = [get_windows(job, n_neighbors) for job in tissue_chunks]

    out_dict = {}
    for k in ks:
        for neighbors, job in zip(tissues, tissue_chunks):
            tissue_name = job[2]
            indices = job[3]
            chunk = np.arange(len(neighbors))

            window = values[neighbors[chunk, :k].flatten()]
            window = window.reshape(len(chunk), k, len(sum_cols)).sum(axis=1)
            out_dict[(tissue_name, k)] = (window.astype(np.float16), indices)

    windows: Dict[int, pd.DataFrame] = {}
    for k in ks:
        dfs = []
        for exp in exps:
            data, idx = out_dict[(exp, k)]
            # Keep your original behavior of indexing by idx.astype(int)
            df_k = pd.DataFrame(data, index=idx.astype(int), columns=sum_cols)
            dfs.append(df_k)

        window_df = pd.concat(dfs, axis=0)

        # Ensure the index types match (your existing guard)
        if window_df.index.dtype != adata.obs.index.dtype:
            window_df.index = window_df.index.astype(str)

        window_df = window_df.loc[adata.obs.index]

        # NEW: prepend kept columns (including x/y if requested / available)
        window_df = pd.concat(
            [adata.obs[keep_cols_existing], window_df],
            axis=1,
        )

        windows[k] = window_df

    return windows
