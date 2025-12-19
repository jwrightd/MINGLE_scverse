import anndata as ad
import pandas as pd
import numpy as np
from typing import Optional
from .gmmgpu_knn import KNN

def centroid_Calculation(
    adata: ad.AnnData,
    *,
    k: int = 10,
    cluster_col: str = "Cell Type",
    neighborhood_col: str = "Neighborhood",
):
    # 1. Get KNN windows
    windows = KNN(adata, cluster_col=cluster_col)
    win = windows[k].copy()

    # Debug: print index types and some examples
    print("Centroids: adata.obs.index dtype:", adata.obs.index.dtype)
    print("Centroids: win.index dtype      :", win.index.dtype)
    print("Centroids: first 5 adata indices:", list(adata.obs.index[:5]))
    print("Centroids: first 5 win indices  :", list(win.index[:5]))

    # Cell type dummy columns created by KNN
    cell_types = [
        c for c in win.columns
        if c not in [neighborhood_col, "unique_region", cluster_col]
    ]

    results = []
    for nb in adata.obs[neighborhood_col].unique():
        # cells in this neighborhood
        idxs = adata.obs.index[adata.obs[neighborhood_col] == nb]

        # Only keep indices that actually exist in win.index
        idxs_in_win = idxs.intersection(win.index)

        if len(idxs_in_win) == 0:
            print(f"[WARN] Neighborhood {nb} has no matching indices in windows; skipping.")
            continue

        subset = win.loc[idxs_in_win]

        row = {neighborhood_col: nb}
        for ct in cell_types:
            if ct in subset.columns:
                row[f"{ct}_mean"] = subset[ct].mean()
                row[f"{ct}_std"] = subset[ct].std()
        results.append(row)

    df = pd.DataFrame(results).set_index(neighborhood_col)

    obs = pd.DataFrame(index=df.index)
    var = pd.DataFrame(index=df.columns)

    X = df.to_numpy(dtype=float)
    centroid_adata = ad.AnnData(X=X, obs=obs, var=var)

    return centroid_adata
