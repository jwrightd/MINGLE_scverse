import anndata as ad
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Sequence
import time

def KNN(
    adata: ad.AnnData,
    *,
    x_key: str = "x",
    y_key: str = "y",
    region_key: str = "unique_region",
    cluster_col: str = "Cell Type",
    ks: Sequence[int] = (5, 10, 20),
) -> Dict[int, pd.DataFrame]:
    """
    Return windows[k] as a DataFrame whose index matches adata.obs.index
    (cell IDs), with count features per cell-type.
    """
    adata = adata.copy()  # work on a copy so we don't mutate caller

    # --- 1. Dummy columns for each cell type ---
    dummies = pd.get_dummies(adata.obs[cluster_col])
    adata.obs = pd.concat([adata.obs, dummies], axis=1)

    cell_types = dummies.columns.tolist()            # one col per cell type
    values = adata.obs[cell_types].to_numpy()        # numpy matrix of counts
    global_index = adata.obs.index                   # index of cells

    tissue_group = adata.obs.groupby(region_key)
    exps = list(adata.obs[region_key].unique())
    n_neighbors = max(ks)

    # --- 2. Helper: get neighbors per tissue (returns labels, not positions) ---
    def get_windows(job, n_neighbors):
        _, _, tissue_name, _ = job

        # Subset to that region (keeps original index labels)
        tissue = adata[adata.obs[region_key] == tissue_name]
        coords = tissue.obs[[x_key, y_key]].values

        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
        distances, idxs = nbrs.kneighbors(coords)

        # sort neighbors by distance
        args = distances.argsort(axis=1)
        row_offsets = np.arange(idxs.shape[0]) * idxs.shape[1]
        sorted_indices = idxs.flatten()[args + row_offsets[:, None]]

        # map from local positions (0..n_tissue-1) to original labels
        neighbor_labels = tissue.obs.index.values[sorted_indices]
        return neighbor_labels  # dtype = same as adata.obs.index

    # jobs store the *labels* for each region’s cells
    jobs = [
        (time.time(), i, region, idxs)
        for i, (region, idxs) in enumerate(tissue_group.groups.items())
    ]

    all_neighbors = [get_windows(job, n_neighbors) for job in jobs]

    # Build windows for each k
    windows: Dict[int, pd.DataFrame] = {}

    # precompute a label→position map so we can index `values` correctly
    label_to_pos = pd.Series(
        np.arange(len(global_index)), index=global_index
    )

    for k in ks:
        dfs = []
        for neighbors, job in zip(all_neighbors, jobs):
            _, _, _, idxs = job  # idxs = cell labels in this region

            chunk = np.arange(len(neighbors))  # each row in `neighbors`
            flat_neighbors = neighbors[chunk, :k].flatten()  # k neighbors per cell

            # convert neighbor labels → integer positions into `values`
            pos_idx = label_to_pos.loc[flat_neighbors].to_numpy()

            summed = (
                values[pos_idx]
                .reshape(len(chunk), k, len(cell_types))
                .sum(axis=1)
            )

            # set index to the *labels* of the query cells for this region
            df = pd.DataFrame(summed, index=idxs, columns=cell_types)
            dfs.append(df)

        win = pd.concat(dfs)

        # reindex to match global adata.obs.index order
        win = win.reindex(global_index)

        # attach metadata
        meta = adata.obs[[region_key, cluster_col]]
        win = pd.concat([meta, win], axis=1)

        windows[k] = win

    return windows
