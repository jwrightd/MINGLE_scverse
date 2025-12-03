#LLM conversion to Scverse

import anndata as ad
import numpy as np
from scipy.stats import norm
import time
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from pathlib import Path
from typing import Union

import anndata as ad
import numpy as np
import pandas as pd


def read_file(path: Union[str, Path]) -> ad.AnnData:
    """
    Read a .csv or .h5ad file and return an AnnData object.

    For .csv files, the entire table is stored in ``adata.obs`` and ``adata.X``
    is an empty matrix (n_obs x 0). This works well for workflows that use
    only metadata / coordinates from `.obs` (e.g. KNN on x/y, neighborhoods).

    Parameters
    ----------
    path
        Path to the input file.

    Returns
    -------
    AnnData
        AnnData loaded from .h5ad, or constructed from .csv.

    Raises
    ------
    ValueError
        If the extension is not .csv or .h5ad.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path)

        # X is an empty matrix; all info lives in obs
        X = np.zeros((df.shape[0], 0), dtype=np.float32)
        adata = ad.AnnData(X=X, obs=df.copy())
        return adata

    elif ext == ".h5ad":
        return ad.read_h5ad(path)

    else:
        raise ValueError(f"Unsupported file type: {ext}. Expected .csv or .h5ad")



def KNN(
    adata: ad.AnnData,
    *,
    x_key: str = "x",
    y_key: str = "y",
    region_key: str = "unique_region",
    cluster_col: str = "cell_type",
    ks: Sequence[int] = (5, 10, 20),
) -> Dict[int, pd.DataFrame]:
    """
    Compute cell-type neighborhood windows using a k-NN in (x,y) per region.

    For each k in `ks`, returns a DataFrame with:
        - columns: region_key, cluster_col, one column per cell type (counts in k-NN window)
        - index: original cell indices

    Parameters
    ----------
    adata
        AnnData with spatial coordinates in `.obs[x_key]`, `.obs[y_key]`,
        region IDs in `.obs[region_key]`, and cluster labels in `.obs[cluster_col]`.
    x_key, y_key
        Names of the columns in `adata.obs` containing x/y coordinates.
    region_key
        Column in `adata.obs` defining regions / tissues (e.g. 'unique_region').
    cluster_col
        Column in `adata.obs` with cell type / cluster labels.
    ks
        Sequence of neighborhood sizes to compute.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping k -> DataFrame of neighborhood summaries.
    """
    ks = list(ks)
    n_neighbors = max(ks)

    # work on a copy so we don't mutate the original AnnData
    adata = adata.copy()

    # one-hot encode cluster_col into extra obs columns
    dummies = pd.get_dummies(adata.obs[cluster_col])
    adata.obs = pd.concat([adata.obs, dummies], axis=1)

    sum_cols = adata.obs[cluster_col].unique()
    values = adata.obs[sum_cols].values  # shape: n_cells x n_celltypes

    # group cells by region
    tissue_group = adata.obs.groupby(region_key)
    exps = list(adata.obs[region_key].unique())

    def get_windows(job, n_neighbors: int) -> np.ndarray:
        """
        Compute neighbor indices (in obs index space) for one region chunk.
        """
        _start_time, _idx, tissue_name, indices = job

        # subset AnnData to this tissue
        tissue = adata[adata.obs[region_key] == tissue_name]

        # fit kNN on this tissue's coordinates
        coords = tissue.obs[[x_key, y_key]].values
        fit = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)

        # neighbors for each point
        distances, neighbor_idx = fit.kneighbors(coords)

        # sort neighbors for determinism
        args = distances.argsort(axis=1)
        add = np.arange(neighbor_idx.shape[0]) * neighbor_idx.shape[1]
        sorted_indices = neighbor_idx.flatten()[args + add[:, None]]

        # map to original indices
        neighbors = tissue.obs.index.values[sorted_indices]
        return neighbors.astype(np.int32)

    # build jobs (here each region is a single chunk, but array_split lets you
    # later change #chunks if you want)
    tissue_chunks = [
        (time.time(), exps.index(tissue_name), tissue_name, indices)
        for tissue_name, indices in tissue_group.groups.items()
        for indices in np.array_split(indices, 1)
    ]

    # run kNN per chunk
    tissues = [get_windows(job, n_neighbors) for job in tissue_chunks]

    # aggregate per k
    out_dict = {}
    for k in ks:
        for neighbors, job in zip(tissues, tissue_chunks):
            tissue_name = job[2]
            indices = job[3]
            chunk = np.arange(len(neighbors))

            # neighbors[chunk, :k] gives neighbor indices for each cell in the chunk
            window = values[neighbors[chunk, :k].flatten()]  # (len(chunk)*k, n_celltypes)
            window = window.reshape(len(chunk), k, len(sum_cols)).sum(axis=1)
            out_dict[(tissue_name, k)] = (window.astype(np.float16), indices)

    # build per-k DataFrames
    windows: Dict[int, pd.DataFrame] = {}
    for k in ks:
        dfs = []
        for exp in exps:
            data, idx = out_dict[(exp, k)]
            df = pd.DataFrame(data, index=idx.astype(int), columns=sum_cols)
            dfs.append(df)

        window_df = pd.concat(dfs, axis=0)
        # ensure same order as adata.obs
        window_df = window_df.loc[adata.obs.index.values]
        window_df = pd.concat(
            [adata.obs[[region_key, cluster_col]], window_df],
            axis=1,
        )
        windows[k] = window_df

    return windows


def centroid_Calculation(
    adata: ad.AnnData,
    *,
    k: int = 10,
    cluster_col: str = "cell_type",
    neighborhood_col: str = "neighborhood",
    store_key: Optional[str] = None,
) -> ad.AnnData:
    """
    Compute per-neighborhood mean and std of cell-type counts in k-NN windows.

    This is the same computation you had before, but:

    - keeps indices aligned (no reset_index)
    - allows choosing k
    - optionally stores the centroid AnnData in `adata.uns[store_key]`
      so you can reuse it for many plots.

    Parameters
    ----------
    adata
        AnnData with:
          - `.obs[cluster_col]` (cell type labels)
          - `.obs[neighborhood_col]` (neighborhood assignment)
          - spatial info needed for KNN (x/y & region columns).
    k
        Neighborhood size to use from the KNN windows (must be in the `ks`
        list passed to KNN; default 10).
    cluster_col
        Column in `adata.obs` with cell-type labels.
    neighborhood_col
        Column in `adata.obs` with neighborhood IDs.
    store_key
        If not None, store the centroid AnnData in `adata.uns[store_key]`.

    Returns
    -------
    AnnData
        AnnData with:
          - obs: neighborhoods
          - var: centroid features (means/stds per cell type)
          - X: numeric matrix (n_neighborhoods x n_features)
    """
    # get KNN windows
    windows = KNN(adata)
    if k not in windows:
        raise ValueError(f"k={k} not in available ks from KNN: {list(windows.keys())}")

    windows_k = windows[k]

    # ensure we have the cluster column on windows_k
    windows_k[cluster_col] = adata.obs[cluster_col]

    # use obs directly; keep original indices (no reset_index)
    filtered_cells = adata.obs.copy()

    # cell types → columns we created in KNN
    cell_type_columns = adata.obs[cluster_col].unique()
    windows_k[cell_type_columns] = windows_k[cell_type_columns].astype("float32")

    neighborhoods_to_loop = adata.obs[neighborhood_col].unique()
    all_results = []

    for neighborhood in neighborhoods_to_loop:
        # cells in this neighborhood (indices are original obs index)
        filtered_neighborhood_df = filtered_cells[
            filtered_cells[neighborhood_col] == neighborhood
        ]
        cell_numbers_in_neighborhood = filtered_neighborhood_df.index.values

        # take matching rows from windows_k
        matching_cells_df = windows_k.loc[cell_numbers_in_neighborhood]

        mean_std_results = {neighborhood_col: neighborhood}

        for column in cell_type_columns:
            if column in matching_cells_df.columns:
                col_values = matching_cells_df[column]
                mean_std_results[f"{column}_mean"] = col_values.mean()
                mean_std_results[f"{column}_std"] = col_values.std()

        all_results.append(mean_std_results)

    # neighborhood × feature table
    results_df = pd.DataFrame(all_results).set_index(neighborhood_col)

    feature_cols = results_df.columns.tolist()
    X = results_df[feature_cols].to_numpy(dtype=np.float32)

    # obs: neighborhoods
    obs = pd.DataFrame(index=results_df.index)
    obs[neighborhood_col] = results_df.index

    # var: feature names
    var = pd.DataFrame(index=feature_cols)

    centroid_adata = ad.AnnData(X=X, obs=obs, var=var)

    if store_key is not None:
        adata.uns[store_key] = centroid_adata

    return centroid_adata



