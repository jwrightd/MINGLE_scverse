from __future__ import annotations
from typing import Tuple, Optional, Union
from typing import Optional

import numpy as np
import pandas as pd
import anndata as ad
from anndata import AnnData


def mergeGMM(
    GMM_adata: AnnData,
    cell_adata: AnnData,
    *,
    join: str = "outer",
) -> AnnData:
    """
    Merge GMM results AnnData with the main cell AnnData along observations.

    This is the scverse-style equivalent of the original `mergeGMM`,
    but implemented with `anndata.concat` and a clear API.

    Parameters
    ----------
    GMM_adata
        AnnData containing GMM results (e.g. neighborhood probabilities, cluster labels).
        Its `.obs_names` must match (a subset of) `cell_adata.obs_names`.
    cell_adata
        AnnData containing the main annotated dataset (cell metadata, coordinates, etc.).
    join
        How to join variables (columns) from the two AnnData objects.
        Passed to `anndata.concat(join=...)`. Common choices:
        - "outer" (default): union of variables
        - "inner": intersection of variables

    Returns
    -------
    AnnData
        New AnnData with:
          - obs: aligned cells (by `obs_names`)
          - var: combined variables from `cell_adata` and `GMM_adata`
          - X/obsm/obsp/misc merged according to `anndata.concat` rules.
    """
    # We want to align on obs (cells) and concatenate along variables.
    # axis=1 → concatenate vars, match obs by name.
    merged = ad.concat(
        {"cell": cell_adata, "gmm": GMM_adata},
        axis=1,
        join=join,
        label=None,
        merge="unique",
    )
    return merged

def findPositives(
    adata: AnnData,
    *,
    prob_key: str = "neighborhood_probabilities",
    threshold: float = 0.25,
    result_key: str = "Count_Above_Threshold",
    prob_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None,
) -> AnnData:
    """
    Count per-cell how many probabilities exceed threshold.

    If prob_matrix is provided (DataFrame or ndarray) it will be used instead
    of adata.obsm[prob_key]. In that case, adata is not modified except to
    store result_key in adata.obs.
    """
    # get matrix
    if prob_matrix is None:
        if prob_key not in adata.obsm:
            raise KeyError(f"{prob_key!r} not found in adata.obsm")
        prob_raw = adata.obsm[prob_key]
        if isinstance(prob_raw, pd.DataFrame):
            probs = prob_raw.reindex(adata.obs_names).to_numpy()
        else:
            probs = np.asarray(prob_raw)
            if probs.shape[0] != adata.n_obs:
                raise ValueError("shape mismatch")
    else:
        if isinstance(prob_matrix, pd.DataFrame):
            # ensure row order aligns with adata.obs_names if indices provided
            if list(prob_matrix.index) == list(adata.obs_names):
                probs = prob_matrix.to_numpy()
            else:
                # try reindex to adata.obs_names if possible
                try:
                    probs = prob_matrix.reindex(adata.obs_names).to_numpy()
                except Exception:
                    probs = prob_matrix.to_numpy()
        else:
            probs = np.asarray(prob_matrix)
            if probs.shape[0] != adata.n_obs:
                raise ValueError("prob_matrix row count does not match adata.n_obs")

    counts = (probs > threshold).sum(axis=1).astype(int)
    adata.obs[result_key] = counts
    return adata