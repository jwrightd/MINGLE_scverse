from __future__ import annotations
from typing import Optional, Tuple, Dict, Union, Sequence, Callable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from anndata import AnnData
import seaborn as sns

from MINGLE.tl.compute_proportions import compute_grouped_proportions
from MINGLE.pl._utils import save_figure


def make_celltype_palette_strict_from_adata(adata: AnnData, cell_type_col: str = "Cell Types"):
    """
    Build a non-repeating palette for cell types using several qualitative palettes.
    Returns (color_dict, palette_list, cell_types).
    """
    cell_types = sorted(adata.obs[cell_type_col].astype(str).fillna("").unique())
    palette_names = ["tab20", "Set3", "Set2", "Paired", "Dark2", "Accent"]
    combined_colors = []
    for name in palette_names:
        combined_colors.extend(sns.color_palette(name))
    if len(combined_colors) < len(cell_types):
        extra = plt.cm.viridis(np.linspace(0, 1, len(cell_types) - len(combined_colors)))
        combined_colors.extend([tuple(c[:3]) for c in extra])
    final = combined_colors[: len(cell_types)]
    color_dict = dict(zip(cell_types, final))
    return color_dict, final, cell_types

def plot_border_enrichment(
    *,
    adata: Optional[AnnData] = None,
    df_probabilities: Optional[pd.DataFrame] = None,
    df_plot: Optional[pd.DataFrame] = None,
    compute_grouped_proportions_fn: Optional[Callable[..., pd.DataFrame]] = None,
    n1: str,
    n2: str,
    # --- configurable parameters (defaults you can override) ---
    cell_type_col: str = "Cell Type",
    prob_key: str = "neighborhood_probabilities",
    pos_threshold: float = 0.25,
    min_count: int = 5,
    eps: float = 1e-9,
    log_base: int = 2,
    # plotting looks
    min_area: float = 50.0,
    max_area: float = 800.0,
    legend_counts: Optional[Sequence[int]] = None,
    color_dict: Optional[Dict[str, Union[str, Tuple[float, float, float]]]] = None,
    dpi: int = 300,
    figsize_scatter: Tuple[float, float] = (8, 8),
    figsize_legend_counts: Tuple[float, float] = (4, 6),
    figsize_legend_colors: Tuple[float, float] = (4, 8),
    # show/save
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    save_prefix: str = "border_enrichment",
    # optional: allow passing a small verbosity flag
    verbose: bool = False,
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    One-shot plotting for border enrichment (scatter + size legend + color legend).

    Either pass `adata` (with `adata.obsm[prob_key]`) or `df_probabilities` (DataFrame of probs + cell type).
    If df_plot is not provided, `compute_grouped_proportions_fn` will be used (or default import).
    Returns: (fig_scatter, fig_legend_counts, fig_legend_colors)
    """
    # --- defaults and helpers ---
    if legend_counts is None:
        legend_counts = np.array([2000, 5000, 20000])
    else:
        legend_counts = np.array(list(legend_counts), dtype=int)

    if compute_grouped_proportions_fn is None:
        compute_grouped_proportions_fn = compute_grouped_proportions

    def _make_df_prob_from_adata(a: AnnData) -> pd.DataFrame:
        # Build a DataFrame of probabilities with named columns if possible.
        if prob_key not in a.obsm:
            raise KeyError(f"{prob_key!r} not found in adata.obsm")
        prob_raw = a.obsm[prob_key]
        # If AnnData stores neighborhood names in .uns, use them as cols
        if isinstance(prob_raw, pd.DataFrame):
            df = prob_raw.reindex(a.obs_names).copy()
        else:
            arr = np.asarray(prob_raw)
            if arr.shape[0] != a.n_obs:
                raise ValueError(f"adata.obsm[{prob_key!r}] shape mismatch: {arr.shape} vs n_obs={a.n_obs}")
            # try to get neighborhood names from uns if present
            nb_names = a.uns.get("neighborhood_probability_neighborhoods", None)
            if nb_names is not None and len(nb_names) == arr.shape[1]:
                cols = list(nb_names)
            else:
                cols = [f"N{i}" for i in range(arr.shape[1])]
            df = pd.DataFrame(arr, index=a.obs_names, columns=cols)
        # attach cell-type column if present in adata.obs
        if cell_type_col not in df.columns and cell_type_col in a.obs.columns:
            df[cell_type_col] = a.obs[cell_type_col].reindex(df.index).astype(str)
        return df

    def _find_prob_col(df: pd.DataFrame, name: str):
        # simple exact-match only (you said you removed fuzzy matching)
        return name if name in df.columns else None

    # --- produce df_probabilities if needed ---
    if df_probabilities is None:
        if adata is None:
            raise ValueError("Please pass either `adata` or `df_probabilities`.")
        df_probabilities = _make_df_prob_from_adata(adata)
        if verbose:
            print("Built df_probabilities from adata; columns:", list(df_probabilities.columns)[:20])

    # --- compute or accept df_plot ---
    if df_plot is None:
        if compute_grouped_proportions_fn is None or not callable(compute_grouped_proportions_fn):
            raise ValueError("Either provide df_plot or a callable compute_grouped_proportions_fn.")
        # compute_grouped_proportions expects (df, n1, n2, cell_type_col=..., threshold=...)
        df_plot = compute_grouped_proportions_fn(df_probabilities, n1, n2, cell_type_col=cell_type_col, threshold=pos_threshold)
        if verbose:
            print("Computed df_plot (grouped proportions) with shape:", df_plot.shape)

    # pivot and average row
    joint_name = f"{n1} +\n{n2}"
    subset_order = [n1, joint_name, "Average", n2]
    pivot = df_plot.pivot(index="Subset", columns="Cell Type", values="Proportion").fillna(0)
    if n1 in pivot.index and n2 in pivot.index:
        pivot.loc["Average"] = (pivot.loc[n1] + pivot.loc[n2]) / 2
    else:
        pivot.loc["Average"] = 0.0
    pivot = pivot.reindex([s for s in subset_order if s in pivot.index])

    # detect cell-type column in df_probabilities
    if cell_type_col in df_probabilities.columns:
        ct_col = cell_type_col
    else:
        # strict (no fuzzy): try common alternatives
        ct_candidates = [c for c in df_probabilities.columns if c.lower() in ("cell type", "cell_types", "celltype", "cell_label", "celllabel", "cell_type_pred")]
        if ct_candidates:
            ct_col = ct_candidates[0]
        else:
            raise ValueError("No cell-type column found in df_probabilities (expected 'Cell Type' or similar).")

    # find prob columns and label fallback (exact match only)
    col_n1 = _find_prob_col(df_probabilities, n1)
    col_n2 = _find_prob_col(df_probabilities, n2)
    label_cols = [c for c in df_probabilities.columns if c.lower() in ("subset", "neighborhood", "label", "assigned", "cn")]

    if col_n1 and col_n2:
        prob_n1 = pd.to_numeric(df_probabilities[col_n1], errors="coerce").astype(float)
        prob_n2 = pd.to_numeric(df_probabilities[col_n2], errors="coerce").astype(float)
        pos_n1 = prob_n1 > pos_threshold
        pos_n2 = prob_n2 > pos_threshold
        mask_n1_only = pos_n1 & ~pos_n2
        mask_n2_only = pos_n2 & ~pos_n1
        mask_border = pos_n1 & pos_n2
    else:
        if not label_cols:
            raise KeyError("Could not find probability columns for n1/n2 nor a label/assigned column to fall back on.")
        label_col = label_cols[0]
        mask_n1_only = df_probabilities[label_col].astype(str) == n1
        mask_n2_only = df_probabilities[label_col].astype(str) == n2
        mask_border = df_probabilities[label_col].astype(str) == joint_name

    # counts per cell type
    all_types = sorted(df_probabilities[ct_col].astype(str).unique())
    counts_n1 = df_probabilities.loc[mask_n1_only].groupby(ct_col).size().reindex(all_types).fillna(0).astype(int)
    counts_n2 = df_probabilities.loc[mask_n2_only].groupby(ct_col).size().reindex(all_types).fillna(0).astype(int)
    counts_border = df_probabilities.loc[mask_border].groupby(ct_col).size().reindex(all_types).fillna(0).astype(int)

    common_cts = [ct for ct in pivot.columns if ct in all_types]
    if len(common_cts) == 0:
        raise ValueError("No matching cell types between pivot and df_probabilities.")

    p1 = pivot.loc[n1, common_cts].astype(float).values
    p2 = pivot.loc[n2, common_cts].astype(float).values
    pb = pivot.loc[joint_name, common_cts].astype(float).values

    c1 = counts_n1.reindex(common_cts).values.astype(int)
    c2 = counts_n2.reindex(common_cts).values.astype(int)
    cb = counts_border.reindex(common_cts).values.astype(int)

    mask_ok = (c1 >= min_count) & (c2 >= min_count) & (cb >= min_count)
    kept_cts = [ct for ct, ok in zip(common_cts, mask_ok) if ok]
    if len(kept_cts) == 0:
        raise ValueError("No cell types pass the min_count filter.")

    p1_f = np.array([pivot.loc[n1, ct] for ct in kept_cts], float)
    p2_f = np.array([pivot.loc[n2, ct] for ct in kept_cts], float)
    pb_f = np.array([pivot.loc[joint_name, ct] for ct in kept_cts], float)
    cb_f = counts_border.reindex(kept_cts).values.astype(int)

  # ---- DEBUG / VERBOSE output (safe: inside function scope) ----
    if verbose:
        print("=== DEBUG: compute_grouped_proportions signature: (df, n1, n2, cell_type_col='Cell Type', threshold=0.25)")
        print("=== DEBUG: shapes and few values ===")
        print("kept_cts (len):", len(kept_cts))
        print("kept_cts sample:", kept_cts[:10])
        print("p1_f shape:", getattr(p1_f, "shape", None))
        print("p2_f shape:", getattr(p2_f, "shape", None))
        print("pb_f shape:", getattr(pb_f, "shape", None))
        print("cb_f shape:", getattr(cb_f, "shape", None))
        # show first few numeric values (safe even if short)
        print("example p1_f[:10]:", np.asarray(p1_f).flatten()[:10])
        print("example p2_f[:10]:", np.asarray(p2_f).flatten()[:10])
        print("example pb_f[:10]:", np.asarray(pb_f).flatten()[:10])
        print("example cb_f[:10]:", np.asarray(cb_f).flatten()[:10])
        # check finite / zeros
        print("any NaN/inf in p1/p2/pb?:",
              np.any(~np.isfinite(p1_f)), np.any(~np.isfinite(p2_f)), np.any(~np.isfinite(pb_f)))
        print("counts cb_f min/max:", np.min(cb_f), np.max(cb_f))
        print("df_probabilities.columns sample:", list(df_probabilities.columns)[:40])
        print("df_probabilities.head():")
        print(df_probabilities.head(3).to_string())
        
    # compute log ratios
    logfn = np.log2 if log_base == 2 else np.log
    x_vals = logfn((pb_f + eps) / (p1_f + eps))
    y_vals = logfn((pb_f + eps) / (p2_f + eps))

    # color dict: use provided or build from adata if available
    if color_dict is None:
        if adata is not None:
            # try to build a palette from adata.obs
            types_seen = kept_cts
            palette = plt.cm.tab20(np.linspace(0, 1, len(types_seen)))
            color_dict = dict(zip(types_seen, palette))
        else:
            color_dict = {ct: "#cccccc" for ct in kept_cts}
    colors = [color_dict.get(ct, "#cccccc") for ct in kept_cts]

    # area mapping
    max_count = max(int(cb_f.max()), int(legend_counts.max()))
    def _count_to_area(k):
        return min_area + (k / max_count) * (max_area - min_area)

    areas = np.array([_count_to_area(c) for c in cb_f], float)
    legend_areas = np.array([_count_to_area(int(c)) for c in legend_counts], float)

    # ---------------- FIGURE 1: SCATTER ----------------
    fig_scatter, ax = plt.subplots(figsize=figsize_scatter, dpi=dpi)
    # ensure numeric and same length
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    areas = np.asarray(areas, dtype=float)
    # drop NaNs/infs
    finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (areas > 0)
    if not np.any(finite_mask):
        raise ValueError("No finite points to plot after filtering NaNs/infs.")
    ax.scatter(x_vals[finite_mask], y_vals[finite_mask], s=areas[finite_mask], c=[colors[i] for i, ok in enumerate(finite_mask) if ok], edgecolor="k", alpha=0.95, linewidth=0.35)

    ax.axhline(0, color="gray", linestyle="--")
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("", fontsize=20)
    ax.set_ylabel("", fontsize=20)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal", "box")
    ax.grid(False)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.tick_params(axis="both", labelsize=12)
    plt.tight_layout()

    # ---------------- FIGURE 2: SIZE LEGEND ----------------
    fig_leg_counts, ax_leg = plt.subplots(figsize=figsize_legend_counts, dpi=dpi)
    handles = [plt.scatter([], [], s=float(ar), color="gray", edgecolors="none") for ar in legend_areas]
    labels = [f"{int(c)} cells" for c in legend_counts]
    ax_leg.legend(handles=handles, labels=labels, title="Border Cell Count", loc="center", frameon=False, labelspacing=1.2, handletextpad=2.0, borderpad=2.0)
    ax_leg.axis("off")
    plt.tight_layout()

    # ---------------- FIGURE 3: COLOR LEGEND ----------------
    fig_leg_colors, ax_leg_colors = plt.subplots(figsize=figsize_legend_colors, dpi=dpi)
    color_patches = [Patch(facecolor=color_dict.get(ct, "#cccccc"), edgecolor="none", label=ct) for ct in kept_cts]
    leg_colors = ax_leg_colors.legend(handles=color_patches, title="Cell Type", loc="center", frameon=False, ncol=1, labelspacing=0.2, handletextpad=0.8, borderpad=0.8)
    if leg_colors is not None:
        leg_colors.set_frame_on(False)
        if leg_colors.get_frame() is not None:
            leg_colors.get_frame().set_alpha(0.0)
    ax_leg_colors.axis("off")
    plt.tight_layout()

    # save logic
    if show is None:
        show = not bool(save)

    if save:
        save_figure(fig_scatter, base=f"{save_prefix}_scatter", save=save)
        save_figure(fig_leg_counts, base=f"{save_prefix}_counts_legend", save=save)
        save_figure(fig_leg_colors, base=f"{save_prefix}_color_legend", save=save)

    if show:
        plt.show()

    return fig_scatter, fig_leg_counts, fig_leg_colors
