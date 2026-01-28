import MINGLE as mg
import pandas as pd

# -------------------------
# Paths
# -------------------------
data_path = r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_from_h5mu.csv"
probabilities_path = r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_esophagus_all_cells_all_neighborhood_probs.csv"

out_probs_path = "local_probs_ad.csv"
out_delta_path = "delta_probs_ad.csv"

# -------------------------
# Column keys
# -------------------------
X = "x"
Y = "y"
reg = "region"
cluster_col = "Cell Type"
cellid = "cellid"
neigh = "neigh_name"
path_col = "Path_report"   # <-- IMPORTANT: this is the column name, not the file path

keep_cols = [X, Y, reg, cluster_col, cellid, neigh, path_col]

# -------------------------
# Load cells (AnnData)
# -------------------------
cells = mg.pp.read_file(data_path)

# Ensure cellid exists (read_file usually does this for h5mu/h5ad; CSV wrapper might not)
if "cellid" not in cells.obs.columns:
    cells.obs["cellid"] = cells.obs_names.astype(str)

copy_cells = cells.obs.copy()

print(cells)
print(cells.obs.head())
print(cells.obs.shape)

# -------------------------
# Make windows2
# -------------------------
windows = mg.tl.KNN2(
    cells,
    x_key=X,
    y_key=Y,
    region_key=reg,
    cluster_col=cluster_col,
    keep_obs_cols=keep_cols,
)

k = 10
windows2 = windows[k]
windows2[cluster_col] = cells.obs[cluster_col].values

# -------------------------
# Features
# -------------------------
cell_type_features = [
    'Squamous Annexin A1+', 'Squamous p63+', 'Endothelial', 'Chief',
    'Squamou p63+ EGFRhi', 'Neutrophil', 'M1 Macrophage', 'Epithelial',
    'Stroma', 'Epithelial Ki67+ p53+', 'Endothelial CD36hi',
    'CD4+ Treg', 'CD4+ T cell PD1+', 'CD4+ T cell', 'Nerve',
    'CD8+ T cell', 'Epithelial MUC1+ Ki67+', 'B cell',
    'CD8+ T cell PD1+', 'M2 Macrophage', 'Foveloar', 'Neuroendocrine',
    'Epithelial CK7+ p53+', 'Smooth Muscle', 'Epithelial pH2AX+', 'DC',
    'Plasma', 'Epithelial p53+', 'Endothelial aSMAhi', 'Lymphatic',
    'Neck', 'Goblet', 'Parietal', 'Epithelial CD73hi',
    'Foveloar Ki67+ p53+', 'Stroma CD73+', 'Goblet p53+', 'Neck p53+',
    'Lymphatic CD73+', 'Epithelial CK7+', 'Foveloar p53+',
    'Goblet Ki67+ p53+', 'Paneth', 'Epithelial HLADR+',
    'Neck Ki67+ p53+'
]

# -------------------------
# Global probs + neighborhoods list
# -------------------------
probabilities_df = pd.read_csv(probabilities_path)

neighborhoods = probabilities_df.columns.tolist()
if "cellid" in neighborhoods:
    neighborhoods.remove("cellid")
if "Unnamed: 0" in neighborhoods:
    neighborhoods.remove("Unnamed: 0")

# assigned_df required by crd validation
assigned_df = cells.obs.set_index("cellid")

# -------------------------
# Run crd (correct signature)
# -------------------------
probs, deltas = mg.tl.crd2(
    windows2=windows2,
    probabilities_df=probabilities_df,
    assigned_df=assigned_df,
    neighborhoods=neighborhoods,
    cell_type_features=cell_type_features,
    copy_cells=copy_cells,
    out_probs_path=out_probs_path,
    out_delta_path=out_delta_path,
    # optional: override params here if you want
    # min_count=10,
    # min_floor_abs=0.05,
    # use_region_frac=0.25,
    # shrink_alpha=5.0,
)

print(probs.shape, deltas.shape)
