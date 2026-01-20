import MINGLE as mg
import pandas as pd
import anndata as ad
from pathlib import Path

    
# ---------------------------
# Paths
# ---------------------------
path = r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_from_h5mu.csv"
probabilities_path = r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_esophagus_all_cells_all_neighborhood_probs.csv"

out_probs_path = "local_probs_ad.csv"
out_delta_path = "delta_probs_ad.csv"

X = 'x'
Y = 'y'
reg = 'region'
cluster_col = 'Cell Type'
cellid = 'cellid'
neigh = 'neigh_name'

keep_cols = [X, Y, reg, cluster_col, cellid, neigh, path]

cells = mg.pp.read_file(path)

copy_cells = cells.obs.copy()

print(cells)           # AnnData summary
print(cells.obs.head())  # show first obs rows (equivalent to your df.head())
print(cells.obs.shape)

windows = mg.tl.KNN2(
    cells,
    x_key=X,
    y_key=Y,
    region_key=reg,          # <-- use 'region'
    cluster_col=cluster_col, # <-- use 'Cell Type'
    keep_obs_cols=keep_cols,
)

k = 10
windows2 = windows[k]
windows2[cluster_col] = cells.obs[cluster_col].values

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

probabilities_df = pd.read_csv(probabilities_path)

probs, deltas = mg.tl.crd(cells, windows2, probabilities_df, cell_type_features)

