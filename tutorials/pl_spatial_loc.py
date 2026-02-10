import MINGLE as mg
# ---- Load with MINGLE ----
adata = mg.pp.read_file(r"/Volumes/data/MINGLE/Data/Melanoma/melanoma_all_information.csv")

neighborhood_cols = [
    "Stromal Enriched", "Vasculature", "Neutrophil Enriched",
    "Macrophage Enriched Immune", "Vasculature & Immune",
    "Productive T cell & Tumor", "Proliferating Tumor",
    "Inflamed Tumor", "Tumor & Immune", "Epithelial/Skin Appendages",
    "Resting Tumor", "PDPN+ Stromal Enriched", "Follicle",
    "Perivascular", "Immune Infiltrate", "DC Enriched Immune",
]

# ---- Example call ----
fig, ax, masks = mg.pl.spatial_loc_region(
    adata,
    region="14_06_23_reg002.tsv",
    n1="Inflamed Tumor",
    n2="Productive T cell & Tumor",
    threshold=0.25,
)
