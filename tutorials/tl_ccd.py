import MINGLE as mg

adata, combined_melted, combo_counts = mg.tl.ccd(
    cells_path=r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_from_h5mu.csv",
    probs_paths={
        "combined": r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_esophagus_all_cells_all_neighborhood_probs.csv",
        "tumor": r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_esophagus_tumor_all_cells_all_neighborhood_probs.csv",
        "normal": r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_esophagus_normal_all_cells_all_neighborhood_probs.csv",
        "metaplasia": r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_esophagus_metaplasia_all_cells_all_neighborhood_probs.csv",
        "dysplasia": r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_esophagus_dysplasia_all_cells_all_neighborhood_probs.csv",
    },
    pp=mg.pp,          
    cellid_key="cellid",
    assigned_neigh_key="neigh_name",
    min_count=10,
    save_deltas=False,     # True if you want the per-context delta CSVs too
    out_dir=None,
    out_prefix=None
)

print(adata)
print("Layers:", list(adata.layers.keys()))
print("combined_melted:", combined_melted.shape)

# BELOW IS FROM gvs_pl.py


epithelial_cell_types = [
 'Squamous Annexin A1+','Squamous p63+','Squamou p63+ EGFRhi','Epithelial',
 'Epithelial Ki67+ p53+','Epithelial MUC1+ Ki67+','Epithelial CK7+ p53+',
 'Epithelial pH2AX+','Epithelial CD73hi','Epithelial p53+','Epithelial CK7+',
 'Epithelial HLADR+','Foveloar','Foveloar Ki67+ p53+','Foveloar p53+',
 'Goblet','Goblet p53+','Goblet Ki67+ p53+','Paneth','Chief','Parietal',
 'Neck','Neck p53+','Neck Ki67+ p53+','Neuroendocrine'
]

mesenchymal_cell_types = [
 'Endothelial','Endothelial CD36hi','Endothelial aSMAhi',
 'Lymphatic','Lymphatic CD73+','Stroma','Stroma CD73+','Smooth Muscle','Nerve'
]

immune_cell_types = [
 'Neutrophil','M1 Macrophage','M2 Macrophage','CD4+ Treg','CD4+ T cell PD1+',
 'CD4+ T cell','CD8+ T cell','CD8+ T cell PD1+','B cell','Plasma','DC'
]

bucket_map = {
    "Epithelial": epithelial_cell_types,
    "Mesenchymal": mesenchymal_cell_types,
    "Immune": immune_cell_types
}

cell_type_color_map = {
  "B cell": "#00ff00",
  "CD4+ T cell": "#ff00ff",
  "CD4+ T cell PD1+": "#0080ff",
  "CD4+ Treg": "#ff8000",
  "CD8+ T cell": "#80bf80",
  "CD8+ T cell PD1+": "#4c06b1",
  "Chief": "#c40129",
  "DC": "#fc82cb",
  "Endothelial": "#ebff1d",
  "Endothelial CD36hi": "#3d8004",
  "Endothelial aSMAhi": "#00ffff",
  "Epithelial": "#00ff80",
  "Epithelial CD73hi": "#008080",
  "Epithelial CK7+": "#87e8fc",
  "Epithelial CK7+ p53+": "#945576",
  "Epithelial HLADR+": "#8080ff",
  "Epithelial Ki67+ p53+": "#81e103",
  "Epithelial MUC1+ Ki67+": "#383049",
  "Epithelial p53+": "#f7d689",
  "Epithelial pH2AX+": "#ae33de",
  "Foveloar": "#f32387",
  "Foveloar Ki67+ p53+": "#0000ff",
  "Foveloar p53+": "#33bfca",
  "Goblet": "#11c23f",
  "Goblet Ki67+ p53+": "#a16411",
  "Goblet p53+": "#d99063",
  "Lymphatic": "#0c44c5",
  "Lymphatic CD73+": "#5bf953",
  "M1 Macrophage": "#a70183",
  "M2 Macrophage": "#b2acc8",
  "Neck": "#ef4028",
  "Neck Ki67+ p53+": "#60020f",
  "Neck p53+": "#5736f8",
  "Nerve": "#000080",
  "Neuroendocrine": "#bdac04",
  "Neutrophil": "#5b855c",
  "Paneth": "#b0ff80",
  "Parietal": "#516bac",
  "Plasma": "#5ff8b0",
  "Smooth Muscle": "#8000ff",
  "Squamou p63+ EGFRhi": "#8f2c33",
  "Squamous Annexin A1+": "#fe44d4",
  "Squamous p63+": "#05410f",
  "Stroma": "#f3c0fb",
  "Stroma CD73+": "#be68b5"
}


mg.pl.plot_global_vs_subset_horizontal_buckets(
    data=adata,
    neighborhood="Mature Intestinal and Immune",
    bucket_map=bucket_map,
    cell_type_color_map=cell_type_color_map,
    min_count=10,
    subset_region="E19_reg003",
    figsize=(11, 3)
)

mg.pl.plot_global_vs_subset_horizontal_buckets(
    data=adata,
    neighborhood="Vasculature",
    bucket_map=bucket_map,
    cell_type_color_map=cell_type_color_map,
    min_count=10,
    subset_region="E08_reg003",
    figsize=(10, 3)
)

mg.pl.plot_global_vs_subset_horizontal_buckets(
    data=adata,
    neighborhood="Vasculature",
    bucket_map=bucket_map,
    cell_type_color_map=cell_type_color_map,
    min_count=10,
    subset_patient="E08",
    figsize=(10, 3)
)

mg.pl.plot_global_vs_subset_horizontal_buckets(
    data=adata,
    neighborhood="Ki67hi p53hi Epithelial and Innate Immune",
    bucket_map=bucket_map,
    cell_type_color_map=cell_type_color_map,
    min_count=10,
    subset_region="E17_reg005",
    figsize=(10, 3)
)
# updated ipynb

# Choose ONE neighborhood and ONE context
neighborhood_name = "Mature Intestinal and Immune"
context_name = "Tumor"

# Minimum number of cells required to be plotted
min_cells = 10

mg.pl.plot_global_vs_subset_horizontal_buckets(
    data=adata,
    neighborhood=neighborhood_name,
    subset_context=context_name,
    bucket_map=bucket_map,
    cell_type_color_map=cell_type_color_map,
    min_count=min_cells,
    figsize=(10, 3),
    title_fontsize=25,
    label_fontsize=25
)
