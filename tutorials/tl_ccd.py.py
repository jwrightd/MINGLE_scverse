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
