import MINGLE as mg

results = mg.pl.cnd(
    cells_path=r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_from_h5mu.csv",
    probs_paths={
        "combined": r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_esophagus_all_cells_all_neighborhood_probs.csv",
        "tumor": r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_esophagus_tumor_all_cells_all_neighborhood_probs.csv",
        "normal": r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_esophagus_normal_all_cells_all_neighborhood_probs.csv",
        "metaplasia": r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_esophagus_metaplasia_all_cells_all_neighborhood_probs.csv",
        "dysplasia": r"/Volumes/data/MINGLE/Data/Esophagus/all_regions_esophagus_dysplasia_all_cells_all_neighborhood_probs.csv",
    },
    out_dir=None,
    out_prefix=None,
    min_count=10,
    save_deltas=True,
    make_plot=True,
)
