import MINGLE as mg

adata = mg.pp.read_file(r"/Volumes/data/MINGLE/Data/Intestine/05_25_HuBMAP_tunit.csv")
cell_type_features = [
        "Plasma Cell Enriched",
        "Mature Epithelial",
        "Innate Immune Enriched",
        "Follicle",
        "Adaptive Immune Enriched",
        "Secretory Epithelial",
        "CD66+ Mature Epithelial",
        "CD8+ T Enriched IEL",
        "Stroma",
        "Smooth Muscle",
    ]
probabilities_df, visualization_df, filtered_region_df = mg.pl.spatial_probability_mapping(
    adata,
    centroids_csv_path=r"/Volumes/data/MINGLE/Data/Intestine/intestine_all_neighborhood_centroids_tissueunit.csv",
    cell_type_features = cell_type_features,
    k=300,
    batch_size=20000,
    desired_region="B008_Sigmoid",
)
