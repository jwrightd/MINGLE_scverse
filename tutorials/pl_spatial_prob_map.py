import MINGLE as mg

adata = mg.pp.read_file(r"/Volumes/data/MINGLE/Data/Intestine/05_25_HuBMAP_tunit.csv")

probabilities_df, visualization_df, filtered_region_df = mg.pl.spatial_probability_mapping(
    adata,
    centroids_csv_path=r"/Volumes/data/MINGLE/Data/Intestine/intestine_all_neighborhood_centroids_tissueunit.csv",
    k=300,
    batch_size=20000,
    desired_region="B008_Sigmoid",
)
