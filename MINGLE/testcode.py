import MINGLE as mg

def main():
    print("version:", getattr(mg, "__version__", None))

    # Test run
    file_path = r"/Volumes/data/MINGLE/Data/Intestine/05_25_huBMAP_tunit.csv"
    cells = mg.pp.read_file(file_path)
    centroids = mg.tl.centroid_Calculation(cells, cluster_col="Cell Type", neighborhood_col="Neighborhood")
    updated_adata = mg.tl.cpu_gmm_probability(cells, centroids, 
                                               cluster_col="Cell Type", 
                                               neighborhood_col="Neighborhood",
                                               ks=(5, 10),
                                               num_processes=None)

    # Optional: print output of combined DataFrame (if needed)
    # X_df = pd.DataFrame(centroids.X, index=centroids.obs.index, columns=centroids.var.index)
    # combined_df = pd.concat([centroids.obs, X_df], axis=1)
    # with pd.option_context("display.max_rows", None, "display.max_columns", None):
    #     print(combined_df)

# Ensure multiprocessing code runs only when executed directly
if __name__ == "__main__":
    main()

