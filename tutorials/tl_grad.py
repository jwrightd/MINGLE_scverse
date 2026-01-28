import MINGLE as mg

#probabilities_df = pd.read_csv(r"/Volumes/data/MINGLE/Data/Intestine/intestine_tissueunit_all_cells_all_neighborhood_probs.csv")
#allinfo_df = pd.read_csv(r"/Volumes/data/MINGLE/Data/Intestine/intestine_all_information_2.csv")

adata = mg.pp.read_file(r"/Volumes/data/MINGLE/Data/Intestine/intestine_all_information_2.csv")
#adata.obs = allinfo_df.copy()

# Call the scverse-compatible function (replaces your long procedural code)
adata, df_sub, fc, windows_k, edges_used = mg.tl.mingle_neighborhoods_scverse(
    adata,
    tu1="Inner Follicle",
    tu2="Outer Follicle",
    ks=(10,20,30,50,100),
    k=20,
    distance_max=1000,
    n_clusters=5,
    target_neighborhoods=("Inner Follicle","Outer Follicle"),
)

# Same prints as your end
print(df_sub)
print(df_sub["Probability_Bin_Cluster"])
print(df_sub["neighborhood20"])  # out_neighborhood_key defaults to "neighborhood20"