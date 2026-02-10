import MINGLE as mg

adata = mg.pp.read_file(r"/Volumes/data/MINGLE/Data/Intestine/20251217_intestine_inner_outerfollicle_probabilitybincluster_2probbins.csv")
ax, rank_df = mg.pl.plot_pooled_violin(
    adata,
    neighborhood_key="Neighborhood",
    neighborhoods_to_plot=("Inner Follicle", "Outer Follicle"),
    min_cells=10,
    cluster_key="Probability_Bin_Cluster",
    score_key="Score",
    plot_means=True,
)