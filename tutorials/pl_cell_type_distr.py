import MINGLE as mg
adata = mg.pp.read_file(r"/Volumes/data/MINGLE/Data/Intestine/20251217_intestine_inner_outerfollicle_probabilitybincluster_2probbins.csv")
fig, ax, rank_df, combined_perc = mg.pl.cell_type_distributions(
     adata,
     neighborhoods_to_plot=("Inner Follicle", "Outer Follicle"),
     min_cells=10,
     neighborhood_key="Neighborhood",
     store_key="inner_outer_combined_stacked",
     return_fig=True,
 )
