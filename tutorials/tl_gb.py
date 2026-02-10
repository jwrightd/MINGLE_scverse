import anndata as ad
import MINGLE as mg
adata = mg.pp.read_file(r"/Volumes/data/MINGLE/Data/Intestine/20251217_intestine_inner_outerfollicle_probabilitybincluster_2probbins.csv")


out = mg.tl.gb(
    adata,
    cluster_key="Probability_Bin_Cluster",
    score_key="Score",
    neighborhood_key="Neighborhood",
    inner_name="Inner Follicle",
    outer_name="Outer Follicle",
    min_cells=10,
    pb_prefix="pb",
    region_key="unique_region",
    region_value="B006_Descending - Sigmoid",
    x_key="x",
    y_key="y",
    k_neighbors=20,
    normalize_by="iqr",
    grad_prefix="grad",
    make_plots=True,
)

# Same “reference tables” your script prints:
print(adata.uns["pb_agg_df"])
print("Summary:", adata.uns["grad_summary"])