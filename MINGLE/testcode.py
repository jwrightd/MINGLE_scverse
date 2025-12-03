import MINGLE as mg
import sys
from pathlib import Path
import inspect
import pandas as pd

print("version:", getattr(mg, "__version__", None))

#test run
file_path = r"/Volumes/data/MINGLE/Data/Intestine/05_25_huBMAP_tunit.csv"
annObj = mg.pp.read_file(file_path)
centroids = mg.tl.centroid_Calculation(annObj, cluster_col="Cell Type", neighborhood_col="Neighborhood")

#X_df = pd.DataFrame(centroids.X, index=centroids.obs.index, columns=centroids.var.index)
#combined_df = pd.concat([centroids.obs, X_df], axis=1)
#with pd.option_context("display.max_rows", None, "display.max_columns", None):
#    print(combined_df)