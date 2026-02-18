import anndata as ad
import pandas as pd
import numpy as np
import MINGLE as mg
file_path = r"/Volumes/data/MINGLE/Data/Melanoma/melanoma_all_information.csv"
cells = mg.pp.read_file(file_path)
centroids = mg.tl.centroid_Calculation(cells, cluster_col="Cell_Type", neighborhood_col="Neighborhood", region_col="region")