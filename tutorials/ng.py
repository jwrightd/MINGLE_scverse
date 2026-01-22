import MINGLE as mg
import pandas as pd
import anndata as ad

file_path = r"/Volumes/data/MINGLE/Data/Intestine/intestine_all_information_2.csv"
cells = mg.pp.read_file(file_path)


prob_cols = [
'Innate Immune Enriched', 'Outer Follicle', 'Plasma Cell Enriched',
'Transit Amplifying Zone', 'Adaptive Immune Enriched', 'Stroma',
'Paneth Enriched', 'Smooth Muscle & Innate Immune', 'Mature Epithelial',
'Microvasculature', 'CD8+ T Enriched IEL', 'Stroma & Innate Immune',
'Macrovasculature', 'Innervated Stroma', 'Secretory Epithelial',
'Innervated Smooth Muscle', 'Smooth Muscle', 'Glandular Epithelial',
'CD66+ Mature Epithelial', 'Inner Follicle'
]
#
G, top_pairs = mg.tl.build_neighborhood_pair_graph(
    cells,
    prob_cols,
    threshold=0.25,
    region_key="unique_region",
    top_n=25,
)
mg.tl.plot_neighborhood_pair_graph(cells)