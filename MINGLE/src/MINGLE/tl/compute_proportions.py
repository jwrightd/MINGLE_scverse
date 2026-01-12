from __future__ import annotations
from typing import Optional, Union
import numpy as np
import pandas as pd
from anndata import AnnData

def compute_grouped_proportions(df, n1, n2, cell_type_col='Cell Type', threshold=0.25):
    pos_1 = df[n1] > threshold
    pos_2 = df[n2] > threshold
 
    only_1 = df[pos_1 & ~pos_2]
    both = df[pos_1 & pos_2]
    only_2 = df[~pos_1 & pos_2]
 
    def summarize(sub, label):
        counts = sub[cell_type_col].value_counts(normalize=True)
        df_sub = counts.reset_index()
        df_sub.columns = [cell_type_col, 'Proportion']
        df_sub['Subset'] = label
        return df_sub
 
    df1 = summarize(only_1, f"{n1}")
    df2 = summarize(both, f"{n1} +\n{n2}")
    df3 = summarize(only_2, f"{n2}")
 
    all_df = pd.concat([df1, df2, df3], ignore_index=True)
 
    # Fill in missing combinations
    celltypes = all_df[cell_type_col].unique()
    groups = [f"{n1}", f"{n1} +\n{n2}", f"{n2}"]
    full_index = pd.MultiIndex.from_product([celltypes, groups], names=[cell_type_col, 'Subset'])
    return all_df.set_index([cell_type_col, 'Subset']).reindex(full_index, fill_value=0).reset_index()