import pandas as pd
import numpy as np


def add_column_bin(df, column, nbins):

    df = df.copy()
    df[f"{column}_bin"] = pd.cut(
        df[column].fillna(value=np.nan),
        np.percentile(df[column].dropna(), np.arange(nbins + 1) / nbins * 100),
        duplicates="drop",
        include_lowest=True,
        right=False,
    )
    return df
