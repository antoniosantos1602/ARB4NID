import numpy as np
import pandas as pd

def get_binary_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return [
        col for col in df.columns
        if df[col].dropna().nunique() == 2 and set(df[col].dropna().unique()).issubset({0, 1})
    ]

def get_integer_columns(df, ignore_binary=True, tol=1e-6):

    integer_like_columns = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors='coerce').dropna()

        if series.empty:
            continue

        if np.all(np.isclose(series, np.round(series), atol=tol)):
            if ignore_binary and set(series.unique()).issubset({0, 1}):
                continue
            integer_like_columns.append(col)

    return integer_like_columns
