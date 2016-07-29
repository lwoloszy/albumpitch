import pandas as pd


def get_genres(df):
    return df['genres'].map(lambda x: x[0] if len(x) else None).tolist()
