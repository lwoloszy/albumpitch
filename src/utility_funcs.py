import pandas as pd


def get_genres(df):
    sel = df['genres'].apply(lambda x: len(x) > 0)
    df = df[sel]
    # df['genre'] =
    return df['genres'].map(lambda x: x[0]).tolist()
