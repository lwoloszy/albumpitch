def get_genres(df):
    """
    Computes genre for a dataframe

    Args:
        df: pandas dataframe that has a column 'genres', where
            each row in genres is a lit
    Returns:
        list: list of genre strings (we take the first genre in each row's
              list)
    """

    return df['genres'].map(lambda x: x[0] if len(x) else None).tolist()
