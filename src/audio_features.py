from __future__ import print_function
import psycopg2
import pandas as pd


def get_mean_audio_features():
    """
    Computes the across track mean audio features for each album in catalog
    and returns the result in a pandas dataframe

    Args:
        None
    Returns:
        df: Pandas DataFrame with each column representing a different
            audio feature (additional columns identify album)
    """

    conn = psycopg2.connect(database='albumpitch', user='lukewoloszyn')
    cur = conn.cursor()

    try:
        cur.execute("""
        SELECT p.id, p.url, avg(acousticness), avg(danceability),
        avg(energy), avg(instrumentalness), avg(key),
        avg(liveness), avg(loudness), avg(speechiness),
        avg(tempo), avg(time_signature), avg(valence)
        FROM pitchfork p
        JOIN spotify_albums sa ON p.spotify_id = sa.id
        JOIN spotify_audio_features saf ON sa.id = saf.album_id
        GROUP BY p.id;
        """)
        result = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(result, columns=['review_id', 'url', 'acoustic', 'dance',
                                       'energy', 'instrument', 'key',
                                       'live', 'loud', 'speech', 'tempo',
                                       'time_signature', 'valence'])
    return df
