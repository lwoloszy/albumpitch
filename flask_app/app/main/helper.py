import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text
from .. import db


basedir = os.path.abspath(os.path.dirname(__file__))


def gen_recc(pitchfork_url, n_recc=30):
    print(basedir)
    urls = np.load(basedir+'/models/urls.npy')
    lsi = np.load(basedir+'/models/lsi.npy')

    idx = np.where(urls == pitchfork_url)[0][0]
    cos_sims = cosine_similarity(lsi[idx, :].reshape(1, -1), lsi).flatten()
    closest_idx = np.argsort(cos_sims)[-n_recc-1:-1][::-1]
    return urls[closest_idx], cos_sims[closest_idx]


def get_20_albums():
    #sql_query = text("""SELECT * FROM pitchfork WHERE url="{:s}";
    #""".format(aq))
    sql_query = text("SELECT artist, album, album_art FROM pitchfork LIMIT 24;")
    cur = db.engine.execute(sql_query)
    return cur.fetchall()
