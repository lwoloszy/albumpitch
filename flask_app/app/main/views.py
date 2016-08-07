import os
import sys
from flask import render_template, session, redirect, url_for, current_app
from flask import request
from flask import jsonify
from sqlalchemy import text

import dill
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .. import db
from . import main

import text_preprocess
sys.modules['text_preprocess'] = text_preprocess

basedir = os.path.abspath(os.path.dirname(__file__))

URLS = None
TFIDF = None
LSI = None
SVD_TRANS = None


@main.route('/', methods=['GET'])
def index():
    n_recc = 200
    album_query = request.args.get('album-query', '')
    keyword_query = request.args.get('keyword-query', '')

    if not album_query and not keyword_query:
        return render_template('index.html', seed_album=None, album_list=[])

    if album_query:
        cmd = """
        SELECT DISTINCT url, artist, album FROM pitchfork
        WHERE concat_ws(': ', artist, album) ilike :album_query
        """
        album_query = u'%{:s}%'.format(album_query)
        cur = db.engine.execute(text(cmd), album_query=album_query)
        results = cur.fetchall()

        if len(results) == 0:
            return render_template('index.html', album_list=[])

        album_query = '{:s}: {:s}'.format(results[0][1], results[0][2])
        urls, sims = gen_recc_aq(results[0][0], n_recc)
    elif keyword_query:
        urls, sims = gen_recc_kq(keyword_query, n_recc)

    sql_query = text("""
    DROP TABLE IF EXISTS sorted;
    CREATE TEMPORARY TABLE sorted (
    url varchar(255) NOT NULL primary key,
    sim real NOT NULL);
    """)
    db.engine.execute(sql_query)
    for url, sim in zip(urls, sims):
        sql_query = text("""
        INSERT INTO sorted
        VALUES ('{:s}', {:f})
        ON CONFLICT DO NOTHING;
        """.format(url, sim))
        db.engine.execute(sql_query)

    sql_query = text("""
    SELECT p.url, p.album_art, p.artist, p.album, p.genres, round(sim::numeric, 3), sa.link
    FROM sorted s JOIN pitchfork p on s.url = p.url
    LEFT JOIN spotify_albums sa ON p.spotify_id = sa.id
    ORDER BY sim DESC
    LIMIT 36;
    """)

    cur = db.engine.execute(sql_query)
    results = cur.fetchall()

    # album_list = [results[i:i+n_col]
    #              for i in xrange(0, len(results), n_col)]

    return render_template('index.html',
                           seed_album=album_query, seed_word=keyword_query,
                           albums=results)


@main.route('/_typeahead')
def typeahead():
    max_results = 20
    partial = request.args.get('q')
    partial = u'%{:s}%'.format(partial)
    cmd = text("""
    SELECT DISTINCT artist, album FROM pitchfork
    WHERE concat_ws(' ', artist_clean, album_clean) ilike :partial
    or concat_ws(': ', artist_clean, album_clean) ilike :partial
    """)
    cur = db.engine.execute(cmd, partial=partial)
    results = [': '.join(result[0:2]) for result in cur.fetchall()]
    return jsonify(matching_results=results[:max_results])


@main.before_app_first_request
def load_global_data():
    global URLS, TFIDF, LSI, SVD_TRANS
    URLS = np.load(basedir+'/models/urls.npy')
    TFIDF = load_dill(basedir+'/models/tfidf.dill')
    LSI = load_dill(basedir+'/models/svd.dill')
    SVD_TRANS = np.load(basedir+'/models/svd_trans.npy')


def gen_recc_aq(pitchfork_url, n_recc=30):
    idx = np.where(URLS == pitchfork_url)[0][0]
    cos_sims = cosine_similarity(
        SVD_TRANS[idx, :].reshape(1, -1), SVD_TRANS).flatten()
    # don't get same album, so skip last one
    closest_idx = np.argsort(cos_sims)[-n_recc-1:-1][::-1]
    return URLS[closest_idx], cos_sims[closest_idx]


def gen_recc_kq(kq, n_recc=30):
    q = TFIDF.get_feature_names()
    kq_tfidf = TFIDF.transform([kq])
    kq_lsi = LSI.transform(kq_tfidf)
    cos_sims = cosine_similarity(kq_lsi.reshape(1, -1), SVD_TRANS).flatten()
    closest_idx = np.argsort(cos_sims)[-n_recc:][::-1]
    return URLS[closest_idx], cos_sims[closest_idx]


def load_dill(filename):
    with open(filename) as f:
        out = dill.loads(f.read())
    return out
