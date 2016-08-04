from flask import render_template, session, redirect, url_for, current_app
from flask import request
from flask import jsonify
from sqlalchemy import text
from .. import db
from . import main

import helper


@main.route('/', methods=['GET'])
def index():
    n_col = 4
    n_recc = 24
    album_query = request.args.get('album-query', '')

    if not album_query:
        return render_template('index.html', seed_album=None, album_list=[])

    print album_query
    cmd = """
    SELECT DISTINCT url, artist, album FROM pitchfork
    WHERE concat_ws(': ', artist, album) ilike :album_query
    """
    cur = db.engine.execute(text(cmd), album_query=album_query)

    #sql_query = text(
    #    """SELECT DISTINCT url, artist, album FROM pitchfork
    #    WHERE concat_ws(': ', artist, album) ilike '%{:s}%'
    #    """
    #    .format(album_query))

    #cur = db.engine.execute(sql_query)
    results = cur.fetchall()
    if len(results) != 1:
        return render_template('index.html', album_list=[])

    urls, sims = helper.gen_recc(results[0][0], n_recc)

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
        SELECT p.url, p.album_art, p.artist, p.album, p.genres, sim, sa.link
        FROM sorted s JOIN pitchfork p on s.url = p.url
        LEFT JOIN spotify_albums sa ON p.spotify_id = sa.id
        ORDER BY sim DESC
        LIMIT 24;
        """)

        cur = db.engine.execute(sql_query)
        results = cur.fetchall()

    album_list = [results[i:i+n_col] for i in xrange(0, len(results), n_col)]
    return render_template('index.html',
                           seed_album=album_query, album_list=album_list)


@main.route('/_typeahead')
def typeahead():
    max_results = 20
    partial = request.args.get('q')
    sql_query = text(
        """SELECT DISTINCT artist, album FROM pitchfork
        WHERE concat_ws(' ', artist, album) ilike '%{:s}%'
        """
        .format(partial))
    cur = db.engine.execute(sql_query)
    results = [': '.join(result[0:2]) for result in cur.fetchall()]
    return jsonify(matching_results=results[:max_results])
