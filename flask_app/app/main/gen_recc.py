import os
from sqlalchemy import text
from .. import db

basedir = os.path.abspath(os.path.dirname(__file__))


def gen_reccs_from_album(aq):
    #sql_query = text("""SELECT * FROM pitchfork WHERE url="{:s}";
    #""".format(aq))
    sql_query = text("SELECT * FROM pitchfork WHERE url='{:s}';".format(aq))
    cur = db.engine.execute(sql_query)
    return cur.fetchall()

def get_20_albums():
    #sql_query = text("""SELECT * FROM pitchfork WHERE url="{:s}";
    #""".format(aq))
    sql_query = text("SELECT artist, album, album_art FROM pitchfork LIMIT 20;")
    cur = db.engine.execute(sql_query)
    return cur.fetchall()
