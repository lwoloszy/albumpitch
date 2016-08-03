from pymongo import MongoClient
import psycopg2


def insert_pitchfork_reviews():
    conn = psycopg2.connect(database='albumpitch', user='lukewoloszyn')
    cur = conn.cursor()

    client = MongoClient()
    db = client['albumpitch']
    coll = db['pitchfork']

    try:
        for doc in coll.find():
            pid = doc['review_id']
            url = doc['url']

            spotify_id = doc.get('putative_spotify_id', None)

            artist = ' / '.join(doc['artists'])
            album = doc['album']
            genres = ', '.join(doc.get('genres', [None]))

            pub_date = doc['pub_date']
            reviewers = ', '.join(doc['reviewers'])

            labels = ', '.join(doc['labels'])
            year = doc.get('year', None)

            score = doc['score']
            abstract = doc['abstract']
            review = doc['review']

            album_art = doc['album_art']

            SQL = """
            INSERT INTO pitchfork (id, url, spotify_id, artist, album, genres,
            pub_date, reviewers, labels, year, score, abstract, review, album_art)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
            """
            data = (pid, url, spotify_id, artist, album, genres,
                    pub_date, reviewers, labels, year, score, abstract,
                    review, album_art)
            cur.execute(SQL, data)
        conn.commit()
    finally:
        conn.close()
        client.close()


def insert_audio_features():
    conn = psycopg2.connect(database='albumpitch', user='lukewoloszyn')
    cur = conn.cursor()

    client = MongoClient()
    db = client['albumpitch']
    coll = db['spotify_audio_features']

    try:
        for doc in coll.find():
            if 'id' not in doc:
                print('Track with no id, skipping')
                continue
            tid = doc['id']
            album_id = doc['album_id']
            acoustic = doc['acousticness']
            dance = doc['danceability']
            energy = doc['energy']
            instrument = doc['instrumentalness']
            key = doc['key']
            liveness = doc['liveness']
            loudness = doc['loudness']
            mode = doc['mode']
            speechiness = doc['speechiness']
            tempo = doc['tempo']
            ts = doc['time_signature']
            valence = doc['valence']
            SQL = """
            INSERT INTO spotify_audio_features (id, album_id, acousticness,
            danceability, energy, instrumentalness, key, liveness, loudness,
            mode, speechiness, tempo, time_signature, valence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
            """
            data = (tid, album_id, acoustic, dance, energy, instrument, key,
                    liveness, loudness, mode, speechiness, tempo, ts, valence)
            cur.execute(SQL, data)
        conn.commit()
    finally:
        conn.close()
        client.close()


def insert_spotify_albums():
    conn = psycopg2.connect(database='albumpitch', user='lukewoloszyn')
    cur = conn.cursor()

    client = MongoClient()
    db = client['albumpitch']
    coll = db['spotify_albums']

    try:
        for doc in coll.find():
            album_id = doc['id']
            album_name = doc['name']
            SQL = """
            INSERT INTO spotify_albums (id, name)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING;
            """
            data = (album_id, album_name)
            cur.execute(SQL, data)
        conn.commit()
    finally:
        conn.close()
        client.close()


def create_table_pitchfork():
    conn = psycopg2.connect(database='albumpitch', user='lukewoloszyn')
    cur = conn.cursor()
    cur.execute("""
    DROP TABLE IF EXISTS pitchfork CASCADE;
    CREATE TABLE pitchfork (
        id varchar(255) NOT NULL primary key,
        url varchar(255) NOT NULL,
        spotify_id varchar(255) default NULL REFERENCES spotify_albums (id),

        artist varchar(255) NOT NULL,
        album varchar(255) NOT NULL,
        genres varchar(255) default NULL,

        pub_date timestamp default NULL,
        reviewers varchar(255) default NULL,

        labels varchar(255) default NULL,
        year varchar(255) default NULL,

        album_art varchar(255) default NULL,

        score int NOT NULL,
        abstract text default NULL,
        review text NOT NULL);
    """)
    conn.commit()
    conn.close()


def create_table_audio_features():
    conn = psycopg2.connect(database='albumpitch', user='lukewoloszyn')
    cur = conn.cursor()
    cur.execute("""
    DROP TABLE IF EXISTS spotify_audio_features CASCADE;
    CREATE TABLE spotify_audio_features (
        id varchar(255) NOT NULL primary key,
        album_id varchar(255) NOT NULL REFERENCES spotify_albums (id),
        acousticness real default NULL,
        danceability real default NULL,
        energy real default NULL,
        instrumentalness real default NULL,
        key int default NULL,
        liveness real default NULL,
        loudness real default NULL,
        mode int default NULL ,
        speechiness real default NULL,
        tempo real default NULL,
        time_signature int default NULL,
        valence real default NULL);
    """)
    conn.commit()
    conn.close()


def create_table_albums():
    conn = psycopg2.connect(database='albumpitch', user='lukewoloszyn')
    cur = conn.cursor()
    cur.execute("""
    DROP TABLE IF EXISTS spotify_albums CASCADE;
    CREATE TABLE spotify_albums (
        id varchar(255) NOT NULL primary key,
        name varchar(255) NOT NULL);
    """)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    create_table_albums()
    print('Created table spotify_albums')
    create_table_audio_features()
    print('Created table spotify_audio_features')
    create_table_pitchfork()
    print('Created table pitchfork')
    insert_spotify_albums()
    print('Inserted into spotify_albums')
    insert_audio_features()
    print('Inserted into spotify_audio_features')
    insert_pitchfork_reviews()
    print('Inserted into pitchfork')
