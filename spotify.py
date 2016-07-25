from __future__ import print_function
import json
import spotipy
import time
from spotipy.oauth2 import SpotifyClientCredentials
import pymongo
from pymongo import MongoClient

ASC = pymongo.ASCENDING


def get_track_features(
        creds_filepath='/Users/lukewoloszyn/.apikeys/spotify.json',
        max_retries=10):

    client = MongoClient()
    db = client['album_reviews']
    coll = db['pitchfork_full']

    with open(creds_filepath) as f:
        creds = json.loads(f.read())
        my_id = creds['id']
        my_secret = creds['secret']

    ccm = SpotifyClientCredentials(
        client_id=my_id,
        client_secret=my_secret)

    # sp = spotipy.Spotify(auth=access_token)
    sp = spotipy.Spotify(client_credentials_manager=ccm)
    sp.trace = False

    db_spotify = client['spotify_info']

    coll_album_info = db_spotify['album_meta']
    coll_album_info.create_index(
        [('id', ASC), ('pitchfork_id', ASC)],
        unique=True)

    coll_track_afs = db_spotify['audio_features']
    coll_track_afs.create_index(
        [('id', ASC), ('album_id', ASC), ('pitchfork_id', ASC)],
        unique=True)

    for i, doc in enumerate(coll.find(), 1):
        if i % 50 == 0:
            print('Got audio features for {:d} albums'.format(i))

        artist = ' '.join(doc['artists']).encode('utf-8')
        album = doc['album'].encode('utf-8')
        query = 'artist:{:s} album:{:s}'.format(artist, album)

        for j in xrange(max_retries):
            try:
                result = sp.search(query, type='album')
                break
            except:
                print('Query for album {:s} failed, {:d} retries left'
                      .format(doc['url'], max_retries-j-1))
                time.sleep(5)
                continue
        else:
            with open('logs/unable_to_search_album', 'a') as f:
                f.write(doc['url']+'\n')
            continue

        if not len(result['albums']['items']):
            with open('logs/query_not_in_spotify_catalog', 'a') as f:
                f.write(doc['url']+'\n')
            continue

        albums = result['albums']['items']

        for j, album in enumerate(albums):
            album['pitchfork_id'] = doc['review_id']
            album['pitchfork_url'] = doc['url']
            album['result_number'] = j

            album_id = album['id']
            tracks = sp.album_tracks(album_id)
            album['tracks'] = tracks
            try:
                coll_album_info.insert_one(album)
            except pymongo.errors.DuplicateKeyError:
                pass

            # now get audio features for all tracks in album
            track_ids = [track['id'] for track in tracks['items']]
            track_afs = sp.audio_features(tracks=track_ids)
            # replace empty tracks with dict (spotify error?)
            track_afs = [track_af if track_af else {'track_corrupt': True}
                         for track_af in track_afs]
            for track_af in track_afs:
                track_af['pitchfork_id'] = doc['review_id']
                track_af['pitchfork_url'] = doc['url']
                track_af['album'] = doc['album']
                track_af['artist'] = doc['artists']
                track_af['album_id'] = album_id

            try:
                coll_track_afs.insert_many(track_afs)
            except pymongo.errors.DuplicateKeyError:
                pass
            time.sleep(1)

        coll.update_one(
            {'_id': doc['_id']},
            {
                '$set': {'spotify': True},
                '$currentDate': {'lastModified': True}
            })

    client.close()
