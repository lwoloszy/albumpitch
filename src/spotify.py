from __future__ import print_function

import re
import json
import time
from unidecode import unidecode

import pymongo
from pymongo import MongoClient

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

ASC = pymongo.ASCENDING


def get_track_features(
        creds_filepath='/Users/lukewoloszyn/.apikeys/spotify.json',
        max_retries=10):

    client = MongoClient()
    db = client['albumpitch']
    coll = db['pitchfork']

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

    coll_spotify_albums = db['spotify_albums']
    coll_spotify_albums.create_index(
        [('id', ASC), ('pitchfork_id', ASC)],
        unique=True)

    coll_audio_features = db['spotify_audio_features']
    coll_audio_features.create_index(
        [('id', ASC), ('album_id', ASC), ('pitchfork_id', ASC)],
        unique=True)

    # for i, doc in enumerate(coll.find(), 1):
    for i, doc in enumerate(coll.find({'spotify_found': {'$exists': 0}}), 1):
        if i % 50 == 0:
            print('Got audio features for {:d} albums'.format(i))

        artist = ' '.join(doc['artists'])
        album = doc['album']

        # spotify doesn't like the EP ending so remove it
        if album.split()[-1] == 'EP':
            album = ' '.join(album.split()[0:-1])

        album = re.sub(':', '', album)
        artist = re.sub(':', '', artist)
        try:
            artist = unidecode(artist)
            album = unidecode(album)
            query = 'artist:{:s} album:{:s}'.format(artist, album)
        except:
            print("Can't decode {:s}".format(query))
            continue

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
            with open('../logs/unable_to_search_album', 'a') as f:
                f.write(doc['url']+'\n')
            continue

        if not len(result['albums']['items']):
            with open('../logs/query_not_in_spotify_catalog_noep', 'a') as f:
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
                coll_spotify_albums.insert_one(album)
            except pymongo.errors.DuplicateKeyError:
                print('Duplicate album')
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

            # coll_audio_features.insert_many(audio_features)
            for track_af in track_afs:
                try:
                    coll_audio_features.insert_one(track_af)
                except pymongo.errors.DuplicateKeyError:
                    print('Duplicate track')
                    pass
            time.sleep(1)

        coll.update_one(
            {'_id': doc['_id']},
            {
                '$set': {'spotify_found': True},
                '$currentDate': {'lastModified': True}
            })

    client.close()


def coregister_albums():
    client = MongoClient()

    db = client['albumpitch']
    coll_pitchfork = db['pitchfork']
    coll_spotify_albums = db['spotify_albums']

    out1 = []
    out2 = []
    out3 = []

    for pitch_album in coll_pitchfork.find({'spotify_found': {'$exists': 1}}):
        cur = coll_spotify_albums.find({
            'pitchfork_url': pitch_album['url'],
            'pitchfork_id': pitch_album['review_id']
        })

        # ideal scenario
        if cur.count() == 1:
            album = cur.next()
        elif cur.count() == 2:
            a1, a2 = cur.next(), cur.next()

            if a1['name'] == a2['name']:

                # sometimes two albums with the exact same name exist but
                # one is a single and the other is a full album; take full
                album_types = [a1['album_type'], a2['album_type']]

                # sometimes two albums with the exact same name exist but one
                # is the explicit version and the other is not; take explicit
                n1_explic = sum([track['explicit']
                                 for track in a1['tracks']['items']])
                n2_explic = sum([track['explicit']
                                 for track in a2['tracks']['items']])

                # sometimes two albums with the exact same name exist but one
                # has more tracks than the other; take the one with bonus trcks
                n1_tracks = len(a1['tracks']['items'])
                n2_tracks = len(a2['tracks']['items'])

                if 'single' in album_types and 'album' in album_types:
                    d = {a1['album_type']: a1, a2['album_type']: a2}
                    album = d['album']
                elif n1_explic > n2_explic:
                    album = a1
                elif n2_explic > n1_explic:
                    album = a2
                elif n1_tracks > n2_tracks:
                    album = a1
                elif n2_tracks > n1_tracks:
                    album = a2
                else:
                    # otherwise, spotify has duplicate albums
                    # (different labels and whatnot, pick first one)
                    album = a1
            else:
                # do some magic to figure out which search result to use
                album = determine_best_match(pitch_album, [a1, a2])
                if not album:
                    out2.append((a1, a2, pitch_album['url'], pitch_album['artists']))
        else:
            albums = list(cur)
            album = determine_best_match(pitch_album, albums)
            if not album:
                out3.append((albums, pitch_album['url'], pitch_album['artists']))

        if album:
            coll_pitchfork.update_one(
                {'_id': pitch_album['_id']},
                {
                    '$set': {'putative_spotify_id': album['id']},
                    '$currentDate': {'lastModified': True}
                })

    client.close()
    return out1, out2, out3


def determine_best_match(pitch_album, spotify_albums):
    # try to see which album matches closest
    pitchfork_name = pitch_album['album']
    pitchfork_name = unidecode(pitchfork_name)
    pitchfork_name = pitchfork_name.lower().strip()
    if pitchfork_name.split()[-1] == 'ep':
        pitchfork_name = ' '.join(pitchfork_name.split()[0:-1])

    pitchfork_name = re.sub(r'&', 'and', pitchfork_name)
    pitchfork_name = ''.join(c for c in pitchfork_name if c.isalnum())

    a_artist = unidecode(' '.join(pitch_album['artists']).lower().strip())
    a_artist = ''.join(c for c in a_artist if c.isalnum())

    for album in spotify_albums:
        a_name = unidecode(album['name'].lower().strip())
        if a_name.split()[-1] == 'ep':
            a_name = ' '.join(a_name.split()[0:-1])

        a_name = re.sub(r'&', 'and', a_name)
        a_name = ''.join(c for c in a_name if c.isalnum())

        special_words = ['(from', 'special', 'version', 'expanded', 'bonus',
                         'deluxe', 'explicit', 'extended', 'anniversary',
                         'remaster', 'reissue', 'release', 'complete',
                         'edition)']
        regex = re.compile('|'.join(special_words))

        # doing the best i can here
        if pitchfork_name == a_name:
            return album
        elif regex.search(a_name):
            return album
        elif a_artist in a_name and 'djkicks' in a_name:
            return album
        else:
            print(pitchfork_name.encode('utf-8'), a_name.encode('utf-8'))

    return None
