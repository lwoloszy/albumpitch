import time
import argparse
import requests as r
import itertools as it
from bs4 import BeautifulSoup
from pymongo import MongoClient

import scrape_common_funcs as scf

BASE_URL = 'http://www.pitchfork.com'
HEADERS = {'User-Agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'}


def run(page_start=1, max_tries=10, overwrite=False):
    client = MongoClient()
    db = client['albumpitch']
    if 'pitchfork' in db.collection_names() and overwrite:
        db['pitchfork'].drop()
    coll = db['pitchfork']
    coll.create_index('url')
    try:
        empty_ctr = 0
        for page_num in it.count(page_start):
            review_links = scf.get_review_links(
                get_links_page, parse_links, page_num, max_tries)

            if empty_ctr > 10:
                print('10 consecutive requests with invalid response, exiting')
                break

            if not review_links:
                print('Unable to get review links from page {:d}'.
                      format(page_num))
                empty_ctr += 1
                continue
            else:
                empty_ctr = 0

            scf.get_insert_reviews(
                get_review_page, review_links, coll, max_tries)
    finally:
        client.close()


def get_links_page(page_num):
    session = r.Session()
    params = {'page': page_num}
    response = session.get(BASE_URL + '/reviews/albums/',
                           params=params, headers=HEADERS)
    return response


def parse_links(html):
    for parser in ['lxml', 'html5lib']:
        soup = BeautifulSoup(html, parser)
        album_anchors = soup.find_all('a', {'class': 'album-link'})
        review_links = [anchor.get('href').split('/')[-2]
                        for anchor in album_anchors]
        if review_links:
            break
    else:
        review_links = []

    return review_links


def get_review_page(review_link):
    session = r.Session()
    response = session.get(BASE_URL + '/reviews/albums/' + review_link,
                           headers=HEADERS)
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape Pitchfork site')
    parser.add_argument('--page_start', default=1, type=int,
                        help='page from which to start scrape')
    parser.add_argument('--overwrite', action='store_const',
                        const=True, default=False,
                        help='include flag to overwrite mongodb collection')
    parser.add_argument('--max_tries', default=10, type=int,
                        help='max number of retries when requesting htmls')

    args = vars(parser.parse_args())
    page_start = args['page_start']
    overwrite = args['overwrite']
    max_tries = args['max_tries']

    run(page_start, max_tries=max_tries, overwrite=overwrite)
