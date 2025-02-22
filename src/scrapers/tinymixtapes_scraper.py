import re
import argparse
import requests as r
import itertools as it
from bs4 import BeautifulSoup
from pymongo import MongoClient

import scrape_common_funcs as scf

BASE_URL = 'http://www.tinymixtapes.com'
HEADERS = {'User-Agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'}


def run(page_start=1, max_tries=10, overwrite=False):
    """
    Gets all reviews from tinymixtapes and stores in MongoDB
    albumpitch database

    Args:
        page_start: the page from which to start scrape process (default=1)
        max_retries: maximum number of requests to issue before
                     moving on (default=10)
        overwrite: whether to overwrite existing collection in MongoDB database
                   (default=False)
    Returns:
        None
    """

    client = MongoClient()
    db = client['album_reviews']
    if 'tinymixtapes' in db.collection_names() and overwrite:
        db['tinymixtapes'].drop()
    coll = db['tinymixtapes']
    coll.create_index('review_id')
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
    """
    Retrieves the raw html that contains the links to a set of
    albums

    Args:
        page_num: the page number to retrieve
    Returns:
        response: a requests response object
    """
    session = r.Session()
    params = {'page': page_num}
    response = session.get(BASE_URL + '/music-reviews',
                           params=params, headers=HEADERS)
    return response


def parse_links(html):
    """
    Parses a raw html using BeautifulSoup to extract out the set
    of links that will take us to the available reviews

    Args:
        html: the raw html
    Returns:
        review_links: list with the review links
    """

    # parse html to retrieve list of reviews available on current page
    soup = BeautifulSoup(html, 'lxml')
    album_anchors = soup.find_all('a', {'class': 'tile__link'})
    all_links = [aa.get('href') for aa in album_anchors]
    review_links = [link.split('/')[-1]
                    for link in all_links if re.match(
                            '/music-review/\w+', link)]
    return review_links


def get_review_page(review_link):
    """
    Retrieves raw html of one review

    Args:
        review_link: a string that is the link to the review
    Returns:
        response: a requests response object
    """

    session = r.Session()
    response = session.get(BASE_URL + '/music-review/' + review_link,
                           headers=HEADERS)
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape Tinymixtapes site')
    parser.add_argument('--page_start', default=0, type=int,
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
