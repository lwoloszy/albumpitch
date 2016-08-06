import argparse
import time
import requests as r
import functools
from bs4 import BeautifulSoup
from pymongo import MongoClient

import scrape_common_funcs as scf


BASE_URL = 'http://www.residentadvisor.net'
HEADERS = {'User-Agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'}


def run(year_start, month_start, format_='album',
        max_tries=10, overwrite=False):
    """
    Gets all reviews from residentadvisor and stores in MongoDB
    albumpitch database

    Args:
        year_start: the year from which to start scraping
        month_start: the month from which to start scraping
        format_: 'single' or 'album'
        max_retries: maximum number of requests to issue before
                     moving on (default=10)
        overwrite: whether to overwrite existing collection in MongoDB database
                   (default=False)
    Returns:
        None
    """

    client = MongoClient()
    db = client['albumpitch']
    if 'residentadvisor' in db.collection_names() and overwrite:
        db['residentadvisor'].drop()
    coll = db['residentadvisor']
    coll.create_index('url')

    try:
        empty_ctr = 0
        for year in xrange(year_start, 2017):
            for month in xrange(month_start, 13):
                review_links = scf.get_review_links(
                    functools.partial(get_links_page, format_=format_),
                    parse_links, (year, month), max_tries)

                if empty_ctr > 10:
                    print('10 consecutive requests with invalid response, exiting')
                    break

                if not review_links:
                    print('Unable to get review links from year {:d}, month {:d}'.
                          format(year, month))
                    empty_ctr += 1
                    continue
                else:
                    empty_ctr = 0

                scf.get_insert_reviews(
                    get_review_page, review_links, coll, max_tries)
    finally:
        client.close()


def get_review_links(year, month, format_='album', max_tries=10):
    print('Getting links from year {:d}, month {:d}'
          .format(year, month))

    review_links = None
    for i in xrange(max_tries):
        response = get_links_page(year, month, format_)
        if response.status_code == 200:
            review_links = parse_links(response.content)
            break
        else:
            print('Request for links on page {:d} failed, {:d} tries left'
                  .format(max_tries - i - 1))
            time.sleep(5)

    return review_links


def get_links_page(year, month, format_):
    """
    Retrieves the raw html that contains the links to a set of
    albums

    Args:
        page_num: the page number to retrieve
    Returns:
        response: a requests response object
    """

    session = r.Session()
    params = {'format': format_, 'yr': year, 'mn': month}
    response = session.get(BASE_URL+'/reviews.aspx',
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

    soup = BeautifulSoup(html, 'lxml')
    articles = soup.find_all('article')
    review_links = [article.find('a').get('href')
                    for article in articles]
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
    response = session.get(BASE_URL + review_link, headers=HEADERS)
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape Resident Advisor site')
    parser.add_argument('--year', default=2003, type=int,
                        help='year from which to start scrape')
    parser.add_argument('--month', default=1, type=int,
                        help='month from which to start scrape')
    parser.add_argument('--overwrite', action='store_const',
                        const=True, default=False,
                        help='include flag to overwrite mongodb collection')
    parser.add_argument('--max_tries', default=10, type=int,
                        help='max number of retries when requesting htmls')
    parser.add_argument('--format', default='album', type=str,
                        help='format which to scrape (album or single)')

    args = vars(parser.parse_args())
    year = args['year']
    month = args['month']
    overwrite = args['overwrite']
    max_tries = args['max_tries']
    format_ = args['format']

    run(year, month, format_=format_, max_tries=max_tries, overwrite=overwrite)
