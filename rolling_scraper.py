import re
import argparse
import time
import requests as r
from bs4 import BeautifulSoup
from pymongo import MongoClient
import itertools as it
from datetime import datetime

BASE_URL = 'http://www.rollingstone.com'
HEADERS = {'User-Agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'}


def run(page_start, max_tries=10, overwrite=False):
    client = MongoClient()
    db = client['album_reviews']
    if 'rolling_stone' in db.collection_names() and overwrite:
        db['rolling_stone'].drop()
    coll = db['rolling_stone']

    try:
        empty_ctr = 0
        for page_num in it.count(page_start):
            review_links = get_review_links(page_num, max_tries)

            if not review_links:
                print('Unable to get review links from page {:d}'.
                      format(page_num))
                empty_ctr += 1
                continue
            else:
                empty_ctr = 0

            if empty_ctr > 10:
                print('10 consecutive requests with invalid response, exiting')
                break

            get_insert_reviews(review_links, coll, max_tries)
    finally:
        client.close()


def get_review_links(page_num, max_tries=10):
    print('Getting links from page {:d}'
          .format(page_num))

    review_links = None
    for i in xrange(max_tries):
        response = get_links_page(page_num)
        if response.status_code == 200:
            review_links = parse_links(response.content)
            break
        else:
            print('Request for links on page {:d} failed, {:d} tries left'
                  .format(max_tries - i - 1))
            time.sleep(5)

    return review_links


def get_links_page(page_num):
    session = r.Session()
    params = {'page': page_num}
    response = session.get(BASE_URL+'/music/albumreviews',
                           params=params, headers=HEADERS)
    return response


def parse_links(html):
    # parse html to retrieve list of reviews available for month/year
    soup = BeautifulSoup(html, 'lxml')
    content_cards = soup.find_all('a', {'class': 'content-card-link'})
    review_links = [cc.get('href') for cc in content_cards]
    review_links = [review_link.split('/')[-1] for review_link in review_links]
    return review_links


def get_insert_reviews(review_links, collection, max_tries=10):
    for review_link in review_links:
        for i in xrange(max_tries):
            response = get_review_page(review_link)
            if response.status_code == 200:
                record = parse_review(response.content)
                rid = record['review_id']
                if not collection.find(
                        {'review_id': rid}).count():
                    collection.insert_one(record)
                else:
                    print('Review of {:s} already exists'
                          .format(rid))
                break
            else:
                print('Request for review {:s} failed, {:d} tries left'
                      .format(review_link, max_tries - i - 1))
                time.sleep(5)


def get_review_page(review_link):
    session = r.Session()
    response = session.get(BASE_URL + '/music/albumreviews/' + review_link,
                           headers=HEADERS)
    return response


def parse_review(html):
    soup = BeautifulSoup(html, 'lxml')
    out = {}

    out['review_id'] = (soup.find('link', {'rel': 'canonical'})
                        .get('href').split('/')[-1])

    pub_date = soup.find('time', {'class': 'content-published-date'}).text
    if pub_date.split()[-1] == 'ago':
        out['pub_date'] = datetime.today()
    else:
        out['pub_date'] = datetime.strptime(pub_date, '%B %d, %Y')
    reviewer = soup.find('a', {'class': 'content-author tracked-offpage'})
    if reviewer:
        out['reviewer'] = reviewer.text.strip()

    artist_and_album = soup.find('h1', {'class': 'content-title'}).text
    # artist, album = artist_and_album.split(': ')
    # out['artist'] = artist
    # out['album'] = album
    out['artist_and_album'] = artist_and_album

    d = {'full': 1, 'half': 0.5, 'empty': 0}
    stars = soup.find_all('span', {'class': 'ratings-star'})
    star_points = [d[star.find('span').get('class')[-1]] for star in stars]
    out['score'] = sum(star_points)

    abstract = soup.find('p', {'class': 'content-description'})
    if abstract:
        out['abstract'] = abstract.text

    content = soup.find('div', {'class': 'article-content'})
    # remove some unwanted text
    [div.extract()
     for div in content.find_all('div', class_='article-list')]
    [div.extract()
     for div in content.find_all('div', {'id': 'module-more-news'})]
    out['review'] = content.text

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape Rolling Stone site')
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
