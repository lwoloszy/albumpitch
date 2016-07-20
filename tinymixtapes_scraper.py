import re
import time
import argparse
import requests as r
import itertools as it
from datetime import datetime
from bs4 import BeautifulSoup
from pymongo import MongoClient

BASE_URL = 'http://www.tinymixtapes.com'
HEADERS = {'User-Agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'}


def run(page_start=1, max_tries=10, overwrite=False):
    client = MongoClient()
    db = client['album_reviews']
    if 'tinymixtapes' in db.collection_names() and overwrite:
        db['tinymixtapes'].drop()
    coll = db['tinymixtapes']
    coll.create_index('review_id')
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
    print('Getting links from page {:d}'.format(page_num))

    review_links = None
    for i in xrange(max_tries):
        response = get_links_page(page_num)
        if response.status_code == 200:
            review_links = parse_links(response.content)
            break
        else:
            print('Request for links on page {:d} failed, {:d} tries left'
                  .format(page_num, max_tries - i - 1))
            time.sleep(5)
    return review_links


def get_links_page(page_num):
    session = r.Session()
    params = {'page': page_num}
    response = session.get(BASE_URL + '/music-reviews',
                           params=params, headers=HEADERS)
    return response


def parse_links(html):
    # parse html to retrieve list of reviews available on current page
    soup = BeautifulSoup(html, 'lxml')
    album_anchors = soup.find_all('a', {'class': 'tile__link'})
    all_links = [aa.get('href') for aa in album_anchors]
    review_links = [link.split('/')[-1]
                    for link in all_links if re.match(
                            '/music-review/\w+', link)]
    return review_links


def get_insert_reviews(review_links, collection, max_tries=10):
    for review_link in review_links:
        for i in xrange(max_tries):
            response = get_review_page(review_link)
            if response.status_code == 200:
                # just use review_link as rid
                rid = review_link
                if collection.find({'review_id': rid}).count():
                    print('Review {:s} already exists'.format(rid))
                    break
                record = parse_review(response.content)
                record['review_id'] = review_link
                record['review_link'] = review_link
                collection.insert_one(record)
                break
            else:
                print('Request for review {:s} failed, {:d} tries left'
                      .format(review_link, max_tries - i - 1))
                time.sleep(5)


def get_review_page(review_link):
    session = r.Session()
    response = session.get(BASE_URL + '/music-review/' + review_link,
                           headers=HEADERS)
    return response


def parse_review(html):
    soup = BeautifulSoup(html, 'lxml')
    out = {}

    reviewer = soup.find(itemprop='author').find(itemprop='name').text
    out['reviewer'] = reviewer

    artist = soup.find(itemprop='byArtist').find(itemprop='name').text.strip()
    album = soup.find(class_='entry__subtitle').text.strip()
    out['artist'] = artist
    out['album'] = album

    rating = soup.find(itemprop='ratingValue').text
    rating_max = soup.find(itemprop='bestRating').text
    # rating_min = soup.find(itemprop='worstRating').text
    rating = rating + '/' + rating_max
    out['score'] = rating

    meta = soup.find(class_="review-heading__details-top")
    label_and_year = meta.find('p', class_='meta').text.strip('[]')
    label, year = label_and_year.split('; ')
    out['label'] = label
    out['year'] = year

    styles = soup.find(class_='review-heading__details-bottom')
    styles = styles.text.split('\n')[1].split(': ')[-1].split(',')
    styles = [style.strip() for style in styles]
    out['genres'] = styles

    review = soup.find(class_='entry__body-text')
    # the following removes track listing and such
    review.find(class_='u-hide').extract()
    out['review'] = review.text

    headings = soup.find_all(class_='heading__text')
    if any([heading.text == 'Eureka!' for heading in headings]):
        out['eureka'] = True

    return out


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
