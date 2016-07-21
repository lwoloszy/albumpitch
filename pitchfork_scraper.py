import re
import time
import argparse
import requests as r
import itertools as it
from datetime import datetime
from bs4 import BeautifulSoup
from pymongo import MongoClient

BASE_URL = 'http://www.pitchfork.com'
HEADERS = {'User-Agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'}


def run(page_start=1, max_tries=10, overwrite=False):
    client = MongoClient()
    db = client['album_reviews']
    if 'pitchfork' in db.collection_names() and overwrite:
        db['pitchfork'].drop()
    coll = db['pitchfork']
    coll.create_index('review_id')
    try:
        empty_ctr = 0
        for page_num in it.count(page_start):
            review_links = get_review_links(page_num, max_tries)

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
    response = session.get(BASE_URL + '/reviews/albums/',
                           params=params, headers=HEADERS)
    return response


def parse_links(html):
    # parse html to retrieve list of reviews available on current page
    soup = BeautifulSoup(html, 'lxml')
    album_anchors = soup.find_all('a', {'class': 'album-link'})
    review_links = [anchor.get('href').split('/')[-2]
                    for anchor in album_anchors]
    return review_links


def get_insert_reviews(review_links, collection, max_tries=10):
    for review_link in review_links:
        for i in xrange(max_tries):
            response = get_review_page(review_link)
            if response.status_code == 200:
                for parser in ['lxml', 'html5lib']:
                    try:
                        record = parse_review(response.content, parser)
                        break
                    except:
                        pass
                else:
                    print('Bad review{:s}, saving...'.format(review_link))
                    with open('data/'+review_link, 'w') as f:
                        f.write(response.content)
                    break
                record['review_link'] = review_link
                rid = record['review_id']
                if not collection.find({'review_id': rid}).count():
                    collection.insert_one(record)
                else:
                    print('Review {:d} already exists'.format(rid))
                break
            else:
                print('Request for review {:s} failed, {:d} tries left'
                      .format(review_link, max_tries - i - 1))
                time.sleep(5)


def get_review_page(review_link):
    session = r.Session()
    response = session.get(BASE_URL + '/reviews/albums/' + review_link,
                           headers=HEADERS)
    return response


def parse_review(html, parser='lxml'):
    soup = BeautifulSoup(html, parser)
    out = {}

    out['review_id'] = int(soup.find('article').get('id').split('-')[-1])

    reviewers = soup.find('ul', {'class': 'authors-detail'}).find_all('li')
    out['reviewers'] = [reviewer.find('a', {'class': 'display-name'}).text
                        for reviewer in reviewers]
    if len(out['reviewers']) != 1:
        print(len(reviewers))

    artists = soup.find('ul', {'class': 'artist-links artist-list'})
    out['artists'] = [artist.text for artist in artists.find_all('li')]
    out['album'] = soup.find('h1', {'class': 'review-title'}).text

    out['score'] = float(soup.find('span', {'class': 'score'}).text)
    bnm = soup.find('p', {'class': 'bnm-txt'})
    if bnm:
        if re.match('best new music', bnm.text, re.IGNORECASE):
            out['bnm'] = True
        elif re.match('best new reissue', bnm.text, re.IGNORECASE):
            out['bnr'] = True

    labels = soup.find('ul', {'class': 'label-list'})
    out['labels'] = [label.text for label in labels.find_all('li')]
    year = soup.find('span', {'class': 'year'}).text.split(' ')[-1]
    if year:
        out['year'] = year

    pub_date = soup.find('span', {'class': 'pub-date'}).text
    if pub_date[-3:] == 'ago':
        out['pub_date'] = datetime.today()
    else:
        out['pub_date'] = datetime.strptime(pub_date, '%B %d %Y')

    genres = soup.find('ul', {'class': 'genre-list'})
    out['genres'] = [genre.text for genre in genres.find_all('li')]

    article = soup.find('div', {'class': 'article-content'})
    out['abstract'] = article.find('div', {'class': 'abstract'}).text
    out['review'] = article.find('div', {'class': 'contents dropcap'}).text

    return out


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
