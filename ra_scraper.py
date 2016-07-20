import re
import argparse
import time
import requests as r
from bs4 import BeautifulSoup
from pymongo import MongoClient

BASE_URL = 'http://www.residentadvisor.net'
HEADERS = {'User-Agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'}


def run(year_start, month_start, format_='album',
        max_tries=10, overwrite=False):
    client = MongoClient()
    db = client['album_reviews']
    if 'resident_advisor' in db.collection_names() and overwrite:
        db['resident_advisor'].drop()
    coll = db['resident_advisor']
    coll.create_index('review_id')

    try:
        empty_ctr = 0
        for year in xrange(year_start, 2017):
            for month in xrange(month_start, 13):
                review_links = get_review_links(year, month,
                                                format_, max_tries)

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

                get_insert_reviews(review_links, coll, max_tries)
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
    session = r.Session()
    params = {'format': format_, 'yr': year, 'mn': month}
    response = session.get(BASE_URL+'/reviews.aspx',
                           params=params, headers=HEADERS)
    return response


def parse_links(html):
    # parse html to retrieve list of reviews available for month/year
    soup = BeautifulSoup(html, 'lxml')
    articles = soup.find_all('article')
    review_links = [article.find('a').get('href')
                    for article in articles]
    return review_links


def get_insert_reviews(review_links, collection, max_tries=10):
    for review_link in review_links:
        for i in xrange(max_tries):
            response = get_review_page(review_link)
            if response.status_code == 200:
                record = parse_review(response.content)
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
    response = session.get(BASE_URL + review_link, headers=HEADERS)
    return response


def parse_review(html):
    soup = BeautifulSoup(html, 'lxml')
    out = {}

    temp = soup.find('a', {'class': 'comment-link'}).get('href').split('?')[0]
    review_id = int(temp.split('/')[-1])
    out['review_id'] = review_id

    album_type = soup.find(
        'a', text=re.compile('(Singles|Albums)', re.IGNORECASE)).text
    if album_type == 'Singles':
        out['EP'] = True
    elif album_type == 'Albums':
        out['LP'] = True

    album_and_artist = soup.find('h1').text
    out['album_and_artist'] = album_and_artist
    out['score'] = soup.find('span', {'class': 'rating'}).text

    label_div = soup.find('div', text=re.compile('Label ', re.IGNORECASE))
    if label_div:
        label = label_div.findNextSibling().text
        out['label'] = label

    released_div = soup.find('div', text=re.compile('Released ', re.IGNORECASE))
    if released_div:
        year_search = re.search(r'\d{4}', released_div.findParent().text)
        if year_search:
            out['year'] = year_search.group(0)

    style_div = soup.find('div', text=re.compile('Style ', re.IGNORECASE))
    if style_div:
        genres = style_div.findParent().text.split('\n')[-2]
        genres = [genre.strip() for genre in genres.split(',')]
        out['genres'] = genres

    reviewer = soup.find('a', {'rel': 'author'})
    if reviewer:
        out['reviewer'] = reviewer.text

    review = soup.find('div', {'class': 'reviewContent'})
    out['review'] = review.find('span', {'itemprop': 'description'}).text

    return out


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
