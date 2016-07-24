import re
from datetime import datetime
from bs4 import BeautifulSoup


def parse_pitchfork(html, parser='lxml'):
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
    out['raw_article'] = article
    out['abstract'] = article.find('div', {'class': 'abstract'}).text
    out['review'] = article.find('div', {'class': 'contents dropcap'}).text

    return out
