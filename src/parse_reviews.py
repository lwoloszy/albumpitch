import re
from pymongo import MongoClient
from datetime import datetime
from bs4 import BeautifulSoup


def parse_reviews(website):
    client = MongoClient()
    db = client['albumpitch']
    coll = db[website]
    for i, doc in enumerate(coll.find()):
        print('Parsed {:d} reviews'.format(i))
        doc_id = doc['_id']
        url = doc['url']
        for parser in ['lxml', 'html5lib']:
            try:
                if website == 'pitchfork':
                    parse_func = parse_pitchfork
                elif website == 'residentadvisor':
                    parse_func = parse_residentadvisor
                elif website == 'rollingstone':
                    parse_func = parse_rollingstone
                elif website == 'lineofbestfit':
                    parse_func = parse_lineofbestfit
                elif website == 'tinymixtapes':
                    parse_func = parse_tinymixtapes

                record = parse_func(doc['html'])

                coll.update_one(
                    {'_id': doc_id},
                    {
                        '$set': record,
                        '$currentDate': {'lastModified': True}
                    }
                )
                break
            except:
                continue
        else:
            print('Unable to parse review {:s}'.format(url))
    client.close()


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
    #    out['raw_article'] = article
    out['abstract'] = article.find('div', {'class': 'abstract'}).text
    out['review'] = article.find('div', {'class': 'contents dropcap'}).text

    # get album art link
    img_link = soup.find('img').get('src')
    out['album_art'] = img_link

    return out


def parse_residentadvisor(html):
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


def parse_rollingstone(html, parser):
    soup = BeautifulSoup(html, parser)
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


def parse_lineofbestfit(html, parser):
    soup = BeautifulSoup(html, parser)
    out = {}

    reviewer_fn = soup.find(itemprop='givenName')
    if reviewer_fn:
        reviewer_fn = reviewer_fn.text
    else:
        reviewer_fn = ''

    reviewer_ln = soup.find(itemprop='familyName')
    if reviewer_ln:
        reviewer_ln = reviewer_ln.text
    else:
        reviewer_ln = ''

    out['reviewer'] = reviewer_fn + ' ' + reviewer_ln

    artist = soup.find(itemprop='byArtist').text.strip()
    album = soup.find(class_='album-meta-title').text
    out['artist'] = artist
    out['album'] = album

    rating = soup.find(itemprop='ratingValue').text
    rating_max = soup.find(itemprop='bestRating').get('content')
    # rating_min = soup.find(itemprop='worstRating').text
    rating = rating + '/' + rating_max
    out['score'] = rating

    review = soup.find(class_='articlebody')
    out['review'] = review.text

    return out


def parse_tinymixtapes(html, parser):
    soup = BeautifulSoup(html, parser)
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
    try:
        label, year = label_and_year.split('; ')
        out['label'] = label
        out['year'] = year
    except:
        out['label_and_year'] = label_and_year

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
