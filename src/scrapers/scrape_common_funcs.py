import time


def get_insert_reviews(scrape_func, review_links, collection, max_tries=10):
    for review_link in review_links:
        for i in xrange(max_tries):
            response = scrape_func(review_link)
            if response.status_code == 200:
                document = {}
                document['url'] = review_link
                document['html'] = response.content
                if not collection.find({'url': review_link}).count():
                    collection.insert_one(document)
                else:
                    print('Review {:s} already exists'.format(review_link))
                break
            else:
                print('Request for review {:s} failed, {:d} tries left'
                      .format(review_link, max_tries - i - 1))
                time.sleep(5)


def get_review_links(scrape_func, parse_func, page_num, max_tries=10):
    if not isinstance(page_num, int):
        print('Getting links from page {:s}'.format(
            '-'.join([str(i) for i in page_num])))
    else:
        print('Getting links from page {:d}'.format(page_num))

    review_links = None
    for i in xrange(max_tries):
        if not isinstance(page_num, int):
            response = scrape_func(*page_num)
        else:
            response = scrape_func(page_num)
        if response.status_code == 200:
            review_links = parse_func(response.content)
            break
        else:
            print('Request for links on page {:d} failed, {:d} tries left'
                  .format(page_num, max_tries - i - 1))
            time.sleep(5)
    return review_links
