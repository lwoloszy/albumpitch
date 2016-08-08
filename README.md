## Motivation
With the advent of online music streaming services like Spotify, Apple Music, Tidal, Pandora and others, we live in an age where it is incredibly easy to access any kind of music.  As the average listener’s curiosity and desire for new music has increased, streaming platforms have adopted a combination of human curators and computer algorithms to help create personalized playlists that they believe the listener will enjoy. The vast majority of such playlists consist of individual songs from a mishmash of artists. However, it is in the album format that artists originally intended their music to be listened. Call me a traditionalist, but I am one of these people who prefers the full album experience. So what’s an album-phile to do? Well, this past decade has seen numerous dedicated music review sites sprout up. I hypothesized that I could use the semantic content of these reviews to generate album suggestions. For this project in particular, I decided to pair Pitchfork, one of the better known and more consistent music blogs, with Spotify, to create AlbumPitch, an album recommendation engine based on text information.

## Data collection
The project began with scraping the entire Pitchfork site, which at the time had 17833 reviews spanning 9 genres, with each revie. Some basic exploratory graphs summarizing this information are shown below. If you'd like to obtain this information yourself, simply run python src/scrapers/pitchfork_scraper.py to get the raw data and then run the function parse_reviews (from module src/parse_reviews) with the argument set to 'pitchfork' to format the data. Note, you'll need MongoDB installed on your local machine to store and process the data.

## Data analysis



