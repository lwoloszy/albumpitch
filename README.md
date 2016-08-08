## Motivation
With the advent of online music streaming services like Spotify, Apple Music, Tidal, Pandora and others, we live in an age where it is incredibly easy to access any kind of music.  As the average listener’s curiosity and desire for new music has increased, streaming platforms have adopted a combination of human curators and computer algorithms to help create personalized playlists that they believe the listener will enjoy. The vast majority of such playlists consist of individual songs from a mishmash of artists. However, it is in the album format that artists originally intended their music to be listened. Call me a traditionalist, but I am one of these people who prefers the full album experience. So what’s an album-phile to do? Well, this past decade has seen numerous dedicated music review sites sprout up. I hypothesized that I could use the semantic content of these reviews to generate album suggestions. For this project in particular, I decided to pair Pitchfork, one of the better known and more consistent music blogs, with Spotify, to create AlbumPitch, an album recommendation engine based on text information.

## Data collection and storage
### Pitchfork site
The project began with scraping the entire Pitchfork site, which at the time had 17833 reviews spanning 9 genres. Some basic exploratory graphs summarizing the meta-information of these reviews are shown below. If you'd like to obtain this data yourself, simply run ```python src/scrapers/pitchfork_scraper.py``` from the root directory to get the raw htmls and then run ```python src/parse_reviews pitchfork``` to put the data in a nice, clean format. Note, you'll need MongoDB installed on your local machine to store and process the data.

### Spotify API
In order to run some validation tests, I used to Spotify's audio features API call. This was a multistage process where I first tried to find the album id in the Spotify catalog corresponding to the Pitchfork album whose audiofeatures I was interested in, then I would find all the track ids for that album, and finally I would retrieve the audio features for each and every track id. To get all this data, run ```python src/spotify.py get_track_features```

Oftentimes, the call to Spotify to find al album id would result in multiple albums being returned. I needed to find the best correspondence between the Pitchfork album that seeded the query and one of the returned Spotify albums, if any. To run this coregistration, which relied on several carefully handcrafted heuristics that you might want to play around with, run ```python src/spotify.py coregister_albums```

### PostgreSQL transfer
Ultimately, to put the Spotify and Pitchfork data into a more easily and quickly manipulable format (a format that was also directly transferable to Heroku), I put all this data into a PostgreSQL database. To do likewise on your local machine, run ```python mongo_to_sql.py```. Note, like with MongoDB, here you'll need a version of PostgreSQL installed on your local machine.

## Data analysis




