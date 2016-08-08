## Motivation
With the advent of online music streaming services like Spotify, Apple Music, Tidal, Pandora and others, we live in an age where it is incredibly easy to access any kind of music.  As the average listener’s curiosity and desire for new music has increased, streaming platforms have adopted a combination of human curators and computer algorithms to help create personalized playlists that they believe the listener will enjoy. The vast majority of such playlists consist of individual songs from a mishmash of artists. However, it is in the album format that artists originally intended their music to be listened. Call me a traditionalist, but I am one of these people who prefers the full album experience. So what’s an album-phile to do? Well, this past decade has seen numerous dedicated music review sites come into existence. I hypothesized that I could use the semantic content of these reviews to make album suggestions. For this project in particular, I decided to pair Pitchfork, one of the better known and more consistent music blogs, with Spotify, to create AlbumPitch, an album recommendation engine based on text information.

## Data collection and storage
### Pitchfork site
As with any data science app, the project began with scraping the entire Pitchfork site, which at the time had 17833 reviews spanning 9 genres. Some basic exploratory graphs summarizing the meta-information of these reviews are shown below. If you'd like to obtain and play with this data yourself, simply run ```python src/scrapers/pitchfork_scraper.py``` from the root directory to get the raw htmls and then run ```python src/parse_reviews pitchfork``` to put the data in a nice, clean format. Note, you'll need MongoDB and pymongo installed on your local machine to store and process the data.

### Spotify API
In order to run some validation tests, I used Spotify's track features API call. This was a multistage process where I first tried to find the album id in the Spotify catalog corresponding to the Pitchfork album whose audio features I was interested in, then I would find all the track ids for that album, and finally I would retrieve the audio features for each and every track id. To get all this data, run ```python src/spotify.py get_track_features```. To do so, you'll need spotipy, a nice little python package to interact with the Spotify API.

Oftentimes, the call to Spotify to find an album id would result in multiple albums being returned. I needed to find the best correspondence between the Pitchfork album that seeded the query and one of the returned Spotify albums, if any. To run this coregistration, which relied on several carefully handcrafted heuristics that you might want to play around with, run ```python src/spotify.py coregister_albums```. I'm not guaranteeing that this is perfect, as there might be some false positive and false negatives in the matching process, but I would say it's within 1-2% of what's achievable (Spotify's catalog, while immense, if not entirely complete with regard to Pitchfork's reviews).

### PostgreSQL transfer
Ultimately, to put the Spotify and Pitchfork data into a more easily and quickly manipulable format (a format that was also directly transferable to Heroku), I put all this data into a PostgreSQL database. To do likewise on your local machine, run ```python mongo_to_sql.py```. Note, here you'll need a version of PostgreSQL installed on your local machine as well as the psycopg2 package.

## Data analysis
### Primary models
My two main tools of attack at generating album recommendation from text data were Latent Semantic Analysis and Latent Dirichlet Allocation. I found from listening to A LOT of recommendations that in general LSA seemed to work better, so that the method that stuck. However, the bulk of my time to get either tool to work reasonably well was spent devising various regular expressions to capture some of the idiosyncracies of music reviews, which, naturally, included a lot of references to artists, bands and albums, which are multi-token patterns that would get lost in simplest bag-of-words approach. ```text_preprocess``` contains most of the regular expressions. One of my other breakthroughs in LSA was using sublinear scaling on the TF term, which helped counteract numerous mentions of one artist which could occur in some reviews, and thus increased the relative importance of some of the more descriptive words.

### Validation
As I mentioned, earballing the recommendations was the primary means by which I evaluated the models, but I did have a few heuristics I used to asses how well LSI was doing. For one, I visualized a lot of the hidden dimensons that LSA would generate, seeing whether they were capturing words relevant to discriminating various genres of music. One example such plot is shown below. We can go one step further and look at the clusters that KMeans algorithm gives us when applied to data that has been transformed with LSI. Again, these seemed to make quite a bit of sense, reassuring me that the approach in general was working.

However, some form of external validation would be nice. For this, I turned to the track features that I got from Spotify (via Echonest). As background, Spotify has quantified for a large collection of songs a number of subjective features, such as acousticess, danceability, energy and so on. I figure that if my recommendations were making any sense, then the further down the recommendation list we go, the more dissimilar these albums should be to the album that initiated the query. Indeed, in the figure below, you can see that this monotonic increase in audio dissimilarity is present for audio features examined, suggesting that the semantic content of music reviews has, to some degree, a relationship with audio features.

## Web app
To culminate this project, I built a little web app that you can visit at that will, given either a seed album or a keyword search, produce a list of albums you're likely to enjoy. It's not perfect by any means, as there are numerous improvements that could be made to the model, but in many instances it gives reasonable suggestions. Keep in mind, these recommendations are based solely on text information, so they're unlikely to be perfect.

## Future directions










