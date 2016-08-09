## Overview
This is a respository for the capstone project I did while attending Galvanize's Data Science Immersive class. In a nutshell, it is a music recommendation system that relies on text information gleaned from the web. It is a project that tries to capture all three phases of a typical data science product: data collection and storage, data analysis and modeling, and model deployment via a website. If you'd like to cut through the weeds, just visit albumpitch.herokuapp.com and start receiving recommendations! If you'd like to learn more about the process of how the project came to be, read on.

## Motivation
With the advent of online music streaming services like Spotify, Apple Music, Tidal, Pandora and others, we live in an age where it is incredibly easy to access any kind of music. As the average listener’s curiosity and desire for new music has increased, streaming platforms have adopted a combination of human curators and computer algorithms to help create personalized playlists that they believe the listener will enjoy. The vast majority of such playlists consist of individual songs from a mishmash of artists. However, it is in the album format that artists originally intended their music to be listened. Call me a traditionalist, but I am one of those people who prefers the full album experience. So what’s an album-phile like me to do? Well, this past decade has seen numerous dedicated music review sites come into existence. I figured that I could use the semantic content of these reviews to make album suggestions. For this project in particular, I decided to pair Pitchfork, one of the better known and more consistent music blogs, with Spotify, to create AlbumPitch, an album recommendation engine based solely on text information.

## Data collection and storage
#### Pitchfork site
As with any data science app, this project began data collection. I scraped the entire Pitchfork site, which at the time had 17833 reviews spanning 9 genres. Some basic exploratory graphs summarizing the review meta-information are shown below. If you'd like to obtain and play with this data yourself, simply run ```python src/scrapers/pitchfork_scraper.py``` from the root directory to get the raw htmls and then run ```python src/parse_reviews pitchfork``` to put the data in a nice, clean format. Note, you'll need MongoDB and pymongo installed on your local machine to store and process the data.

###### Figure 1 - Some very basic descriptive statistics of reviews scraped from Pitchfork.
![alt tag](https://raw.github.com/lwoloszy/albumpitch/master/figures/genre_dist.png) ![alt tag](https://raw.github.com/lwoloszy/albumpitch/master/figures/reviewer_dist.png) ![alt tag](https://raw.github.com/lwoloszy/albumpitch/master/figures/review_length_dist.png)

#### Spotify API
In order to run some validation tests, I used Spotify's audio features API call. This was a multistage process where I first tried to find the album id in the Spotify catalog corresponding to the Pitchfork album whose audio features I was interested in, then I would find all the track ids for that album, and finally I would retrieve the audio features for each and every track id. To get all this data, run ```python src/spotify.py get_track_features```. To do so, you'll need spotipy, a nice little python package to interact with the Spotify API.

Oftentimes, the call to Spotify to find an album id would result in multiple albums being returned. I needed to find the best correspondence between the Pitchfork album that seeded the query and one of the albums returned by Spotify, if any. To run this coregistration, which relied on several carefully handcrafted heuristics that you might want to play around with, run ```python src/spotify.py coregister_albums```. I'm not guaranteeing that this matching process is perfect, as there will be some false positive and false negatives, but I would say it's within 1-2% of what's achievable. Also, Spotify's catalog, while immense, is far from complete with respect to albums that Pitchfork has reviewed.

#### PostgreSQL transfer
Ultimately, to put the Spotify and Pitchfork data into a more quickly manipulable format, a format that was also directly transferable to Heroku, I put all this data into a PostgreSQL database. To do likewise on your local machine, run ```python mongo_to_sql.py```. Note, you'll need a version of PostgreSQL installed on your local machine as well as the psycopg2 driver.

## Data analysis
#### Primary models
My two main tools for generating album recommendation from text data were Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA), two algorithms famous for discovering hidden topics within a corpus of documents. I found from listening to A LOT of sample recommendations that in general LSA seemed to work better, so that's the method I stuck with. However, the bulk of my time to get either tool to work reasonably well was spent devising various regular expressions to capture some of the idiosyncracies of music reviews, which, naturally, included a lot of references to artists, bands and albums, which are multi-token patterns that would get lost in the simplest of bag-of-words approach. ```text_preprocess``` contains most of the regular expressions. One of my other breakthroughs in LSA was using sublinear scaling on the TF term, which helped counteract numerous mentions of one artist that could occur in some reviews, and thus increased the relative importance of some of the more descriptive words.

#### Validation
As I mentioned, earballing the recommendations was the primary means by which I evaluated the models, but I did have a few heuristics I used to gauge how well LSI was doing. For one, I visualized (again) A LOT of the hidden dimensions that LSA would produce, seeing whether they were capturing words relevant to discriminating various genres of music (where relevant was based on my own personal experience reading music reviews). One such plot is shown below. As you can see, many of the terms do cluster along obvious genres such as rock, rap, electronic and acoustic (though it should be acknowledged that many terms are also less obviously related to music per se).

###### Figure 2 - Top 10 latent semantic analysis components from a model with 200 dimensions. Each subplot shows a single hidden component discovered by LSA, with the 6 words having the heighest weight shown in red and the 6 words having the lowest weight shown in blue.
![alt tag](https://raw.github.com/lwoloszy/albumpitch/master/figures/svd.png)

We can go one step further and look at the clusters that k-means algorithm gives us when applied to data that has been transformed with LSI. Again, these seemed to make quite a bit of sense, reassuring me that the LSA approach in general was working.

###### Figure 3 - 10 random clusters discovered by KMeans algorithm (clustering was done in the LSI space). Each subplot shows a single cluster, with 12 words having the heighest weight shown.
![alt tag](https://raw.github.com/lwoloszy/albumpitch/master/figures/kmeans.png)

However, some form of external validation would be nice. For this, I turned to the track features that I got from Spotify (via Echonest). As background, Spotify has quantified for a large collection of songs a number of subjective features, such as acousticess, danceability, energy and so on. I figure that if my recommendations were making any sense, then the further down the recommendation list we go, the more dissimilar these albums should be to the album that initiated the query. Indeed, in the figure below, you can see that this monotonic increase in audio dissimilarity is present for all audio features examined, suggesting that the semantic content of music reviews has, to some degree, a relationship with audio features.

## Web app
To wrap this project , I built a little web app that you can visit at albumpitch.herokuapp.com that will produce, given either a seed album or a keyword search, a list of albums you might enjoy. It's not perfect by any means, as there are numerous improvements that could be made to the model, but in many instances it gives reasonable suggestions. Keep in mind, these recommendations are based solely on text information, so they're unlikely to be perfect.

## Future directions










