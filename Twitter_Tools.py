import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


stop_words = set(stopwords.words('english'))

class Twitter:
 # Twitter Auth code snippet
    def __init__(self):
        # Access tokens from Twitter App console
        consumer_key = 'XXXXYOUNEEDTOHAVEMETOWORKXXXX'
        consumer_secret = 'XXXXXFILLMEINXXXXXXX'
        access_token = 'XXXXXHELPIMTRAPPEDINHEREXXXX'
        access_token_secret = 'XXXXXXPLEASESENDFORHELPSOSXXXX'

        # authenticate
        try:
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Could not Authenticate")

    def clean_tweet(self, tweet):
        # Function Pulled from Gaurav Singhal's python sentiment analysis.
        # https://www.pluralsight.com/guides/building-a-twitter-sentiment-analysis-in-python
        # Performs basic cleaning techniques for easier classification
        tweet.lower()
        # Remove urls
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        # Remove user @ references and '#' from tweet
        tweet = re.sub(r'\@\w+|\#', '', tweet)
        # Remove punctuations
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        # Remove stopwords
        tweet_tokens = word_tokenize(tweet)
        filtered_words = [w for w in tweet_tokens if not w in stop_words]
        # stem and lemmatize words, makes discrimination easier for textblob
        ps = PorterStemmer()
        stemmed_words = [ps.stem(w) for w in filtered_words]
        lemmatizer = WordNetLemmatizer()
        lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]

        #return " ".join(filtered_words)
        return " ".join(lemma_words)

    def get_tweets(self, query, count=10):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []

        try:
            # call twitter api to fetch tweets.
            # twitter doesn't return the total possible tweets within a specific query, but we can keep track
            # of tweet ID's we've already queried and just repeatedly ask twitter for more tweets until we're satisfied

            max_queries = 100  # arbitrarily chosen value
            fetched_tweets = tweet_batch = self.api.search(q=query, count=count)
            ct = 1
            while len(fetched_tweets) < count and ct < max_queries:
                tweet_batch = self.api.search(q=query,
                                              count=count - len(tweets),
                                              max_id=tweet_batch.max_id)
                fetched_tweets.extend(tweet_batch)
                ct += 1
            print("Fetched a total of " + str(len(fetched_tweets)) + " tweets for ticker " + query)

            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}
                # saving date of tweet
                parsed_tweet['created_at'] = tweet.created_at.date()
                # saving text features of tweet
                parsed_tweet['text'] = tweet.text
                # saving sentiment of tweet
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)

                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)

            # return parsed tweets
            return tweets

        except tweepy.TweepError as e:
            print("Error : " + str(e))

    def get_tweet_sentiment(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        NLP_result = TextBlob(self.clean_tweet(tweet))

        return NLP_result.sentiment.polarity


def query_twitter(query, max_tweets=5000):
    # Takes a specific twitter query and returns a 7 day daily interval of tweets
    # Returns a pandas dataframe of aggregate sentiment score, total tweets, and total positive/negative tweets
    api = Twitter()
    tweets = api.get_tweets(query=query, count=max_tweets)
    agg_sentiment_score = {}
    num_tweets_on_date = {}
    positive_tweets = {}
    negative_tweets = {}

    for tweet in tweets:

        sentiment_score = tweet['sentiment']
        # for initializing a dictionary with a date key
        if agg_sentiment_score.get(tweet['created_at']) is None:
            agg_sentiment_score[tweet['created_at']] = sentiment_score
            num_tweets_on_date[tweet['created_at']] = 1

            positive_tweets[tweet['created_at']] = 0  # Assign 0 to avoid key error in inner boolean operation
            negative_tweets[tweet['created_at']] = 0

            if sentiment_score > 0:
                positive_tweets[tweet['created_at']] = 1
            elif sentiment_score < 0:
                negative_tweets[tweet['created_at']] = 1

        # if the date key is already initialized, aggregate new sentiment on the key
        else:
            agg_sentiment_score[tweet['created_at']] += sentiment_score
            num_tweets_on_date[tweet['created_at']] += 1

            if sentiment_score > 0:
                positive_tweets[tweet['created_at']] += 1
            elif sentiment_score < 0:
                negative_tweets[tweet['created_at']] += 1

    # Return a pandas dataframe instead of a dict
    agg_sentiment_score = pd.DataFrame.from_dict(agg_sentiment_score, orient='index')
    num_tweets_on_date = pd.DataFrame.from_dict(num_tweets_on_date, orient='index')
    positive_tweets = pd.DataFrame.from_dict(positive_tweets, orient='index')
    negative_tweets = pd.DataFrame.from_dict(negative_tweets, orient='index')

    # Return a combined dataframe with datetime as index
    agg_sentiment_score = agg_sentiment_score.join(num_tweets_on_date, lsuffix="agg")
    agg_sentiment_score = agg_sentiment_score.join(positive_tweets, lsuffix="pos")
    agg_sentiment_score = agg_sentiment_score.join(negative_tweets, lsuffix="neg")
    agg_sentiment_score.columns = ['Aggregate Sentiment','Number of Tweets','Positive Tweets','Negative Tweets']
    agg_sentiment_score.index.names = ['Date']
    agg_sentiment_score.index = pd.to_datetime(agg_sentiment_score.index)

    return agg_sentiment_score
