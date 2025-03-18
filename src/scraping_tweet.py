# !pip install tweepy

# !pip install python-dotenv

import os
from dotenv import load_dotenv
import tweepy
import pandas as pd
import numpy as np

load_dotenv()

api_key = os.getenv("API_KEY_TWITTER")
api_secret = os.getenv("API_KEY_SECRET_TWITTER")
access_token = os.getenv("ACCESS_TOKEN_TWITTER")
access_token_secret = os.getenv("ACCESS_TOKEN_SECRET_TWITTER")
bearer_token = os.getenv("BEARER_TOKEN_TWITTER")

client = tweepy.Client(bearer_token=bearer_token)

hashtags = ["#fail", "#disappointed", "#unhappy", "#hate", "#annoyed", "#disgusted", "#offended", "#fuming", "#rage", "#trigger", "#revenge", "#neutral"]

query = " OR ".join(hashtags) + " -is:retweet lang:en"

try:
    tweets = client.search_recent_tweets(query=query, max_results=90, tweet_fields=["created_at", "text", "author_id"])

    if tweets.data:
        tweet_list = [[tweet.created_at, tweet.text] for tweet in tweets.data]

        df = pd.DataFrame(tweet_list, columns=["Date", "Tweet"])
        df.to_csv("tweets.csv", index=False)

        print("Tweets saved to tweets.csv!")

        for tweet in tweets.data:
            print(f"\n {tweet.text}\n{'-'*50}")
    else:
        print("No tweets found! Try different hashtags.")

except tweepy.errors.TweepyException as e:
    print(f" Error: {e}")

