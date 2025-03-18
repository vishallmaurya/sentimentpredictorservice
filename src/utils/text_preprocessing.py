# pip install emoji

import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

all_stopwords = stopwords.words('english')
ps = PorterStemmer()

def extract_emojis(text):
    return ''.join(c for c in text if c in emoji.EMOJI_DATA)

def cleaning_tweets(tweet):
    extracted_emojis = extract_emojis(tweet)
    tweet = re.sub(r'\\p{So}', '', tweet)
    tweet = re.sub('[^a-zA-Z]', ' ', tweet)
    tweet = tweet.lower()
    tweet = tweet.split()

    tweet = [ps.stem(word) for word in tweet if word not in all_stopwords]
    tweet = ' '.join(tweet)

    return tweet, extracted_emojis

