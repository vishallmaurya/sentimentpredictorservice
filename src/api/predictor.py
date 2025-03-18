# pip install FastAPI emoji uvicorn pyngrok pydantic

from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import emoji
import os
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from src.utils.text_preprocessing import cleaning_tweets

app = FastAPI()

clf = pickle.load(open(os.path.join(os.path.dirname(__file__), "../../model.pkl"), "rb"))
vectorizer = SentenceTransformer("all-MiniLM-L6-v2")

class TweetInput(BaseModel):
    tweet: str

def emoji_to_text(emoji_string):
    return ' '.join(emoji.demojize(e) for e in emoji_string) if emoji_string else ''

@app.post("/predict/")
async def predict_sentiment(tweet_input: TweetInput):
  try:
    tweet = tweet_input.tweet
    clean_text, extracted_emojis = cleaning_tweets(tweet)

    emoji_text = emoji_to_text(extracted_emojis)

    text_embedding = vectorizer.encode([clean_text], convert_to_numpy=True)
    embedding_dim = vectorizer.get_sentence_embedding_dimension()

    emoji_embedding = [
        vectorizer.encode(emoji_text, convert_to_numpy=True)
        if emoji_text.strip()
        else np.zeros(embedding_dim)
    ]

    text_features = np.hstack((text_embedding, emoji_embedding))

    sentiment_class = clf.predict(text_features)[0]

    return {"tweet": tweet, "predicted_class": int(sentiment_class)}
  except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# from predictor import app

# !ngrok authtoken 2uTTXXiQsbS2OXEh6NZe5KExFqJ_UoGVYBpiu1KuNq9U8Rn2

# from pyngrok import ngrok
# import uvicorn
# from predictor import app


# public_url = ngrok.connect(8000).public_url
# print(f"Public URL: {public_url}")
# !uvicorn predictor:app --host 0.0.0.0 --port 8000 &