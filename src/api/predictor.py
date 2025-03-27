# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import pickle
# import numpy as np
# import emoji
# import os
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
# from src.utils.text_preprocessing import cleaning_tweets

# load_dotenv()
# app = FastAPI()

# origins = [
#     os.getenv("FRONTEND_CORS"),
#     os.getenv("BACKEND_CORS"),
#     os.getenv("LOCALHOST_CORS"),
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,  # Allow only these origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all HTTP methods
#     allow_headers=["*"],  # Allow all headers
# )

# clf = pickle.load(open(os.path.join(os.path.dirname(__file__), "../../model.pkl"), "rb"))
# vectorizer = SentenceTransformer("all-MiniLM-L6-v2")

# class TweetInput(BaseModel):
#     tweet: str

# def emoji_to_text(emoji_string):
#     return ' '.join(emoji.demojize(e) for e in emoji_string) if emoji_string else ''

# @app.post("/predict/")
# async def predict_sentiment(tweet_input: TweetInput):
#     try:
#         tweet = tweet_input.tweet
#         clean_text, extracted_emojis = cleaning_tweets(tweet)

#         emoji_text = emoji_to_text(extracted_emojis)

#         text_embedding = vectorizer.encode([clean_text], convert_to_numpy=True)
#         embedding_dim = vectorizer.get_sentence_embedding_dimension()

#         emoji_embedding = [
#             vectorizer.encode(emoji_text, convert_to_numpy=True)
#             if emoji_text.strip()
#             else np.zeros(embedding_dim)
#         ]

#         text_features = np.hstack((text_embedding, emoji_embedding))

#         sentiment_class = clf.predict(text_features)[0]

#         return {"tweet": tweet, "predicted_class": int(sentiment_class)}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from celery import Celery
import pickle
import numpy as np
import emoji
import os
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from src.utils.text_preprocessing import cleaning_tweets

load_dotenv()
app = FastAPI()

celery = Celery(
    'tasks',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
)

# CORS setup remains the same
origins = [
    os.getenv("FRONTEND_CORS"),
    os.getenv("BACKEND_CORS"),
    os.getenv("LOCALHOST_CORS"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
clf = pickle.load(open(os.path.join(os.path.dirname(__file__), "../../model.pkl"), "rb"))
vectorizer = SentenceTransformer("all-MiniLM-L6-v2")

class TweetInput(BaseModel):
    tweet: str

def emoji_to_text(emoji_string):
    return ' '.join(emoji.demojize(e) for e in emoji_string) if emoji_string else ''

@celery.task
def process_tweet_task(tweet: str):
    try:
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
        raise Exception(f"Error processing tweet: {str(e)}")

@app.post("/predict/")
async def predict_sentiment(tweet_input: TweetInput, background_tasks: BackgroundTasks):
    try:
        task = process_tweet_task.delay(tweet_input.tweet)
        return {"task_id": str(task.id), "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    try:
        task = process_tweet_task.AsyncResult(task_id)
        if task.failed():
            raise HTTPException(status_code=500, detail=str(task.result))
        return {
            "ready": task.ready(),
            "result": task.result if task.ready() else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))