# !pip install emoji

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sentence_transformers import SentenceTransformer
import emoji
import os
import numpy as np
from sklearn.metrics import accuracy_score
from src.utils.text_preprocessing import cleaning_tweets
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight

model_path = os.path.join(os.path.dirname(__file__), "../model.pkl")
incremental_clf = pickle.load(open(model_path, "rb"))
model = SentenceTransformer("all-MiniLM-L6-v2")

data = pd.read_csv("/content/synthetic_sentiment_data.csv")

data.head(1)

data.drop(['Unnamed: 0', 'Date', 'class'], inplace=True, axis=1)

cleaned_tweets = []
emoji_list = []

for i in range(data.shape[0]):
  clean_text, extracted_emojis = cleaning_tweets(str(data.iloc[i]["Tweet"]))
  cleaned_tweets.append(clean_text)
  emoji_list.append(extracted_emojis)

data["clean_tweet"] = cleaned_tweets
data["extracted_emojis"] = emoji_list

def emoji_to_text(emoji_string):
    return ' '.join(emoji.demojize(e) for e in emoji_string) if emoji_string else ''

data["emoji_text"] = data["extracted_emojis"].apply(emoji_to_text)

text_embeddings = model.encode(data["clean_tweet"].tolist(), convert_to_numpy=True)
embedding_dim = model.get_sentence_embedding_dimension()
emoji_embeddings = [
    model.encode(emoji_text, convert_to_numpy=True) if emoji_text.strip() else np.zeros(embedding_dim)
    for emoji_text in data["emoji_text"]
]

X_new = np.hstack((text_embeddings, np.array(emoji_embeddings)))

y_new = data["gpt_label"].values

class_weights = compute_class_weight("balanced", classes=np.unique(y_new), y=y_new)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

class_weight_dict

incremental_clf.class_weight = class_weight_dict
incremental_clf.partial_fit(X_new, y_new, classes=np.array([0, 1, 2]))

y_pred_new = incremental_clf.predict(X_new)
accuracy = accuracy_score(y_new, y_pred_new)
print(f"Incremental Learning Accuracy: {accuracy:.4f}")

pickle.dump(incremental_clf, open("model.pkl", "wb"))

