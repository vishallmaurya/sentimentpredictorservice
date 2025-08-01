# !pip install emoji

import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
import emoji
import os
from src.utils.text_preprocessing import cleaning_tweets

data = pd.read_csv('/content/sentiment_data_2_label_annotated.csv')
data1 = pd.read_csv('/content/sentiment_data_3csv_annotated.csv')
data2 = pd.read_csv('/content/sentiment_data_label_annotated.csv') 

data = pd.concat([data, data1, data2])

data.drop(['Unnamed: 0', 'Date', 'class'], inplace=True, axis=1)

cleaned_tweets = []
emoji_list = []

for i in range(data.shape[0]):
  clean_text, extracted_emojis = cleaning_tweets(str(data.iloc[i]["Tweet"]))
  cleaned_tweets.append(clean_text)
  emoji_list.append(extracted_emojis)

data["clean_tweet"] = cleaned_tweets
data["extracted_emojis"] = emoji_list

model = SentenceTransformer("all-MiniLM-L6-v2")

def emoji_to_text(emoji_string):
    return ' '.join(emoji.demojize(e) for e in emoji_string) if emoji_string else ''

data["emoji_text"] = data["extracted_emojis"].apply(emoji_to_text)

text_embeddings = model.encode(data["clean_tweet"].tolist(), convert_to_numpy=True)
data["text_embeddings"] = list(text_embeddings)

embedding_dim = model.get_sentence_embedding_dimension()

emoji_embeddings = [
    model.encode(emoji_text, convert_to_numpy=True) if emoji_text.strip() else np.zeros(embedding_dim)
    for emoji_text in data["emoji_text"]
]

data["emoji_embeddings"] = list(emoji_embeddings)

from imblearn.over_sampling import SMOTE
smote_balance = SMOTE()

X_text = np.array(data["text_embeddings"].tolist())
X_emoji = np.array(data["emoji_embeddings"].tolist())


X_final = np.hstack((X_text, X_emoji))


y = data["class"].values

smote = SMOTE()
X_SMOTE, Y_SMOTE = smote.fit_resample(X_final, y)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_SMOTE, Y_SMOTE, test_size=0.2, random_state=42)

incremental_clf = SGDClassifier(loss="perceptron", warm_start=True, class_weight="balanced")
incremental_clf.fit(X_train1, y_train1)

y_pred1 = incremental_clf.predict(X_test1)
print(f"SGD Accuracy: {accuracy_score(y_test1, y_pred1): .4f}" )


model_path = os.path.join(os.path.dirname(__file__), "../model.pkl")
pickle.dump(incremental_clf, open(model_path, "wb"))

