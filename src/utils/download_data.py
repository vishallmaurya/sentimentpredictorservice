import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
import os

load_dotenv()

uri = os.getenv("MONGODB_URI")
dbname = os.getenv("DB_NAME") 
client = MongoClient(uri+"/"+dbname)
db = client[dbname]
collection = db["datas"]

data = list(collection.find({}))
df = pd.DataFrame(data)
df.drop(columns=['_id', 'user_id', '__v'], inplace=True)
df.to_csv('../output.csv', index=False, encoding='utf-8')