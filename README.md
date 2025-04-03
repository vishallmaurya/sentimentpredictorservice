# 🤖 Sentiment Analysis - Machine Learning Model

## 📌 Overview
This is the **Machine Learning (ML) service** of the Sentiment Analysis Web Application. Built using **FastAPI and Scikit-Learn**, it provides sentiment predictions based on text input and supports **incremental learning** to enhance model performance over time. The ML service is hosted on **AWS** for scalability and speed.

## 🚀 Features
- **Text Sentiment Prediction** (Positive, Negative, Neutral)
- **Incremental Learning** for continuous model updates
- **FastAPI-powered API** for seamless integration
- **MongoDB Storage** for user-labeled data
- **Hosted on AWS** for performance and reliability

## 🛠 Tech Stack
- **Python** - Core programming language
- **FastAPI** - Web framework for ML service
- **Scikit-Learn** - Machine learning library
- **AWS Hosting** - Ensuring smooth model inference

## 🏗 Setup & Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/vishallmaurya/sentimentpredictorservice
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv .
source /bin/activate 
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 5️⃣ Run the ML Service
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🚀 Deployment
- **Hosted on AWS** for scalability and performance.
- Ensure environment variables are properly configured in `.env`.

## 📌 Future Enhancements
- Improve model performance with more labeled data.
- Add real-time monitoring and logging.

🤖 **A powerful ML backend for sentiment analysis using FastAPI, SVM, and incremental learning!**