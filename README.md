# Disaster-Tweet (Kaggle)

This code is for the Kaggle competition - https://www.kaggle.com/competitions/nlp-getting-started/overview

# Disaster Tweet Prediction using Doc2Vec & XGBoost

This project demonstrates a simple machine learning web app built using Flask that predicts whether a tweet is about a disaster or not. It uses a Doc2Vec model for text embedding and an XGBoost classifier for prediction.

---

## 🚀 Features

- Trains a `Doc2Vec` model to vectorize tweet texts
- Classifies tweets using `XGBoost`
- Web interface built with `Flask`
- Accepts user input and displays binary prediction (`0` or `1`)

---

## 🧠 Model Pipeline

1. Preprocess tweets using NLTK tokenization
2. Train a `Doc2Vec` model from Gensim
3. Infer vector for each tweet
4. Train an `XGBoost` classifier
5. Save both models (`model.pkl` for Doc2Vec, `xgb_best.pkl` for classifier)

---

## 🌐 Flask Implementation

- The `app.py` script sets up the Flask server.
- It loads the trained models (`model.pkl`, `xgb_best.pkl`) at runtime.
- The `/` route renders an HTML form (`index.html`) where users can enter a tweet.
- When a tweet is submitted, the `/predict` route:
  - Tokenizes and vectorizes the input
  - Uses the XGBoost model to make a prediction
  - Returns the result to the HTML template

This allows for real-time, interactive predictions directly in the browser.
