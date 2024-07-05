import streamlit as st
import yfinance as yf
import requests
from transformers import pipeline
from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Function to get historical data
def get_historical_data(stock_symbol):
    data = yf.download(stock_symbol, start="2020-01-01", end="2024-01-01")
    return data


# Function to get news data
def get_news(api_key, query):
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}'
    response = requests.get(url)
    news_data = response.json()
    return news_data


# Function to analyze sentiment using transformers
def analyze_sentiment_transformers(news_data):
    sentiment_analysis = pipeline('sentiment-analysis')
    sentiments = [sentiment_analysis(article['title'])[0] for article in news_data['articles']]
    return sentiments


# Function to analyze sentiment using VADER
def analyze_sentiment_vader(news_data):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(article['title']) for article in news_data['articles']]
    return sentiments


# Function to train an XGBoost model
def train_xgboost_model(data):
    data['Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Return']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return model, rmse


# Function to make investment decision
def make_investment_decision(sentiments, predictions, threshold=0.5):
    positive_count = sum(1 for sentiment in sentiments if sentiment['compound'] > 0.05)
    negative_count = sum(1 for sentiment in sentiments if sentiment['compound'] < -0.05)

    if predictions[-1] > threshold and positive_count > negative_count:
        return "BUY"
    elif predictions[-1] < -threshold and negative_count > positive_count:
        return "SELL"
    else:
        return "HOLD"


# Function to place order using Zerodha Kite API
def place_order(api_key, api_secret, access_token, stock_symbol, action):
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    try:
        if action == "BUY":
            kite.place_order(tradingsymbol=stock_symbol, exchange='NSE', transaction_type='BUY', quantity=1,
                             order_type='MARKET', product='CNC')
        elif action == "SELL":
            kite.place_order(tradingsymbol=stock_symbol, exchange='NSE', transaction_type='SELL', quantity=1,
                             order_type='MARKET', product='CNC')
        else:
            st.write("HOLD - No action taken")
    except Exception as e:
        st.error(f"Error placing order: {e}")


# Streamlit UI
st.title("QuantVision.ai - AI-Powered Investment Strategy")

stock_symbol = st.text_input("Enter Stock Symbol:")
if st.button("Get Data"):
    data = get_historical_data(stock_symbol)
    st.write(data)

api_key = st.text_input("Enter News API Key:")
if st.button("Get News"):
    news_data = get_news(api_key, stock_symbol)
    st.write(news_data)

if st.button("Analyze Sentiment"):
    sentiment_method = st.selectbox("Choose Sentiment Analysis Method", ["Transformers", "VADER"])
    if sentiment_method == "Transformers":
        sentiments = analyze_sentiment_transformers(news_data)
    else:
        sentiments = analyze_sentiment_vader(news_data)
    st.write(sentiments)

if st.button("Make Decision"):
    data = get_historical_data(stock_symbol)
    model, rmse = train_xgboost_model(data)
    st.write(f"Model RMSE: {rmse}")
    data['Prediction'] = model.predict(data[['Open', 'High', 'Low', 'Volume']])
    decision = make_investment_decision(sentiments, data['Prediction'])
    st.write(f"Decision: {decision}")

    if decision in ["BUY", "SELL", "HOLD"]:
        zerodha_api_key = st.text_input("Enter Zerodha API Key:")
        zerodha_api_secret = st.text_input("Enter Zerodha API Secret:")
        zerodha_access_token = st.text_input("Enter Zerodha Access Token:")
        place_order(zerodha_api_key, zerodha_api_secret, zerodha_access_token, stock_symbol, decision)
