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
from dotenv import load_dotenv
import os
import openai
import joblib  # for saving and loading models

# Load environment variables
load_dotenv()

NEWS_API_KEY = os.getenv('NEWS_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

MODEL_PATH = 'xgboost_model.pkl'


# Function to get historical data
def get_historical_data(stock_symbol):
    stock_symbol = stock_symbol + ".NS"  # For Indian stocks listed on NSE
    data = yf.download(stock_symbol, start="2020-01-01", end="2024-01-01")
    return data


# Function to get news query using OpenAI GPT-4
def get_news_query(stock_symbol):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Generate a news query for stock symbol: {stock_symbol}"}
        ]
    )
    query = response['choices'][0]['message']['content'].strip()
    return query


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


# Function to train or update an XGBoost model
def train_xgboost_model(data, update=False):
    data['Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Return']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if update and os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        model.fit(X_train, y_train, xgb_model=model.get_booster())
    else:
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05)
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    joblib.dump(model, MODEL_PATH)  # Save the trained model
    return model, rmse


# Function to make investment decision
def make_investment_decision(sentiments, predictions, threshold=0.5):
    positive_count = sum(1 for sentiment in sentiments if sentiment['label'] == 'positive')
    negative_count = sum(1 for sentiment in sentiments if sentiment['label'] == 'negative')

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
if st.button("Analyze and Make Decision"):
    st.write("Fetching historical data...")
    data = get_historical_data(stock_symbol)
    st.write("Historical data fetched.")

    st.write("Generating news query using OpenAI GPT-4...")
    news_query = get_news_query(stock_symbol)
    st.write(f"News query generated: {news_query}")

    st.write("Fetching news and analyzing sentiment...")
    news_data = get_news(NEWS_API_KEY, news_query)
    sentiments = analyze_sentiment_transformers(news_data)
    st.write("Sentiment analysis completed.")

    st.write("Training or updating model...")
    model, rmse = train_xgboost_model(data, update=True)
    st.write(f"Model trained or updated. RMSE: {rmse}")

    st.write("Making investment decision...")
    data['Prediction'] = model.predict(data[['Open', 'High', 'Low', 'Volume']])
    decision = make_investment_decision(sentiments, data['Prediction'])
    st.write(f"Investment Decision: {decision}")

    if decision in ["BUY", "SELL", "HOLD"]:
        zerodha_api_key = st.text_input("Enter Zerodha API Key:")
        zerodha_api_secret = st.text_input("Enter Zerodha API Secret:")
        zerodha_access_token = st.text_input("Enter Zerodha Access Token:")
        place_order(zerodha_api_key, zerodha_api_secret, zerodha_access_token, stock_symbol, decision)
