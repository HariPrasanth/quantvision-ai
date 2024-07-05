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
import joblib
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import pytz

# Load environment variables
load_dotenv()

NEWS_API_KEY = os.getenv('NEWS_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY

MODEL_DIR = 'models'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


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
    query = response.choices[0]['message']['content'].strip()
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
    sentiments = [{'title': article['title'], 'sentiment': sentiment_analysis(article['title'])[0]} for article in
                  news_data['articles']]
    return sentiments


# Function to train or update an XGBoost model
def train_xgboost_model(data, stock_symbol, update=False):
    model_path = os.path.join(MODEL_DIR, f'{stock_symbol}.pkl')
    data['Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Return']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if update and os.path.exists(model_path):
        model = joblib.load(model_path)
        model.fit(X_train, y_train, xgb_model=model.get_booster())
    else:
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05)
        model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    joblib.dump(model, model_path)  # Save the trained model
    return model, rmse


# Function to make investment decision
def make_investment_decision(sentiments, predictions, threshold=0.5):
    positive_count = sum(1 for sentiment in sentiments if sentiment['sentiment']['label'] == 'positive')
    negative_count = sum(1 for sentiment in sentiments if sentiment['sentiment']['label'] == 'negative')

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


# Function to update the model and make predictions
def update_model_and_predict(stock_symbol):
    st.write(f"Fetching historical data for {stock_symbol}...")
    data = get_historical_data(stock_symbol)
    st.write("Historical data fetched.")
    st.write(data)  # Display historical data

    st.write("Generating news query using OpenAI GPT-4...")
    news_query = get_news_query(stock_symbol)
    st.write(f"News query generated: {news_query}")

    st.write("Fetching news and analyzing sentiment...")
    news_data = get_news(NEWS_API_KEY, news_query)
    sentiments = analyze_sentiment_transformers(news_data)
    st.write("Sentiment analysis completed.")

    # Displaying number of news articles and their sentiments
    st.write(f"Number of news articles received: {len(sentiments)}")
    sentiment_df = pd.DataFrame(sentiments)
    st.write("Sentiment results:")
    st.write(sentiment_df)

    st.write("Training or updating model...")
    model, rmse = train_xgboost_model(data, stock_symbol, update=True)
    st.write(f"Model trained or updated. RMSE: {rmse}")

    st.write("Making investment decision...")
    data['Prediction'] = model.predict(data[['Open', 'High', 'Low', 'Volume']])
    decision = make_investment_decision(sentiments, data['Prediction'])
    st.write(f"Investment Decision: {decision}")

    return decision


# Streamlit UI
st.title("QuantVision.ai - AI-Powered Investment Strategy")

# Add a sidebar to enter the stock symbol
stock_symbol = st.sidebar.text_input("Enter Stock Symbol:", value="RELIANCE")

# Display the initial decision when the app loads
if stock_symbol:
    st.write(f"Fetching initial decision for {stock_symbol}...")
    initial_decision = update_model_and_predict(stock_symbol)
    st.write(f"Initial Investment Decision for {stock_symbol}: {initial_decision}")

# Button to manually update and make decision
if st.button("Update and Make Decision"):
    decision = update_model_and_predict(stock_symbol)
    st.write(f"Investment Decision: {decision}")


# Schedule the update_model_and_predict function to run at 6 AM IST every day
def schedule_daily_update():
    scheduler = BackgroundScheduler()
    ist = pytz.timezone('Asia/Kolkata')
    scheduled_time = datetime.now(ist).replace(hour=6, minute=0, second=0, microsecond=0)
    if datetime.now(ist) > scheduled_time:
        scheduled_time += timedelta(days=1)
    scheduler.add_job(update_model_and_predict, 'interval', days=1, start_date=scheduled_time, args=[stock_symbol])
    scheduler.start()


# Run the scheduling function
schedule_daily_update()
