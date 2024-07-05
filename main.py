import streamlit as st
import yfinance as yf
import requests
from transformers import pipeline
from kiteconnect import KiteConnect
import numpy as np


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


# Function to analyze sentiment
sentiment_analysis = pipeline('sentiment-analysis')


def analyze_sentiment(news_data):
    sentiments = [sentiment_analysis(article['title'])[0] for article in news_data['articles']]
    return sentiments


# Simple moving average based decision model
def simple_moving_average(data, window_size):
    return data.rolling(window=window_size).mean()


# Function to make investment decision
def make_investment_decision(sentiments, sma_short, sma_long):
    positive_count = sum(1 for sentiment in sentiments if sentiment['label'] == 'positive')
    negative_count = sum(1 for sentiment in sentiments if sentiment['label'] == 'negative')

    if sma_short.iloc[-1] > sma_long.iloc[-1] and positive_count > negative_count:
        return "BUY"
    elif sma_short.iloc[-1] < sma_long.iloc[-1] and negative_count > positive_count:
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
st.title("AI-Powered Investment Strategy")

stock_symbol = st.text_input("Enter Stock Symbol:")
if st.button("Get Data"):
    data = get_historical_data(stock_symbol)
    st.write(data)

api_key = st.text_input("Enter News API Key:")
if st.button("Get News"):
    news_data = get_news(api_key, stock_symbol)
    st.write(news_data)

if st.button("Analyze Sentiment"):
    sentiments = analyze_sentiment(news_data)
    st.write(sentiments)

if st.button("Make Decision"):
    data = get_historical_data(stock_symbol)
    sma_short = simple_moving_average(data['Close'], window_size=50)
    sma_long = simple_moving_average(data['Close'], window_size=200)
    decision = make_investment_decision(sentiments, sma_short, sma_long)
    st.write(f"Decision: {decision}")

    if decision in ["BUY", "SELL", "HOLD"]:
        zerodha_api_key = st.text_input("Enter Zerodha API Key:")
        zerodha_api_secret = st.text_input("Enter Zerodha API Secret:")
        zerodha_access_token = st.text_input("Enter Zerodha Access Token:")
        place_order(zerodha_api_key, zerodha_api_secret, zerodha_access_token, stock_symbol, decision)
