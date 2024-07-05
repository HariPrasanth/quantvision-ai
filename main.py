import streamlit as st
import yfinance as yf
import requests
from transformers import pipeline
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv
import os
import joblib

# Load environment variables
load_dotenv()

NEWS_API_KEY = os.getenv('NEWS_API_KEY')

MODEL_DIR = 'models'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


# Function to get historical data
def get_historical_data(stock_symbol):
    stock_symbol = stock_symbol + ".NS"  # For Indian stocks listed on NSE
    data = yf.download(stock_symbol, start="2020-01-01", end="2024-01-01")
    return data


# Function to get news query by appending the word "stock"
def get_news_query(stock_symbol):
    query = f"stock {stock_symbol}"
    return query


# Function to get news data
def get_news(api_key, query):
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        if 'articles' in news_data:
            return news_data['articles']
    return []


# Function to analyze sentiment using transformers
def analyze_sentiment_transformers(articles):
    sentiment_analysis = pipeline('sentiment-analysis')
    sentiments = [{'title': article['title'], 'sentiment': sentiment_analysis(article['title'])[0]} for article in
                  articles]
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
def make_investment_decision(sentiments, predictions, stock_symbol):
    positive_count = sum(1 for sentiment in sentiments if sentiment['sentiment']['label'] == 'positive')
    negative_count = sum(1 for sentiment in sentiments if sentiment['sentiment']['label'] == 'negative')

    if predictions[-1] > 0.05 and positive_count > negative_count:
        if stock_symbol in st.session_state.portfolio and st.session_state.portfolio[stock_symbol] > 0:
            return "HOLD"  # Already bought, so hold
        else:
            return "BUY"  # Buy if not bought yet
    elif predictions[-1] < -0.05 and negative_count > positive_count:
        if stock_symbol in st.session_state.portfolio and st.session_state.portfolio[stock_symbol] > 10:
            return "SELL PART"  # Sell a few quantities
        elif stock_symbol in st.session_state.portfolio and st.session_state.portfolio[stock_symbol] > 0:
            return "SELL ALL"  # Sell all quantities
        else:
            return "HOLD"  # Nothing to sell, so hold
    else:
        return "HOLD"


# Function to update the model and make predictions
def update_model_and_predict(stock_symbol):
    progress_text = st.empty()
    progress_bar = st.progress(0)

    progress_text.text(f"Fetching historical data for {stock_symbol}...")
    data = get_historical_data(stock_symbol)
    progress_bar.progress(10)

    progress_text.text(f"Generating news query for {stock_symbol}...")
    news_query = get_news_query(stock_symbol)
    progress_bar.progress(20)

    progress_text.text(f"Fetching news for {stock_symbol}...")
    articles = get_news(NEWS_API_KEY, news_query)
    progress_bar.progress(30)

    if articles:
        progress_text.text(f"Analyzing sentiment of news articles for {stock_symbol}...")
        sentiments = analyze_sentiment_transformers(articles)
        progress_bar.progress(50)

        # Displaying number of news articles and their sentiments
        sentiment_df = pd.DataFrame(sentiments)
        progress_text.text(f"Sentiment results for {stock_symbol}:")
        st.dataframe(sentiment_df)
    else:
        progress_text.text(f"No news articles received to analyze for {stock_symbol}.")
        sentiments = []
        progress_bar.progress(50)

    progress_text.text(f"Training or updating model for {stock_symbol}...")
    model, rmse = train_xgboost_model(data, stock_symbol, update=True)
    progress_text.text(f"Model trained or updated for {stock_symbol}. RMSE: {rmse}")
    progress_bar.progress(70)

    progress_text.text(f"Making investment decision for {stock_symbol}...")
    data['Prediction'] = model.predict(data[['Open', 'High', 'Low', 'Volume']])
    decision = make_investment_decision(sentiments, data['Prediction'], stock_symbol)
    progress_text.text(f"Investment Decision for {stock_symbol}: {decision}")
    progress_bar.progress(100)

    return decision


# Initialize session state for storing used stock symbols and portfolio
if 'used_stock_symbols' not in st.session_state:
    st.session_state.used_stock_symbols = {}
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

# Streamlit UI
st.set_page_config(page_title="QuantVision.ai", page_icon="📈", layout="wide")
st.title("QuantVision.ai - AI-Powered Investment Strategy")
st.markdown("""
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Input for stock symbol
stock_symbol = st.sidebar.text_input("Enter Stock Symbol:")

# Button to add stock symbol to the list
if st.sidebar.button("Add Stock Symbol"):
    if stock_symbol and stock_symbol not in st.session_state.used_stock_symbols:
        st.session_state.used_stock_symbols[stock_symbol] = {'status': 'new'}
        st.sidebar.success(f"Added {stock_symbol} to the list.")
    else:
        st.sidebar.error(f"{stock_symbol} is already in the list or invalid input.")

# Display stored stock symbols and provide an option to restart the process
st.sidebar.header("Stored Stock Symbols")
for symbol in list(st.session_state.used_stock_symbols.keys()):
    if st.sidebar.button(f"Restart Process for {symbol}"):
        st.session_state.used_stock_symbols[symbol]['status'] = 'new'


# Function to run analysis for all stocks sequentially with progress tracking
def run_analysis_for_all_stocks():
    total_stocks = len(st.session_state.used_stock_symbols)
    progress_bar = st.progress(0)
    progress_increment = 1 / total_stocks

    for idx, symbol in enumerate(st.session_state.used_stock_symbols.keys()):
        try:
            decision = update_model_and_predict(symbol)
            st.session_state.used_stock_symbols[symbol]['status'] = 'completed'
            st.session_state.used_stock_symbols[symbol]['decision'] = decision
            if decision == "BUY":
                st.session_state.portfolio[symbol] = st.session_state.portfolio.get(symbol,
                                                                                    0) + 10  # Simulate buying 10 shares
            elif decision == "SELL PART":
                st.session_state.portfolio[symbol] -= 5  # Simulate selling 5 shares
            elif decision == "SELL ALL":
                st.session_state.portfolio[symbol] = 0  # Simulate selling all shares
        except Exception as e:
            st.session_state.used_stock_symbols[symbol]['status'] = 'failed'
            st.session_state.used_stock_symbols[symbol]['error'] = str(e)
            st.error(f"Error processing {symbol}: {e}")
        progress_bar.progress(min((idx + 1) * progress_increment, 1.0))


# Button to run analysis for all stocks
if st.sidebar.button("Run Analysis for All Stocks"):
    run_analysis_for_all_stocks()

# Display results in the main area
st.header("Analysis Results")
for symbol, details in st.session_state.used_stock_symbols.items():
    with st.expander(f"Results for {symbol}"):
        st.write(f"Symbol: {symbol}")
        if details['status'] == 'completed':
            st.write(f"Decision: {details['decision']}")
        elif details['status'] == 'failed':
            st.write(f"Error: {details['error']}")
        else:
            st.write("Status: Pending")

        # Display portfolio information
        if symbol in st.session_state.portfolio:
            st.write(f"Shares owned: {st.session_state.portfolio[symbol]}")
        else:
            st.write(f"Shares not owned: {st.session_state.portfolio[symbol]}")