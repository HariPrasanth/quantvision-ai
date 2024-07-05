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
import concurrent.futures

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
    positive_count = sum(1 for sentiment in sentiments if sentiment['sentiment']['label'] == 'POSITIVE')
    negative_count = sum(1 for sentiment in sentiments if sentiment['sentiment']['label'] == 'NEGATIVE')

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
    st.write(f"Fetching historical data for {stock_symbol}...")
    data = get_historical_data(stock_symbol)
    st.write("Historical data fetched.")

    st.write("Generating news query...")
    news_query = get_news_query(stock_symbol)
    st.write(f"News query generated: {news_query}")

    st.write("Fetching news...")
    articles = get_news(NEWS_API_KEY, news_query)
    st.write(f"Number of news articles received: {len(articles)}")

    if articles:
        st.write("Analyzing sentiment of news articles...")
        sentiments = analyze_sentiment_transformers(articles)
        st.write("Sentiment analysis completed.")

        # Displaying number of news articles and their sentiments
        sentiment_df = pd.DataFrame(sentiments)
        st.write("Sentiment results:")
        st.dataframe(sentiment_df)
    else:
        st.write("No news articles received to analyze.")
        sentiments = []

    st.write("Training or updating model...")
    model, rmse = train_xgboost_model(data, stock_symbol, update=True)
    st.write(f"Model trained or updated. RMSE: {rmse}")

    st.write("Making investment decision...")
    data['Prediction'] = model.predict(data[['Open', 'High', 'Low', 'Volume']])
    decision = make_investment_decision(sentiments, data['Prediction'], stock_symbol)
    st.write(f"Investment Decision: {decision}")

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
for symbol in st.session_state.used_stock_symbols.keys():
    if st.sidebar.button(f"Restart Process for {symbol}"):
        st.session_state.used_stock_symbols[symbol]['status'] = 'new'


# Function to run analysis for all stocks in parallel
def run_analysis_for_all_stocks():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(update_model_and_predict, symbol): symbol for symbol in
                   st.session_state.used_stock_symbols.keys()}
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                decision = future.result()
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
            st.write("No shares owned")
