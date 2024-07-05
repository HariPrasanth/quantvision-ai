# QuantVision.ai

QuantVision.ai is an AI-powered investment strategy application that uses advanced machine learning models and sentiment analysis to provide investment recommendations for stocks listed on NSE, BSE, and US exchanges.

## Features

- **Historical Data Analysis**: Fetches historical stock data for detailed analysis.
- **News Sentiment Analysis**: Uses GPT-4 to analyze the sentiment of news articles related to the stock.
- **Model Training**: Trains or updates an XGBoost model based on historical data to predict future stock returns.
- **Investment Decisions**: Provides investment decisions such as BUY, HOLD, SELL PART, and SELL ALL based on model predictions and sentiment analysis.
- **Multi-Exchange Support**: Supports stocks listed on NSE, BSE, and US exchanges.
- **User-Friendly Interface**: Provides a user-friendly interface using Streamlit with options to add, restart, and delete stocks.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/quantvision-ai.git
    cd quantvision-ai
    ```

2. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**
    Create a `.env` file in the root directory with the following content:
    ```
    NEWS_API_KEY=your_news_api_key
    OPENAI_API_KEY=your_openai_api_key
    ```

4. **Run the application:**
    ```bash
    streamlit run app.py
    ```

## Usage

### Adding a Stock Symbol

1. **Enter Stock Symbol**: In the sidebar, enter the stock symbol (e.g., `RELIANCE`, `TCS`, `AAPL`).
2. **Select Exchange**: Choose the appropriate exchange (NSE, BSE, or US).
3. **Add Stock Symbol**: Click on the "Add Stock Symbol" button to add the stock to the list.

### Managing Stock Symbols

1. **Restart Process**: Click on the "Restart" button next to the stock symbol to restart the analysis process for that stock.
2. **Delete Stock**: Click on the "del" button next to the stock symbol to remove it from the list.

### Running Analysis

- **Run Analysis for All Stocks**: Click on the "Run Analysis for All Stocks" button to start the analysis process for all stored stocks. The progress of each analysis will be displayed.

### Viewing Results

- **Analysis Results**: The results for each stock will be displayed in the main area, including the investment decision and portfolio information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

