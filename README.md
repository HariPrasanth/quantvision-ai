# QuantVision.ai

QuantVision.ai leverages AI to automate investment strategies by integrating market sentiment analysis from news APIs and historical data analysis. It places buy or sell orders using the Zerodha Kite API.

## Features

- Real-time stock market data retrieval
- News sentiment analysis
- Historical data analysis using simple moving averages
- Automated investment decisions (Buy/Sell/Hold)
- Order placement using Zerodha Kite API
- Streamlit-based user interface

## Setup Instructions

### Prerequisites

- Python 3.7 or above
- API keys for NewsAPI and Zerodha Kite

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/quantvision-ai.git
    cd quantvision-ai
    ```

2. Create and activate a virtual environment (recommended):
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Usage

1. Enter the stock symbol and fetch historical data.
2. Enter your NewsAPI key and fetch news data.
3. Analyze the sentiment of the news articles.
4. Make investment decisions based on the sentiment analysis and historical data predictions.
5. Place buy/sell/hold orders using the Zerodha Kite API.

## License

This project is licensed under the MIT License.
