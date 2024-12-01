# Imports

import pandas as pd
import yfinance as yf


# Create function to fetch stock name and id
def fetch_stocks():
    
    df = pd.read_csv("D:/SAMLFRFM/data/stockss.csv")
    df = df[["CODE", "COMPANY"]]
    stock_dict = dict(zip(df["CODE"], df["COMPANY"]))
    
    return stock_dict

# Create function to fetch periods and intervals
def fetch_periods_intervals():
    
    periods = {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"],
    }
    
    return periods

#function to fetch models
def fetch_models():
    
    models = {
        "Random Forest": 1,
        "Support Vector Machine": 2,
        "Decision Tree": 3,
        "KNN": 4,
        "ARIMA": 5,
        "LSTM": 6,
        "GRU": 7,
        "XGBoost": 8,
    }
    
    return models


# Function to fetch the stock info
def fetch_stock_info(stock_ticker):
    
    stock_data = yf.Ticker(stock_ticker)
    stock_data_info = stock_data.info

    # Function to safely get value from dictionary or return "N/A"
    def safe_get(data_dict, key):
        return data_dict.get(key, "N/A")

    # Extract only the important information
    stock_data_info = {
        "Basic Information": {
            "symbol": safe_get(stock_data_info, "symbol"),
            "longName": safe_get(stock_data_info, "longName"),
            "currency": safe_get(stock_data_info, "currency"),
            "exchange": safe_get(stock_data_info, "exchange"),
        },
        "Market Data": {
            "currentPrice": safe_get(stock_data_info, "currentPrice"),
            "previousClose": safe_get(stock_data_info, "previousClose"),
            "open": safe_get(stock_data_info, "open"),
            "dayLow": safe_get(stock_data_info, "dayLow"),
            "dayHigh": safe_get(stock_data_info, "dayHigh"),
            "regularMarketPreviousClose": safe_get(
                stock_data_info, "regularMarketPreviousClose"
            ),
            "regularMarketOpen": safe_get(stock_data_info, "regularMarketOpen"),
            "regularMarketDayLow": safe_get(stock_data_info, "regularMarketDayLow"),
            "regularMarketDayHigh": safe_get(stock_data_info, "regularMarketDayHigh"),
            "fiftyTwoWeekLow": safe_get(stock_data_info, "fiftyTwoWeekLow"),
            "fiftyTwoWeekHigh": safe_get(stock_data_info, "fiftyTwoWeekHigh"),
            "fiftyDayAverage": safe_get(stock_data_info, "fiftyDayAverage"),
            "twoHundredDayAverage": safe_get(stock_data_info, "twoHundredDayAverage"),
        }
    }

    # Return the stock data
    return stock_data_info


# Function to fetch the stock history
def fetch_stock_history(stock_ticker, period, interval):
    
    stock_data = yf.Ticker(stock_ticker)
    stock_data_history = stock_data.history(period=period, interval=interval)[
        ["Open", "High", "Low", "Close"]
    ]

    return stock_data_history