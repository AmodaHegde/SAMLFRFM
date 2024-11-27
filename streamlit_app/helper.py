# Imports
import datetime as dt
import os
from pathlib import Path
import math
from datetime import datetime, timedelta

# Import pandas
import pandas as pd

# Import yfinance
from sklearn.tree import DecisionTreeRegressor
import yfinance as yf

# Import the required libraries
from statsmodels.tsa.ar_model import AutoReg
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,GRU, Dense, Dropout

from randomforest import get_randomforest

# Create function to fetch stock name and id
def fetch_stocks():
    # Load the data
    df = pd.read_csv("D:/Stockastic/data/equity_issuers.csv")

    # Filter the data
    df = df[["Security Code", "Issuer Name"]]

    # Create a dictionary
    stock_dict = dict(zip(df["Security Code"], df["Issuer Name"]))

    # Return the dictionary
    return stock_dict


# Create function to fetch periods and intervals
def fetch_periods_intervals():
    # Create dictionary for periods and intervals
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

    # Return the dictionary
    return periods

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
        "Regression": 9,
    }
    
    return models


# Function to fetch the stock info
def fetch_stock_info(stock_ticker):
    # Pull the data for the first security
    stock_data = yf.Ticker(stock_ticker)

    # Extract full of the stock
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
        },
        "Volume and Shares": {
            "volume": safe_get(stock_data_info, "volume"),
            "regularMarketVolume": safe_get(stock_data_info, "regularMarketVolume"),
            "averageVolume": safe_get(stock_data_info, "averageVolume"),
            "averageVolume10days": safe_get(stock_data_info, "averageVolume10days"),
            "averageDailyVolume10Day": safe_get(
                stock_data_info, "averageDailyVolume10Day"
            ),
            "sharesOutstanding": safe_get(stock_data_info, "sharesOutstanding"),
            "impliedSharesOutstanding": safe_get(
                stock_data_info, "impliedSharesOutstanding"
            ),
            "floatShares": safe_get(stock_data_info, "floatShares"),
        },
        "Dividends and Yield": {
            "dividendRate": safe_get(stock_data_info, "dividendRate"),
            "dividendYield": safe_get(stock_data_info, "dividendYield"),
            "payoutRatio": safe_get(stock_data_info, "payoutRatio"),
        },
        "Valuation and Ratios": {
            "marketCap": safe_get(stock_data_info, "marketCap"),
            "enterpriseValue": safe_get(stock_data_info, "enterpriseValue"),
            "priceToBook": safe_get(stock_data_info, "priceToBook"),
            "debtToEquity": safe_get(stock_data_info, "debtToEquity"),
            "grossMargins": safe_get(stock_data_info, "grossMargins"),
            "profitMargins": safe_get(stock_data_info, "profitMargins"),
        },
        "Financial Performance": {
            "totalRevenue": safe_get(stock_data_info, "totalRevenue"),
            "revenuePerShare": safe_get(stock_data_info, "revenuePerShare"),
            "totalCash": safe_get(stock_data_info, "totalCash"),
            "totalCashPerShare": safe_get(stock_data_info, "totalCashPerShare"),
            "totalDebt": safe_get(stock_data_info, "totalDebt"),
            "earningsGrowth": safe_get(stock_data_info, "earningsGrowth"),
            "revenueGrowth": safe_get(stock_data_info, "revenueGrowth"),
            "returnOnAssets": safe_get(stock_data_info, "returnOnAssets"),
            "returnOnEquity": safe_get(stock_data_info, "returnOnEquity"),
        },
        "Cash Flow": {
            "freeCashflow": safe_get(stock_data_info, "freeCashflow"),
            "operatingCashflow": safe_get(stock_data_info, "operatingCashflow"),
        },
        "Analyst Targets": {
            "targetHighPrice": safe_get(stock_data_info, "targetHighPrice"),
            "targetLowPrice": safe_get(stock_data_info, "targetLowPrice"),
            "targetMeanPrice": safe_get(stock_data_info, "targetMeanPrice"),
            "targetMedianPrice": safe_get(stock_data_info, "targetMedianPrice"),
        },
    }

    # Return the stock data
    return stock_data_info


# Function to fetch the stock history
def fetch_stock_history(stock_ticker, period, interval):
    # Pull the data for the first security
    stock_data = yf.Ticker(stock_ticker)

    # Extract full of the stock
    stock_data_history = stock_data.history(period=period, interval=interval)[
        ["Open", "High", "Low", "Close"]
    ]

    # Return the stock data
    return stock_data_history

def gen_lstm(stock_ticker):
    
    data = yf.Ticker(stock_ticker)
    hist = data.history(period="4y", interval="1d")
    hist = hist[["Close"]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(hist)
    
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    def create_sequences(data, seq_length=60):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    seq_length = 60  # Use 60 previous days to predict the next
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
    
    # Step 5: Predict on test data and visualize
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    actual_prices = scaler.inverse_transform(test_data[seq_length:])
    
    input_date = input("Enter a future date (YYYY-MM-DD): ")
    target_date = datetime.strptime(input_date, '%Y-%m-%d')

    # Calculate days to predict from the last available date in the data
    last_date = hist.index[-1].to_pydatetime().replace(tzinfo=None)  # Convert to timezone-naive datetime
    days_to_predict = (target_date - last_date).days

    if days_to_predict <= 0:
        print("The entered date is not in the future. Please enter a future date.")
    else:
        # Generate predictions up to the target date
        last_sequence = test_data[-seq_length:]
        predicted_future_prices = []

        for _ in range(days_to_predict):
            prediction = model.predict(last_sequence.reshape(1, seq_length, 1))
            predicted_future_prices.append(prediction[0, 0])
            last_sequence = np.append(last_sequence[1:], prediction, axis=0)

        predicted_future_prices = scaler.inverse_transform(np.array(predicted_future_prices).reshape(-1, 1))

        # Print the predicted stock price for the specific future date
        predicted_price_on_target_date = predicted_future_prices[-1][0]
        print(f"Predicted Stock Price for {input_date}: ${predicted_price_on_target_date:.2f}")
        actual_dates = hist.index[-len(actual_prices):]
    return actual_dates, actual_prices, predicted_prices 

def gen_arima(stock_ticker):
    
    data = yf.Ticker(stock_ticker)
    hist = data.history(period="4y", interval="1d")
    hist = hist["Close"]
    train_size = int(0.85*len(hist))
    test_size = len(hist) - train_size
    univariate_df = hist
    univariate_df.columns = ['date', 'close']
    univariate_df = univariate_df.reset_index()
    x_train, y_train = pd.DataFrame(univariate_df.iloc[:train_size, 0]), pd.DataFrame(univariate_df.iloc[:train_size, 1])
    x_valid, y_valid = pd.DataFrame(univariate_df.iloc[train_size:, 0]), pd.DataFrame(univariate_df.iloc[train_size:, 1])

    # Fit model
    model = ARIMA(y_train, order=(4,0,4))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(151)
    y_pred = forecast.predicted_mean
    #conf = forecast.conf_int()
    # score_mae = mean_absolute_error(y_valid, y_pred)
    # print(score_mae)
    # score_rmse = math.sqrt(mean_squared_error(y_valid, y_pred))
    # print('RMSE: {}'.format(score_rmse))
    
    return y_pred
# Function to generate the stock prediction
def generate_stock_prediction1(stock_ticker):
    # Try to generate the predictions
    try:
        # Pull the data for the first security
        stock_data = yf.Ticker(stock_ticker)

        # Extract the data for last 1yr with 1d interval
        stock_data_hist = stock_data.history(period="2y", interval="1d")

        # Clean the data for to keep only the required columns
        stock_data_close = stock_data_hist[["Close"]]

        # Change frequency to day
        stock_data_close = stock_data_close.asfreq("D", method="ffill")

        # Fill missing values
        stock_data_close = stock_data_close.ffill()

        # Define training and testing area
        train_df = stock_data_close.iloc[: int(len(stock_data_close) * 0.9) + 1]  # 90%
        test_df = stock_data_close.iloc[int(len(stock_data_close) * 0.9) :]  # 10%

        # Define Regression training model1
        model1 = AutoReg(train_df["Close"], 250).fit(cov_type="HC0")

        # Predict data for test data
        predictions = model1.predict(
            start=test_df.index[0], end=test_df.index[-1], dynamic=True
        )

        # Predict 90 days into the future
        forecast = model1.predict(
            start=test_df.index[0],
            end=test_df.index[-1] + dt.timedelta(days=90),
            dynamic=True,
        )

        # Return the required data
        return train_df, test_df, forecast, predictions

    # If error occurs
    except:
        # Return None
        return None, None, None, None

def generate_stock_prediction2(stock_ticker):
    
    try:
        get_randomforest(stock_ticker)
    #     # Pull the data for the first security
    #     stock_data = yf.Ticker(stock_ticker)

    #     # Extract the data for last 1yr with 1d interval
    #     stock_data_hist = stock_data.history(period="2y", interval="1d")
    #     # Clean the data for to keep only the required columns
    #     stock_data_close = stock_data_hist[["Close"]]

    #     # Change frequency to day
    #     stock_data_close = stock_data_close.asfreq("D", method="ffill")

    #     # Fill missing values
    #     stock_data_close = stock_data_close.ffill()

    #     # Define training and testing area
    #     train_df = stock_data_close.iloc[: int(len(stock_data_close) * 0.9) + 1]  # 90%
    #     test_df = stock_data_close.iloc[int(len(stock_data_close) * 0.9) :]  # 10%
    #    # Convert dates to ordinal for RandomForest
    #     X_train = np.array([d.toordinal() for d in train_df.index]).reshape(-1, 1)
    #     y_train = train_df["Close"].values
    #     X_test = np.array([d.toordinal() for d in test_df.index]).reshape(-1, 1)
    #     y_test = test_df["Close"].values

    #     # Train model
    #     model = RandomForestRegressor(n_estimators=100, random_state=42)
    #     model.fit(X_train, y_train)

    #     # Predictions on test data
    #     y_pred = model.predict(X_test)

    #     # Forecast for 90 days into the future
    #     last_date = test_df.index[-1]
    #     future_dates = [last_date + dt.timedelta(days=i) for i in range(1, 91)]
    #     future_dates_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    #     future_forecast = model.predict(future_dates_ordinals)

    #     # Calculate metrics
    #     mse = mean_squared_error(y_test, y_pred)
    #     mae = mean_absolute_error(y_test, y_pred)
    #     r2 = r2_score(y_test, y_pred)
    #     rmse = math.sqrt(mse)
        
    #     save_to_csv("Random Forest", mae, mse, rmse, r2)

        # return train_df, test_df, X_train, X_test, y_train, y_test, y_pred, future_dates, future_forecast, mse, mae, r2

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None, None,None,None,None,None,None, None
    
def randomForest(stock_ticker):
    
    start_date = '2020-01-01'
    end_date = '2024-11-25'
    
    
    
    
    
# def gen_gru(stock_ticker):
#     # Step 1: Fetch stock data
#     data = yf.Ticker(stock_ticker)
#     hist = data.history(period="4y", interval="1d")
#     hist = hist[["Close"]]
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(hist)

#     # Step 2: Split the data into training and testing sets
#     train_size = int(len(scaled_data) * 0.8)
#     train_data = scaled_data[:train_size]
#     test_data = scaled_data[train_size:]

#     # Step 3: Create sequences for training and testing
#     def create_sequences(data, seq_length=60):
#         X, y = [], []
#         for i in range(seq_length, len(data)):
#             X.append(data[i-seq_length:i, 0])
#             y.append(data[i, 0])
#         return np.array(X), np.array(y)

#     seq_length = 60  # Use 60 previous days to predict the next
#     X_train, y_train = create_sequences(train_data, seq_length)
#     X_test, y_test = create_sequences(test_data, seq_length)

#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#     # Step 4: Build the GRU model
#     model = Sequential([
#         GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
#         Dropout(0.2),
#         GRU(units=50, return_sequences=False),
#         Dropout(0.2),
#         Dense(units=25),
#         Dense(units=1)
#     ])

#     # Compile the model
#     model.compile(optimizer='adam', loss='mean_squared_error')

#     # Train the model
#     model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

#     # Step 5: Predict on test data
#     predicted_prices = model.predict(X_test)
#     predicted_prices = scaler.inverse_transform(predicted_prices)
#     actual_prices = scaler.inverse_transform(test_data[seq_length:])

#     # Step 6: Future stock price prediction
#     input_date = input("Enter a future date (YYYY-MM-DD): ")
#     target_date = datetime.strptime(input_date, '%Y-%m-%d')

#     # Calculate days to predict from the last available date in the data
#     last_date = hist.index[-1].to_pydatetime().replace(tzinfo=None)  # Convert to timezone-naive datetime
#     days_to_predict = (target_date - last_date).days

#     if days_to_predict <= 0:
#         print("The entered date is not in the future. Please enter a future date.")
#     else:
#         # Generate predictions up to the target date
#         last_sequence = test_data[-seq_length:]
#         predicted_future_prices = []

#         for _ in range(days_to_predict):
#             prediction = model.predict(last_sequence.reshape(1, seq_length, 1))
#             predicted_future_prices.append(prediction[0, 0])
#             last_sequence = np.append(last_sequence[1:], prediction, axis=0)

#         predicted_future_prices = scaler.inverse_transform(np.array(predicted_future_prices).reshape(-1, 1))

#         # Print the predicted stock price for the specific future date
#         predicted_price_on_target_date = predicted_future_prices[-1][0]
#         print(f"Predicted Stock Price for {input_date}: ${predicted_price_on_target_date:.2f}")
#         actual_dates = hist.index[-len(actual_prices):]
#     return actual_dates, actual_prices, predicted_prices

def gen_gru(stock_ticker):
    try:
        # Step 1: Fetch stock data
        data = yf.Ticker(stock_ticker)
        hist = data.history(period="4y", interval="1d")
        hist = hist[["Close"]]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(hist)

        # Step 2: Split the data into training and testing sets
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        # Step 3: Create sequences for training and testing
        def create_sequences(data, seq_length=60):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i - seq_length:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        seq_length = 60  # Use 60 previous days to predict the next
        X_train, y_train = create_sequences(train_data, seq_length)
        X_test, y_test = create_sequences(test_data, seq_length)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Step 4: Build the GRU model
        model = Sequential([
            GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            GRU(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

        # Step 5: Predict on test data
        predicted_prices = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)
        actual_prices = scaler.inverse_transform(test_data[seq_length:])

        # Step 6: Future stock price prediction for 90 days
        last_sequence = test_data[-seq_length:]
        predicted_future_prices = []

        for _ in range(90):  # Predict 90 days ahead
            prediction = model.predict(last_sequence.reshape(1, seq_length, 1))
            predicted_future_prices.append(prediction[0, 0])
            last_sequence = np.append(last_sequence[1:], prediction, axis=0)

        predicted_future_prices = scaler.inverse_transform(np.array(predicted_future_prices).reshape(-1, 1))

        # Generate future dates
        last_date = hist.index[-1].to_pydatetime().replace(tzinfo=None)
        future_dates = [last_date + timedelta(days=i) for i in range(1, 91)]

        # Step 7: Calculate metrics
        mse = mean_squared_error(actual_prices, predicted_prices)
        mae = mean_absolute_error(actual_prices, predicted_prices)
        r2 = r2_score(actual_prices, predicted_prices)

        return train_data, test_data, X_train, X_test, y_train, y_test, predicted_prices, future_dates, predicted_future_prices, mse, mae, r2
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None, None, None, None, None, None, None, None


def gen_dt(stock_ticker):
    # Fetch stock data
    data = yf.Ticker(stock_ticker)
    hist = data.history(period="4y", interval="1d")
    hist = hist[["Close"]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(hist)

    # Split the data into training and testing sets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    # Prepare sequences for training and testing
    def create_sequences(data, seq_length=60):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i - seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    seq_length = 60  # Use 60 previous days to predict the next
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # Train Decision Tree model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Step 5: Predict on test data
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
    actual_prices = scaler.inverse_transform(test_data[seq_length:])

    # Step 6: Future stock price prediction for 90 days
    last_sequence = test_data[-seq_length:]
    predicted_future_prices = []

    for _ in range(90):  # Predict 90 days ahead
        prediction = model.predict(last_sequence.reshape(1, seq_length))
        predicted_future_prices.append(prediction[0])
        last_sequence = np.append(last_sequence[1:], prediction.reshape(1, 1), axis=0)

    predicted_future_prices = scaler.inverse_transform(np.array(predicted_future_prices).reshape(-1, 1))

    # Generate future dates
    last_date = hist.index[-1].to_pydatetime().replace(tzinfo=None)
    future_dates = [last_date + timedelta(days=i) for i in range(1, 91)]

    # Step 7: Calculate metrics
    mse = mean_squared_error(actual_prices, predicted_prices)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)

    return train_data, test_data, X_train, X_test, y_train, y_test, predicted_prices, future_dates, predicted_future_prices, mse, mae, r2

def gen_svm(stock_ticker):
    
    data = yf.Ticker(stock_ticker)
    hist = data.history(period="4y", interval="1d")
    hist = hist[["Close"]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(hist)

    # Step 2: Split the data into training and testing sets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    
    return

def save_to_csv(model_name,mae,mape, mse, rmse, r2):
    metrics_file = "D:/Stockastic/notebooks/model_metrics.csv"
    with open(metrics_file, "a") as f:
        f.write(f"{model_name},{mae},{mape},{mse},{rmse},{r2}\n")
        
def update_to_csv(model_name,mae,mape, mse, rmse, r2):
    
    return
        
