from datetime import timedelta
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU

from updatecsv import update_to_csv

# Step 1: Fetch data from Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Step 2: Prepare the data with lagged features and scaling
def prepare_data(data, lag=5):
    data['Date'] = data.index
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Scaled_Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data['Scaled_Close'].values[i-lag:i])
        y.append(data['Scaled_Close'].values[i])
    
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

# Step 3: Train a GRU model
def train_gru_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        GRU(50),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, y_pred

# Step 4: Evaluate model performance with proper inverse scaling
def evaluate_model(y_test, y_pred, scaler, ticker):
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred = scaler.inverse_transform(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    #print(f"Mean Squared Error (MSE): {mse}")
    #print(f"Mean Absolute Error (MAE): {mae}")
    update_to_csv("D:/SAMLFRFM/notebooks/metrics/gru.csv",ticker, mae,mape, mse, rmse, r2)
    return y_test, y_pred

# Step 5: Predict future prices
def predict_future_prices(model, last_known_data, last_date, scaler, days=90):
    future_prices = []
    current_data = last_known_data.copy()
    
    for _ in range(days):
        current_data_scaled = scaler.transform(current_data.reshape(-1, 1))
        current_data_scaled = np.array([current_data_scaled])
        
        next_price_scaled = model.predict(current_data_scaled)[0, 0]
        next_price = scaler.inverse_transform([[next_price_scaled]])[0, 0]
        future_prices.append(next_price)
        
        # Update lagged features for the next prediction
        current_data = np.roll(current_data, shift=-1)
        current_data[-1] = next_price
    
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    return future_dates, future_prices



# Example usage
#if __name__ == "__main__":
def get_gru(ticker):
    #ticker = 'TSLA'  # Example stock ticker
    start_date = '2020-01-01'
    end_date = '2024-11-25'
    
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    features, targets, scaler = prepare_data(stock_data, lag=5)
    
    # Reshape features for GRU model
    features = features.reshape(features.shape[0], features.shape[1], 1)
    
    model, X_train, X_test, y_train, y_test, y_pred = train_gru_model(features, targets)
    y_test_original, y_pred_original = evaluate_model(y_test, y_pred, scaler, ticker)
    
    # Predict future prices
    last_known_data = stock_data['Close'].values[-5:]  # Last 5 known prices
    last_date = stock_data.index[-1].to_pydatetime().replace(tzinfo=None)
    future_dates, future_prices = predict_future_prices(model, last_known_data,last_date, scaler, days=90)
    
    # Plot results
    #plot_results1(y_test_original, y_pred_original)
    #plot_results2(future_dates, future_prices)
    
    # Print future prices
    print(future_dates[2:6], future_prices[2:6])
    print(f"The predicted stock price for {future_dates[-1].strftime('%Y-%m-%d')} is ${future_prices[-1]:.2f}")
    
    return stock_data, y_test_original, y_pred_original, future_dates, future_prices