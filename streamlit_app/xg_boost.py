#imports

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

from updatecsv import update_to_csv

# Step 1: Fetch data from Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Step 2: Prepare features for XGBoost
def prepare_features(data, lookback=30):
    
    # Initialize scaler
    price_scaler = MinMaxScaler()
    
    # Scale prices
    prices = data['Close'].values
    prices_scaled = price_scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    
    # Create feature matrix
    X, y = [], []
    for i in range(len(prices_scaled) - lookback):
        X.append(prices_scaled[i:i+lookback])
        y.append(prices_scaled[i+lookback])
    
    return np.array(X), np.array(y), price_scaler

# Step 3: Prepare train and test datasets
def prepare_data(X, y, test_size=0.2):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        shuffle=False
    )
    
    return X_train, X_test, y_train, y_test

# Step 4: Train XGBoost model
def train_xgboost_model(X_train, y_train, X_test, y_test):
    
    # Define parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # Create and train model
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    return model

# Step 5: Make predictions
def make_predictions(model, X_test):
    
    predictions = model.predict(X_test)
    return predictions

# Step 6: Evaluate model performance
def evaluate_model(actual, predictions, price_scaler, ticker):
    
    # Inverse transform predictions
    actual_original = price_scaler.inverse_transform(actual.reshape(-1, 1))
    predictions_original = price_scaler.inverse_transform(predictions.reshape(-1, 1))
    
    # Calculate metrics
    mse = mean_squared_error(actual_original, predictions_original)
    mae = mean_absolute_error(actual_original, predictions_original)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_original - predictions_original) / actual_original)) * 100
    r2 = r2_score(actual_original, predictions_original)
    update_to_csv("D:/SAMLFRFM/notebooks/metrics/xgboost.csv",ticker, mae,mape, mse, rmse, r2)
    return mse, mae, rmse, mape

# Step 7: Forecast future prices
def predict_future_prices(model, last_prices, price_scaler, days=89):
    
    # Scale last prices
    last_prices_scaled = price_scaler.transform(last_prices.reshape(-1, 1)).flatten()
    
    # Forecast
    future_predictions_scaled = []
    current_window = last_prices_scaled.copy()
    
    for _ in range(days):
        # Predict next price
        next_pred = model.predict(current_window.reshape(1, -1))[0]
        
        # Append scaled prediction
        future_predictions_scaled.append(next_pred)
        
        # Update sliding window
        current_window = np.roll(current_window, -1)
        current_window[-1] = next_pred
    
    # Inverse transform predictions
    future_predictions = price_scaler.inverse_transform(
        np.array(future_predictions_scaled).reshape(-1, 1)
    ).flatten()
    
    return future_predictions

#runner
def get_xgboost(ticker):
    
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    lookback = 30  # Number of previous days to use for prediction
    forecast_days = 89
    
    # Fetch stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Prepare features
    X, y, price_scaler = prepare_features(stock_data, lookback)
    
    # Split data
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    
    # Train model
    model = train_xgboost_model(X_train, y_train, X_test, y_test)
    
    # Make predictions
    test_predictions = make_predictions(model, X_test)
    
    # Evaluate model
    evaluate_model(y_test, test_predictions, price_scaler, ticker)
    
    # Predict future prices
    last_prices = stock_data['Close'].values[-lookback:]
    future_dates = pd.date_range(start=stock_data.index[-1], periods=forecast_days+1)[1:]
    future_predictions = predict_future_prices(model, last_prices, price_scaler, days=forecast_days)
    
    # Calculate test index
    test_index = stock_data.index[-len(y_test):]
    
    return stock_data, y_test, test_predictions, test_index, future_dates, future_predictions, price_scaler