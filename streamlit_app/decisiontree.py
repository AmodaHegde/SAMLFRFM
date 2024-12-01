#import libraries
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
from datetime import timedelta

from updatecsv import update_to_csv

# Step 1: Fetch data from Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Step 2: Prepare the data
def prepare_data(data):
    data['Date'] = data.index
    data['Date'] = data['Date'].map(pd.Timestamp.toordinal)  # Convert dates to ordinal
    features = np.array(data[['Date']])
    targets = np.array(data['Close'])
    return features, targets

# Step 3: Train a Decision Tree model
def train_decision_tree_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return model, X_train, X_test, y_train, y_test, y_pred, scaler

# Step 4: Evaluate model performance
def evaluate_model(y_test, y_pred, ticker):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    
    update_to_csv("D:/SAMLFRFM/notebooks/metrics/dt.csv",ticker, mae,mape, mse, rmse, r2)

# Step 5: Predict future prices
def predict_future_prices(model, scaler, last_date, days=90):
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_dates_ordinal = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
    future_dates_scaled = scaler.transform(future_dates_ordinal)
    predicted_prices = model.predict(future_dates_scaled)
    return future_dates, predicted_prices

#runner function
def get_dt(ticker):
    
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    features, targets = prepare_data(stock_data)
    model, X_train, X_test, y_train, y_test, y_pred, scaler = train_decision_tree_model(features, targets)
    evaluate_model(y_test, y_pred, ticker)
    
    # Predict future prices
    last_date = stock_data.index[-1].to_pydatetime().replace(tzinfo=None)
    future_dates, future_prices = predict_future_prices(model, scaler, last_date, days=90)
    
    return stock_data, y_test, y_pred, future_dates, future_prices
