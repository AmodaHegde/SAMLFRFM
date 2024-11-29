from datetime import timedelta
from sklearn.metrics import r2_score
from sklearn.discriminant_analysis import StandardScaler
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt

#from helper import update_to_csv

import pandas as pd

from updatecsv import update_to_csv

# Step 1: Fetch data from Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Step 2: Prepare the data with lagged features
def prepare_data(data, lag=5):
    data['Date'] = data.index
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    
    # Add lagged features
    for i in range(1, lag + 1):
        data[f'Lag_{i}'] = data['Close'].shift(i)
    
    # Drop rows with NaN values due to lagging
    data.dropna(inplace=True)
    
    # Features and target
    features = data[['Year', 'Month', 'Day'] + [f'Lag_{i}' for i in range(1, lag + 1)]]
    targets = data['Close']
    return np.array(features), np.array(targets)

# Step 3: Train a Random Forest model
def train_random_forest_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # scale = StandardScaler()
    # X_train = scale.fit_transform(X_train)
    # X_test = scale.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=500, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, y_pred

# Step 4: Evaluate model performance
def evaluate_model(y_test, y_pred, ticker):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    #print(f"Mean Squared Error (MSE): {mse}")
    #print(f"Mean Absolute Error (MAE): {mae}")
    update_to_csv("D:/SAMLFRFM/notebooks/metrics/rf.csv",ticker, mae,mape, mse, rmse, r2)

# Step 5: Predict future prices
def predict_future_prices(model, last_known_data, last_date, days=90):
    future_prices = []
    current_data = last_known_data.copy()
    
    for _ in range(days):
        # Predict the next price
        next_price = model.predict([current_data])[0]
        future_prices.append(next_price)
        
        # Update lagged features for the next prediction
        current_data = np.roll(current_data, shift=-1)
        current_data[-1] = next_price
    
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    return future_dates, future_prices

# def plot_combined_results(stock_data, y_test, y_pred, future_dates, future_prices):
#     plt.figure(figsize=(15, 7))
    
#     # Get the dates corresponding to test data
#     test_dates = stock_data.index[-len(y_test):]
    
#     # Plot historical data (actual vs predicted)
#     plt.plot(test_dates, y_test, label='Actual Prices', color='blue')
#     plt.plot(test_dates, y_pred, label='Predicted Prices', color='red')
    
#     # Plot future predictions
#     plt.plot(future_dates, future_prices, label='Future Prices', color='green')
    
#     plt.legend()
#     plt.title(f'Stock Price Analysis: Historical and Future Predictions')
#     plt.xlabel('Date')
#     plt.ylabel('Price ($)')
    
#     # Rotate x-axis labels for better readability
#     plt.xticks(rotation=45)
    
#     # Adjust layout to prevent label cutoff
#     plt.tight_layout()
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.show()

#Main
def get_randomforest(ticker):
    start_date = '2020-01-01'
    end_date = '2024-11-25'
    
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    features, targets = prepare_data(stock_data, lag=5)
    model, X_train, X_test, y_train, y_test, y_pred = train_random_forest_model(features, targets)
    evaluate_model(y_test, y_pred, ticker)
    
    # Predict future prices
    last_known_data = features[-1]  # Last feature set
    last_date = stock_data.index[-1].to_pydatetime().replace(tzinfo=None)
    future_dates, future_prices = predict_future_prices(model, last_known_data, last_date, days=90)
    
    # Plot results
    #plot_results1(y_test, y_pred)
    #plot_results2(future_dates, future_prices)
    #plot_combined_results(stock_data, y_test, y_pred, future_dates, future_prices)
    
    # Print future prices
    #print(future_dates[2:6], future_prices[2:6])
    #print(f"The predicted stock price for {future_dates[-1].strftime('%Y-%m-%d')} is ${future_prices[-1]:.2f}")
    return stock_data, y_test, y_pred, future_dates, future_prices
    
    