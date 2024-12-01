#imports
from datetime import timedelta
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
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

# Step 3: Train an SVM model
def train_svm_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
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
    
    update_to_csv("D:/SAMLFRFM/notebooks/metrics/svm.csv", ticker, mae,mape, mse, rmse, r2)

# Step 5: Predict future price
def predict_future_prices(model, scaler, last_date, days=90):
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_dates_ordinal = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
    future_dates_scaled = scaler.transform(future_dates_ordinal)
    predicted_prices = model.predict(future_dates_scaled)
    return future_dates, predicted_prices

# Step 6: Plot predictions
def plot_results1(y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='red')
    plt.legend()
    plt.title('Actual vs Predicted Prices')
    plt.show()

def plot_results2(future_dates, future_prices):
    plt.figure(figsize=(10, 5))
    plt.plot(future_dates, future_prices, label='Future Prices', color='green')
    plt.legend()
    plt.title('Actual vs Predicted Prices')
    plt.show()
    
def plot_combined_results(stock_data, y_test, y_pred, future_dates, future_prices):
    plt.figure(figsize=(15, 7))
    
    # Get the dates corresponding to test data
    test_dates = stock_data.index[-len(y_test):]
    
    # Plot historical data (actual vs predicted)
    plt.plot(test_dates, y_test, label='Actual Prices', color='blue')
    plt.plot(test_dates, y_pred, label='Predicted Prices', color='red')
    
    # Plot future predictions
    plt.plot(future_dates, future_prices, label='Future Prices', color='green')
    
    plt.legend()
    plt.title(f'Stock Price Analysis: Historical and Future Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

#runner
def get_svm(ticker):
    
    start_date = '2020-01-01'
    end_date = '2024-11-25'
    
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    features, targets = prepare_data(stock_data)
    model, X_train, X_test, y_train, y_test, y_pred, scaler = train_svm_model(features, targets)
    evaluate_model(y_test, y_pred, ticker)
    
    # Predict future prices
    last_date = stock_data.index[-1].to_pydatetime().replace(tzinfo=None)
    future_dates, future_prices = predict_future_prices(model, scaler, last_date, days=90)
    
    return stock_data, y_test, y_pred, future_dates, future_prices
    
