#import libraries
import yfinance as yf
import numpy as np
from datetime import timedelta
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score
import warnings

from updatecsv import update_to_csv
warnings.filterwarnings('ignore')

# Step 1: Fetch data from Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Step 2: Prepare the data for Auto ARIMA
def prepare_data(data):
    prices = data['Close']
    train_size = int(len(prices) * 0.8)
    train = prices[:train_size]
    test = prices[train_size:]
    return train, test

# Step 3: Train Auto ARIMA model
def train_auto_arima_model(train_data):
    model = auto_arima(train_data,
                      start_p=0, start_q=0,
                      max_p=5, max_q=5,
                      m=7,  # Monthly seasonal pattern
                      d=1,  # Difference order
                      seasonal=True,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)
    
    print(f"ARIMA order (p,d,q): {model.order}")
    print(f"Seasonal order (P,D,Q,s): {model.seasonal_order}")
    return model

# Step 4: Make predictions with confidence intervals
def make_predictions(model, n_periods, alpha=0.05):
    predictions, conf_int = model.predict(n_periods=n_periods, return_conf_int=True, alpha=alpha)
    return predictions, conf_int

# Step 5: Predict future prices with confidence intervals
def predict_future_prices(model, last_date, days=90, alpha=0.05):
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    forecast, conf_int = model.predict(n_periods=days, return_conf_int=True, alpha=alpha)
    return future_dates, forecast, conf_int

# Step 6: Evaluate model performance
def evaluate_model(y_test, y_pred, ticker):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    
    update_to_csv("D:/SAMLFRFM/notebooks/metrics/arima.csv",ticker, mae,mape, mse, rmse, r2)

#function to run
def get_arima(ticker):
    
    start_date = '2020-01-01'
    end_date = '2024-11-25'
    
    # Fetch and prepare data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    train_data, test_data = prepare_data(stock_data)
    
    # Train model
    model = train_auto_arima_model(train_data)
    
    # Make predictions for test period with confidence intervals
    test_predictions, test_conf_int = make_predictions(model, len(test_data))
    
    # Evaluate model
    evaluate_model(test_data, test_predictions, ticker)
    
    # Predict future prices with confidence intervals
    last_date = stock_data.index[-1].to_pydatetime().replace(tzinfo=None)
    future_dates, future_prices, future_conf_int = predict_future_prices(model, last_date, days=90)
    
    # Plot results
    return train_data, test_data, test_predictions, test_conf_int, future_dates, future_prices, future_conf_int