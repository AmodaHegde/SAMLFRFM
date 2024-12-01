# Imports
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import streamlit as st

# Import functions
from helper import *
from randomforest import get_randomforest
from svm import get_svm
from decisiontree import get_dt
from knn import get_knn
from arima import get_arima
from lstm import get_lstm
from gru import get_gru
from xg_boost import get_xgboost

# Configure the page
st.set_page_config(
    page_title="Stock Price Prediction",
)


#Sidebar Start

# Add a sidebar
st.sidebar.markdown("## **User Input Features**")

# Fetch and store the stock data
stock_dict = fetch_stocks()

# Add a dropdown for selecting the stock
st.sidebar.markdown("### **Select stock**")
stock_ticker = st.sidebar.selectbox("Choose a stock", list(stock_dict.keys()))

# Fetch and store periods and intervals
periods = fetch_periods_intervals()

# Add a selector for period
st.sidebar.markdown("### **Select period**")
period = st.sidebar.selectbox("Choose a period", list(periods.keys()))

# Add a selector for interval
st.sidebar.markdown("### **Select interval**")
interval = st.sidebar.selectbox("Choose an interval", periods[period])


#Select a machine learning model
models = fetch_models()
st.sidebar.markdown("### **Select model**")
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))
model = models[model_name]

#Sidebar End

#Title

# Add title to the app
st.markdown("# **Stock Price Prediction**")

#Title End

# Fetch the stock historical data
stock_data = fetch_stock_history(stock_ticker, period, interval)

#Historical Data Graph

# Add a title to the historical data graph
st.markdown("## **Historical Data**")

# Create a plot for the historical data
fig = go.Figure(
    data=[
        go.Candlestick(
            x=stock_data.index,
            open=stock_data["Open"],
            high=stock_data["High"],
            low=stock_data["Low"],
            close=stock_data["Close"],
        )
    ]
)

# Customize the historical data graph
fig.update_layout(xaxis_rangeslider_visible=False)

# Use the native streamlit theme.
st.plotly_chart(fig, use_container_width=True)

#Historical Data Graph End


#Stock Prediction Graphs

if model == 1:
    stock_data, y_test, y_pred, future_dates, future_prices = get_randomforest(stock_ticker)
    
    st.markdown("## **Stock Prediction**")
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Get the dates corresponding to test data
    test_dates = stock_data.index[-len(y_test):]
    
    # Add traces for each dataset
    fig.add_trace(go.Scatter(
        x=test_dates, 
        y=y_test, 
        mode='lines', 
        name='Actual Prices', 
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=test_dates, 
        y=y_pred, 
        mode='lines', 
        name='Predicted Prices', 
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=future_prices, 
        mode='lines', 
        name='Future Prices', 
        line=dict(color='green')
    ))
    
    # Customize layout
    fig.update_layout(
        title='Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='closest',
        legend_title_text='Price Types',
    )
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)
    
    # Display in Streamlit
    st.plotly_chart(fig)
    
if model == 2:
    stock_data, y_test, y_pred, future_dates, future_prices = get_svm(stock_ticker)
    
    st.markdown("## **Stock Prediction**")
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Get the dates corresponding to test data
    test_dates = stock_data.index[-len(y_test):]
    
    # Add traces for each dataset
    fig.add_trace(go.Scatter(
        x=test_dates, 
        y=y_test, 
        mode='lines', 
        name='Actual Prices', 
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=test_dates, 
        y=y_pred, 
        mode='lines', 
        name='Predicted Prices', 
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=future_prices, 
        mode='lines', 
        name='Future Prices', 
        line=dict(color='green')
    ))
    
    # Customize layout
    fig.update_layout(
        title='Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='closest',
        legend_title_text='Price Types',
    )
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)
    
    # Display in Streamlit
    st.plotly_chart(fig)
    
if model == 3:
    stock_data, y_test, y_pred, future_dates, future_prices = get_dt(stock_ticker)
    
    st.markdown("## **Stock Prediction**")
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Get the dates corresponding to test data
    test_dates = stock_data.index[-len(y_test):]
    
    # Add traces for each dataset
    fig.add_trace(go.Scatter(
        x=test_dates, 
        y=y_test, 
        mode='lines', 
        name='Actual Prices', 
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=test_dates, 
        y=y_pred, 
        mode='lines', 
        name='Predicted Prices', 
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=future_prices, 
        mode='lines', 
        name='Future Prices', 
        line=dict(color='green')
    ))
    
    # Customize layout
    fig.update_layout(
        title='Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='closest',
        legend_title_text='Price Types',
    )
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)
    
    # Display in Streamlit
    st.plotly_chart(fig)
    
if model == 4:
    stock_data, y_test, y_pred, future_dates, future_prices = get_knn(stock_ticker)
    
    st.markdown("## **Stock Prediction**")
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Get the dates corresponding to test data
    test_dates = stock_data.index[-len(y_test):]
    
    # Add traces for each dataset
    fig.add_trace(go.Scatter(
        x=test_dates, 
        y=y_test, 
        mode='lines', 
        name='Actual Prices', 
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=test_dates, 
        y=y_pred, 
        mode='lines', 
        name='Predicted Prices', 
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=future_prices, 
        mode='lines', 
        name='Future Prices', 
        line=dict(color='green')
    ))
    
    # Customize layout
    fig.update_layout(
        title='Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='closest',
        legend_title_text='Price Types',
    )
    
    # Rotate x-axis labels
    fig.update_xaxes(tickangle=45)
    
    # Display in Streamlit
    st.plotly_chart(fig)

if model == 5:
    train_data, test_data, test_predictions, test_conf_int, future_dates, future_prices, future_conf_int = get_arima(stock_ticker)
    
    st.markdown("## **Stock Prediction**")
    
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(15, 7))
    
    #plt.figure(figsize=(15, 7))
    
    # Plot training data
    ax.plot(train_data.index, train_data, label='Training Data', color='gray', alpha=0.5)
    
    # Plot test data and predictions with confidence intervals
    ax.plot(test_data.index, test_data, label='Actual Test Data', color='blue')
    ax.plot(test_data.index, test_predictions, label='Predicted Prices', color='red')
    ax.fill_between(test_data.index, 
                     test_conf_int[:, 0], 
                     test_conf_int[:, 1], 
                     color='red', 
                     alpha=0.1, 
                     label='Test Prediction CI')
    
    # Plot future predictions with confidence intervals
    ax.plot(future_dates, future_prices, label='Future Predictions', color='green')
    ax.fill_between(future_dates, 
                     future_conf_int[:, 0], 
                     future_conf_int[:, 1], 
                     color='green', 
                     alpha=0.1, 
                     label='Future Prediction CI')
    
    ax.legend()
    plt.title('Stock Price Analysis using Auto ARIMA with Confidence Intervals')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
    plt.close(fig)

if model == 6:
    y_test, y_pred, future_dates, future_prices, scaler = get_lstm(stock_ticker)
    # Rescale the predictions and actual values to their original scale
    
    st.markdown("## **Stock Prediction**")
    
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(15, 7))
    
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    
    plt.figure(figsize=(12, 6))
    
    # Plot actual vs predicted
    ax.plot(range(len(y_test_rescaled)), y_test_rescaled, label='Actual Prices (Test Set)', color='blue')
    ax.plot(range(len(y_pred_rescaled)), y_pred_rescaled, label='Predicted Prices (Test Set)', color='red')
    
    # Append future predictions to the same graph
    future_indices = range(len(y_test_rescaled), len(y_test_rescaled) + len(future_prices))
    ax.plot(future_indices, future_prices, label='Future Predicted Prices', color='green')
    
    plt.title('Combined Graph: Actual vs Predicted and Future Predictions')
    plt.xlabel('Time')
    plt.ylabel('Stock Prices')
    plt.legend()
    plt.show()
    st.pyplot(fig)
    plt.close(fig)

if model == 7:
    stock_data, y_test, y_pred, future_dates, future_prices = get_gru(stock_ticker)
    
    st.markdown("## **Stock Prediction**")
    
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Get the dates corresponding to test data
    test_dates = stock_data.index[-len(y_test):]
    
    # Plot all data
    ax.plot(test_dates, y_test, label='Actual Prices', color='blue')
    ax.plot(test_dates, y_pred, label='Predicted Prices', color='red')
    ax.plot(future_dates, future_prices, label='Future Prices', color='green')
    
    # Customize plot
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig)
    
    # Clear the figure
    plt.close(fig)
    
if model == 8:
    
    stock_data, test_data, test_predictions, test_index, future_dates, future_predictions, price_scaler = get_xgboost(stock_ticker)
    
    st.markdown("## **Stock Prediction**")
    
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Plot historical data
    ax.plot(stock_data.index, stock_data['Close'], label='Historical Price', color='gray', alpha=0.5)
    
    # Plot test data and predictions
    ax.plot(
        test_index, 
        price_scaler.inverse_transform(test_data.reshape(-1, 1)), 
        label='Actual Test Prices', 
        color='blue'
    )
    ax.plot(
        test_index, 
        price_scaler.inverse_transform(test_predictions.reshape(-1, 1)), 
        label='Predicted Test Prices', 
        color='red', 
        linestyle='--'
    )
    
    # Plot future predictions
    ax.plot(future_dates, future_predictions, label='Future Predictions', color='green')
    
    plt.title('Stock Price Prediction using XGBoost')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    st.pyplot(fig)
    plt.close(fig)



