# Imports
import plotly.graph_objects as go
import streamlit as st

# Import helper functions
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


#####Sidebar Start#####

# Add a sidebar
st.sidebar.markdown("## **User Input Features**")

# Fetch and store the stock data
stock_dict = fetch_stocks()

# Add a dropdown for selecting the stock
st.sidebar.markdown("### **Select stock**")
stock_ticker = st.sidebar.selectbox("Choose a stock", list(stock_dict.keys()))

# Add a selector for stock exchange
# st.sidebar.markdown("### **Select stock exchange**")
# stock_exchange = st.sidebar.radio("Choose a stock exchange", ("BSE", "NSE"), index=0)

# # Build the stock ticker
# stock_ticker = f"{stock_dict[stock]}.{'BO' if stock_exchange == 'BSE' else 'NS'}"

# Add a disabled input for stock ticker
# st.sidebar.markdown("### **Stock ticker**")
# st.sidebar.text_input(
#     label="Stock ticker code", placeholder=stock_ticker, disabled=True
# )

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

#####Sidebar End#####


#####Title#####

# Add title to the app
st.markdown("# **Stock Price Prediction**")

# Add a subtitle to the app
st.markdown("##### **Enhance Investment Decisions through Data-Driven Forecasting**")

#####Title End#####


# Fetch the stock historical data
stock_data = fetch_stock_history(stock_ticker, period, interval)


#####Historical Data Graph#####

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

#####Historical Data Graph End#####


#####Stock Prediction Graph#####

# Unpack the data
if model == 9:    
    train_df, test_df, forecast, predictions = generate_stock_prediction1(stock_ticker)

    # Check if the data is not None
    if train_df is not None and (forecast >= 0).all() and (predictions >= 0).all():
        # Add a title to the stock prediction graph
        st.markdown("## **Stock Prediction**")

        # Create a plot for the stock prediction
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=train_df.index,
                    y=train_df["Close"],
                    name="Train",
                    mode="lines",
                    line=dict(color="blue"),
                ),
                go.Scatter(
                    x=test_df.index,
                    y=test_df["Close"],
                    name="Test",
                    mode="lines",
                    line=dict(color="orange"),
                ),
                go.Scatter(
                    x=forecast.index,
                    y=forecast,
                    name="Forecast",
                    mode="lines",
                    line=dict(color="red"),
                ),
                go.Scatter(
                    x=test_df.index,
                    y=predictions,
                    name="Test Predictions",
                    mode="lines",
                    line=dict(color="green"),
                ),
            ]
        )

        # Customize the stock prediction graph
        fig.update_layout(xaxis_rangeslider_visible=False)

        # Use the native streamlit theme.
        st.plotly_chart(fig, use_container_width=True)

    # If the data is None
    else:
        # Add a title to the stock prediction graph
        st.markdown("## **Stock Prediction**")

        # Add a message to the stock prediction graph
        st.markdown("### **No data available for the selected stock**")

    #####Stock Prediction Graph End#####
    
# if model == 1 :    
#     train_df, test_df, X_train, X_test, y_train, y_test, y_pred, future_dates, future_forecast, mse, mae, r2 = generate_stock_prediction2(stock_ticker)

#     # Check if the data is not None
#     if X_train is not None and (future_forecast >= 0).all() and (y_pred >= 0).all():
#         # Add a title to the stock prediction graph
#         st.markdown("## **Stock Prediction**")
#         # Plot results
#         plt.figure(figsize=(12, 6))
#         plt.plot(train_df.index, y_train, label="Train Data")
#         plt.plot(test_df.index, y_test, label="Test Data")
#         plt.plot(test_df.index, y_pred, label="Predicted Data")
#         plt.plot(future_dates, future_forecast, label="90-Day Forecast", linestyle="--")
#         plt.legend()
#         st.pyplot(plt)
        
#         # Display metrics
#         st.write("Mean Squared Error:", mse)
#         st.write("Mean Absolute Error:", mae)
#         st.write("R-Squared:", r2)

#     # If the data is None
#     else:
#         # Add a title to the stock prediction graph
#         st.markdown("## **Stock Prediction**")

#         # Add a message to the stock prediction graph
#         st.markdown("### **No data available for the selected stock**")

#     #####Stock Prediction Graph End#####

if model == 1:
    stock_data, y_test, y_pred, future_dates, future_prices = get_randomforest(stock_ticker)
    
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
    
    # Display metrics
    st.markdown("### **Model Performance Metrics**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.2f}")
    col2.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, y_pred):.2f}")
    col3.metric("R-squared Score", f"{r2_score(y_test, y_pred):.2f}")
    
if model == 2:
    stock_data, y_test, y_pred, future_dates, future_prices = get_svm(stock_ticker)
    
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
    
    # Display metrics
    st.markdown("### **Model Performance Metrics**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.2f}")
    col2.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, y_pred):.2f}")
    col3.metric("R-squared Score", f"{r2_score(y_test, y_pred):.2f}")
    
if model == 3:
    stock_data, y_test, y_pred, future_dates, future_prices = get_dt(stock_ticker)
    
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
    
    # Display metrics
    st.markdown("### **Model Performance Metrics**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.2f}")
    col2.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, y_pred):.2f}")
    col3.metric("R-squared Score", f"{r2_score(y_test, y_pred):.2f}")

if model == 4:
    stock_data, y_test, y_pred, future_dates, future_prices = get_knn(stock_ticker)
    
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
    
    # Display metrics
    st.markdown("### **Model Performance Metrics**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.2f}")
    col2.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, y_pred):.2f}")
    col3.metric("R-squared Score", f"{r2_score(y_test, y_pred):.2f}")


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
# if model == 5:    
#     y_pred = gen_arima(stock_ticker)

#     # Check if the data is not None
#     if y_pred is not None:
#         # Add a title to the stock prediction graph
#         st.markdown("## **Stock Prediction**")

#         # Create a plot for the stock prediction
#         fig = go.Figure(
#             data=[
#                 go.Scatter(
#                     x=y_pred.index,
#                     y=y_pred,
#                     name="Prediction",
#                     mode="lines",
#                     line=dict(color="blue"),
#                 ),
#             ]
#         )

#         # Customize the stock prediction graph
#         fig.update_layout(xaxis_rangeslider_visible=False)

#         # Use the native streamlit theme.
#         st.plotly_chart(fig, use_container_width=True)

#     # If the data is None
#     else:
#         # Add a title to the stock prediction graph
#         st.markdown("## **Stock Prediction**")

#         # Add a message to the stock prediction graph
#         st.markdown("### **No data available for the selected stock**")

#     #####Stock Prediction Graph End#####
    
# if model == 6:    
#     actual_dates, actual_prices, predicted_prices = gen_lstm(stock_ticker)

#     # Check if the data is not None
#     if predicted_prices is not None:
#         # Add a title to the stock prediction graph
#         st.markdown("## **Stock Prediction**")

#         fig = go.Figure(
#             data=[
#                 go.Scatter(
#                     x=actual_dates, 
#                     y=actual_prices.flatten(),
#                     name="Actual Prices",
#                     mode="lines",
#                     line=dict(color="blue"),
#                 ),
#                 go.Scatter(
#                     x=actual_dates,
#                     y=predicted_prices.flatten(),  # Flatten the array to a 1D list
#                     name="Predicted Prices",
#                     mode="lines",
#                     line=dict(color="orange"),
#                 ),
#                 ]
#             )
#         st.plotly_chart(fig)

#     # If the data is None
#     else:
#         # Add a title to the stock prediction graph
#         st.markdown("## **Stock Prediction**")

#         # Add a message to the stock prediction graph
#         st.markdown("### **No data available for the selected stock**")

#     #####Stock Prediction Graph End#####

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
    
    
# if model == 7:    
#     actual_dates, actual_prices, predicted_prices = gen_gru(stock_ticker)

#     # Check if the data is not None
#     if predicted_prices is not None:
#         # Add a title to the stock prediction graph
#         st.markdown("## **Stock Prediction**")

#         fig = go.Figure(
#             data=[
#                 go.Scatter(
#                     x=actual_dates, 
#                     y=actual_prices.flatten(),
#                     name="Actual Prices",
#                     mode="lines",
#                     line=dict(color="blue"),
#                 ),
#                 go.Scatter(
#                     x=actual_dates,
#                     y=predicted_prices.flatten(),  # Flatten the array to a 1D list
#                     name="Predicted Prices",
#                     mode="lines",
#                     line=dict(color="orange"),
#                 ),
#                 go.Scatter(
#                     x=future_dates,
#                     y=predicted_future_prices.flatten(),  # Flatten the array to a 1D list
#                     name="Forecasted Prices",
#                     mode="lines",
#                     line=dict(color="orange"),
#                 ),
#                 ]
#             )
#         st.plotly_chart(fig)

#     # If the data is None
#     else:
#         # Add a title to the stock prediction graph
#         st.markdown("## **Stock Prediction**")

#         # Add a message to the stock prediction graph
#         st.markdown("### **No data available for the selected stock**")

#     #####Stock Prediction Graph End#####

# if model == 7 :    
#     train_df, test_df, X_train, X_test, y_train, y_test, y_pred, future_dates, future_forecast, mse, mae, r2 = gen_gru(stock_ticker)

#     # Check if the data is not None
#     if X_train is not None and (future_forecast >= 0).all() and (y_pred >= 0).all():
#         # Add a title to the stock prediction graph
#         st.markdown("## **Stock Prediction**")
#         # Plot results
#         plt.figure(figsize=(12, 6))
#         plt.plot(train_df.index, y_train, label="Train Data")
#         plt.plot(test_df.index, y_test, label="Test Data")
#         plt.plot(test_df.index, y_pred, label="Predicted Data")
#         plt.plot(future_dates, future_forecast, label="90-Day Forecast", linestyle="--")
#         plt.legend()
#         st.pyplot(plt)
        
#         # Display metrics
#         st.write("Mean Squared Error:", mse)
#         st.write("Mean Absolute Error:", mae)
#         st.write("R-Squared:", r2)

#     # If the data is None
#     else:
#         # Add a title to the stock prediction graph
#         st.markdown("## **Stock Prediction**")

#         # Add a message to the stock prediction graph
#         st.markdown("### **No data available for the selected stock**")

#     #####Stock Prediction Graph End#####

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
    
    # Display metrics
    st.markdown("### **Model Performance Metrics**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.2f}")
    col2.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, y_pred):.2f}")
    col3.metric("R-squared Score", f"{r2_score(y_test, y_pred):.2f}")

if model == 8:
    
    stock_data, test_data, test_predictions, test_index, future_dates, future_predictions, price_scaler = get_xgboost(stock_ticker)
    
    st.markdown("## **Stock Prediction**")
    
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(15, 7))
    
    #plt.figure(figsize=(15, 7))
    
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

if model == 3 :    
    train_index, test_index, X_train, X_test, y_train, y_test, y_pred, future_dates, future_forecast, mse, mae, r2 = get_dt(stock_ticker)

    # Check if the data is not None
    if X_train is not None and (future_forecast >= 0).all() and (y_pred >= 0).all():
        # Add a title to the stock prediction graph
        st.markdown("## **Stock Prediction**")
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(train_index[:len(y_train)], y_train, label="Train Data", color="Blue")
        plt.plot(test_index[:len(y_test)], y_test, label="Test Data", color = "Red")
        plt.plot(test_index[:len(y_pred)], y_pred, label="Predicted Data", color = "Black")
        plt.plot(future_dates, future_forecast, label="90-Day Forecast", linestyle="--")
        plt.legend()
        st.pyplot(plt)
        
        # Display metrics
        st.write("Mean Squared Error:", mse)
        st.write("Mean Absolute Error:", mae)
        st.write("R-squared:", r2)



