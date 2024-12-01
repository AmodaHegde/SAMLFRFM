#imports

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Model Metrics",
)
# Title for the Streamlit app
st.title("Model Metrics Viewer")

# File path (replace with your actual file path)
file1 = "D:/SAMLFRFM/notebooks/metrics/rf.csv"
file2 = "D:/SAMLFRFM/notebooks/metrics/svm.csv"
file3 = "D:/SAMLFRFM/notebooks/metrics/dt.csv"
file4 = "D:/SAMLFRFM/notebooks/metrics/knn.csv"
file5 = "D:/SAMLFRFM/notebooks/metrics/arima.csv"
file6 = "D:/SAMLFRFM/notebooks/metrics/lstm.csv"
file7 = "D:/SAMLFRFM/notebooks/metrics/gru.csv"
file8 = "D:/SAMLFRFM/notebooks/metrics/xgboost.csv"

try:
    # Read the CSV file
    df = pd.read_csv(file1)

    # Display the data as a table
    st.subheader("Random Forest Metrics Data")
    st.dataframe(df)

    # Optionally, display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    # Read the CSV file
    df = pd.read_csv(file2)

    # Display the data as a table
    st.subheader("SVM Metrics Data")
    st.dataframe(df)

    # Optionally, display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    # Read the CSV file
    df = pd.read_csv(file3)

    # Display the data as a table
    st.subheader("Decision Tree Metrics Data")
    st.dataframe(df)

    # Optionally, display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    # Read the CSV file
    df = pd.read_csv(file4)

    # Display the data as a table
    st.subheader("KNN Metrics Data")
    st.dataframe(df)

    # Optionally, display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    # Read the CSV file
    df = pd.read_csv(file5)

    # Display the data as a table
    st.subheader("ARIMA Metrics Data")
    st.dataframe(df)

    # Optionally, display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    # Read the CSV file
    df = pd.read_csv(file6)

    # Display the data as a table
    st.subheader("LSTM Metrics Data")
    st.dataframe(df)

    # Optionally, display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    # Read the CSV file
    df = pd.read_csv(file7)

    # Display the data as a table
    st.subheader("GRU Metrics Data")
    st.dataframe(df)

    # Optionally, display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    # Read the CSV file
    df = pd.read_csv(file8)

    # Display the data as a table
    st.subheader("XGBoost Metrics Data")
    st.dataframe(df)

    # Optionally, display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

except FileNotFoundError:
    st.error("File not found. Please upload the CSV file to the specified path.")
except Exception as e:
    st.error(f"An error occurred: {e}")
