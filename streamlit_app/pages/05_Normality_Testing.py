#imports

import streamlit as st
import pandas as pd
from scipy.stats import shapiro

# Title
st.title("Testing Normality")
st.write("## Shapiro-Wilk Test for Normality Across Multiple Models")

# File paths and corresponding model names
files_and_names = {
    "Random Forest": "D:/SAMLFRFM/notebooks/metrics/rf.csv",
    "Support Vector Machine": "D:/SAMLFRFM/notebooks/metrics/svm.csv",
    "Decision Tree": "D:/SAMLFRFM/notebooks/metrics/dt.csv",
    "KNN": "D:/SAMLFRFM/notebooks/metrics/knn.csv",
    "ARIMA": "D:/SAMLFRFM/notebooks/metrics/arima.csv",
    "LSTM": "D:/SAMLFRFM/notebooks/metrics/lstm.csv",
    "GRU": "D:/SAMLFRFM/notebooks/metrics/gru.csv",
    "XGBoost": "D:/SAMLFRFM/notebooks/metrics/xgboost.csv",
}

# Iterate over all files with friendly names
for model_name, file_path in files_and_names.items():
    st.write(f"## Model: {model_name}")  # Display name

    try:
        # Load data
        data = pd.read_csv(file_path)

        # Identify numerical columns
        numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

        if len(numerical_columns) > 0:
            # Prepare Shapiro-Wilk test results
            results = []
            for col in numerical_columns:
                stat, p_value = shapiro(data[col].dropna())
                results.append({
                    "Metric": col,
                    "Statistic": f"{stat:.4f}",
                    "p-value": f"{p_value:.4f}",
                    "Normality": "Fail to Reject" if p_value > 0.05 else "Reject"
                })
            
            # Convert results to a DataFrame for table display
            results_df = pd.DataFrame(results)
            st.table(results_df)  # Display results in table format
        else:
            st.write("No numerical columns found in the dataset.")
    except Exception as e:
        st.write(f"Error processing file for {model_name}: {e}")
    st.write("===================================================================================")
