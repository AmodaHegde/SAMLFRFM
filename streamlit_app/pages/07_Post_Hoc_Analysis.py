import pandas as pd
import streamlit as st
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt

# Predefined file paths and model names
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

# Streamlit App Title
st.title("Tukey's HSD Post-Hoc Analysis for Machine Learning Models")

# Load Data from Files
st.sidebar.header("Loaded Models and Metrics")
model_data = {}
for model_name, file_path in files_and_names.items():
    try:
        model_data[model_name] = pd.read_csv(file_path)
        st.sidebar.success(f"{model_name} loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Failed to load {model_name}: {e}")

# Ensure at least 2 models are loaded
if len(model_data) < 2:
    st.warning("At least two models must be successfully loaded to perform Tukey's HSD.")
else:
    # Select a Metric
    metrics = model_data[list(model_data.keys())[0]].columns[1:]  # Assume metrics start from 2nd column
    selected_metric = st.sidebar.selectbox("Select Metric for Tukey's HSD Test", metrics)

    # Combine metric data across models
    combined_data = pd.concat(
        [
            pd.DataFrame({
                "Value": data[selected_metric],
                "Model": model_name
            })
            for model_name, data in model_data.items()
        ]
    )

    # Perform Tukey's HSD
    st.header(f"Tukey's HSD Results for Metric: {selected_metric}")
    try:
        tukey_result = pairwise_tukeyhsd(
            endog=combined_data["Value"],
            groups=combined_data["Model"],
            alpha=0.05
        )

        # Display Tukey's HSD Summary
        #st.write(tukey_result.summary())

        # Visualization: Mean Differences
        st.header("Tukey's HSD Plot")
        plt.figure(figsize=(12, 8))
        tukey_result.plot_simultaneous(comparison_name=combined_data["Model"].iloc[0], xlabel=selected_metric)
        plt.title(f"Tukey's HSD Mean Differences for {selected_metric}")
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error during Tukey's HSD Test: {e}")

    # Additional Visualization: Boxplot
    st.header("Metric Distribution Across Models")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_data, x="Model", y="Value")
    plt.title(f"Distribution of {selected_metric} Across Models")
    plt.xlabel("Model")
    plt.ylabel(selected_metric)
    plt.xticks(rotation=45)
    st.pyplot(plt)
