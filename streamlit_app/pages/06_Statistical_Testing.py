import pandas as pd
from scipy.stats import f_oneway
import streamlit as st

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
st.title("ANOVA Test for Machine Learning Models")

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
    st.warning("At least two models must be successfully loaded to perform ANOVA.")
else:
    # Select a Metric
    metrics = model_data[list(model_data.keys())[0]].columns[1:]  # Assume metrics start from 2nd column
    selected_metric = st.sidebar.selectbox("Select Metric for ANOVA Test", metrics)

    # Extract metric data for each model
    metric_groups = []
    for model_name, data in model_data.items():
        metric_groups.append(data[selected_metric])

    # Perform ANOVA test
    anova_statistic, anova_p_value = f_oneway(*metric_groups)

    # Display Results
    st.header(f"ANOVA Results for Metric: {selected_metric}")
    st.write(f"**ANOVA Statistic:** {anova_statistic:.4f}")
    st.write(f"**p-value:** {anova_p_value:.4f}")

    # Interpretation
    if anova_p_value > 0.05:
        st.success("No significant difference between the models for this metric (p > 0.05).")
    else:
        st.error("Significant difference exists between the models for this metric (p ≤ 0.05).")

    # Display Mean Values for Each Model
    st.header("Mean Values for Each Model")
    mean_values = {model_name: data[selected_metric].mean() for model_name, data in model_data.items()}
    mean_values_df = pd.DataFrame(list(mean_values.items()), columns=["Model", "Mean Value"]).sort_values(by="Mean Value")
    st.write(mean_values_df)

    # Visualization
    import seaborn as sns
import matplotlib.pyplot as plt

# Visualization: Metric Distribution Across Models
st.header("Metric Distribution Across Models")

# Prepare data for visualization
all_metric_data = pd.concat(
    {model: data[selected_metric] for model, data in model_data.items()},
    axis=0
).reset_index()
all_metric_data.columns = ['Model', 'Index', selected_metric]
all_metric_data = all_metric_data.drop(columns='Index')  # Drop unnecessary index column

# Create boxplot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(data=all_metric_data, x='Model', y=selected_metric)
plt.title(f"Distribution of {selected_metric} Across Models")
plt.xlabel("Model")
plt.ylabel(selected_metric)

# Display the plot in Streamlit
st.pyplot(plt)

