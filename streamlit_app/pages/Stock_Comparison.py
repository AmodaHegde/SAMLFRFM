import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from helper import *

st.set_page_config(
    page_title="Model Comparison",
)

st.markdown("# **Error metrics of models**")

df = pd.read_csv("D:/SAMLFRFM/notebooks/model_metrics.csv")

st.markdown("## **Random Forest**")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("MAE", round(df[df["Model"] == "Random Forest"]["MAE"].values[0], 4))
col2.metric("MAPE", round(df[df["Model"] == "Random Forest"]["MAPE"].values[0], 4))
col3.metric("MSE", round(df[df["Model"] == "Random Forest"]["MSE"].values[0], 4))
col4.metric("RMSE", round(df[df["Model"] == "Random Forest"]["RMSE"].values[0], 4))
col5.metric("R²", round(df[df["Model"] == "Random Forest"]["R2"].values[0], 4))

st.markdown("## **Decision Tree**")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("MAE", round(df[df["Model"] == "Decision Tree"]["MAE"].values[0], 4))
col2.metric("MAPE", round(df[df["Model"] == "Decision Tree"]["MAPE"].values[0], 4))
col3.metric("MSE", round(df[df["Model"] == "Decision Tree"]["MSE"].values[0], 4))
col4.metric("RMSE", round(df[df["Model"] == "Decision Tree"]["RMSE"].values[0], 4))
col5.metric("R²", round(df[df["Model"] == "Decision Tree"]["R2"].values[0], 4))

st.markdown("## **KNN**")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("MAE", round(df[df["Model"] == "KNN"]["MAE"].values[0], 4))
col2.metric("MAPE", round(df[df["Model"] == "KNN"]["MAPE"].values[0], 4))
col3.metric("MSE", round(df[df["Model"] == "KNN"]["MSE"].values[0], 4))
col4.metric("RMSE", round(df[df["Model"] == "KNN"]["RMSE"].values[0], 4))
col5.metric("R²", round(df[df["Model"] == "KNN"]["R2"].values[0], 4))

st.markdown("## **LSTM**")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("MAE", round(df[df["Model"] == "LSTM"]["MAE"].values[0], 4))
col2.metric("MAPE", round(df[df["Model"] == "LSTM"]["MAPE"].values[0], 4))
col3.metric("MSE", round(df[df["Model"] == "LSTM"]["MSE"].values[0], 4))
col4.metric("RMSE", round(df[df["Model"] == "LSTM"]["RMSE"].values[0], 4))
col5.metric("R²", round(df[df["Model"] == "LSTM"]["R2"].values[0], 4))

st.markdown("## **GRU**")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("MAE", round(df[df["Model"] == "GRU"]["MAE"].values[0], 4))
col2.metric("MAPE", round(df[df["Model"] == "GRU"]["MAPE"].values[0], 4))
col3.metric("MSE", round(df[df["Model"] == "GRU"]["MSE"].values[0], 4))
col4.metric("RMSE", round(df[df["Model"] == "GRU"]["RMSE"].values[0], 4))
col5.metric("R²", round(df[df["Model"] == "GRU"]["R2"].values[0], 4))

st.markdown("## **SVM-SVR**")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("MAE", round(df[df["Model"] == "SVM-SVR"]["MAE"].values[0], 4))
col2.metric("MAPE", round(df[df["Model"] == "SVM-SVR"]["MAPE"].values[0], 4))
col3.metric("MSE", round(df[df["Model"] == "SVM-SVR"]["MSE"].values[0], 4))
col4.metric("RMSE", round(df[df["Model"] == "SVM-SVR"]["RMSE"].values[0], 4))
col5.metric("R²", round(df[df["Model"] == "SVM-SVR"]["R2"].values[0], 4))

st.markdown("## **ARIMA**")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("MAE", round(df[df["Model"] == "ARIMA"]["MAE"].values[0], 4))
col2.metric("MAPE", round(df[df["Model"] == "ARIMA"]["MAPE"].values[0], 4))
col3.metric("MSE", round(df[df["Model"] == "ARIMA"]["MSE"].values[0], 4))
col4.metric("RMSE", round(df[df["Model"] == "ARIMA"]["RMSE"].values[0], 4))
col5.metric("R²", round(df[df["Model"] == "ARIMA"]["R2"].values[0], 4))

st.markdown("## **XG Boost**")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("MAE", round(df[df["Model"] == "XG Boost"]["MAE"].values[0], 4))
col2.metric("MAPE", round(df[df["Model"] == "XG Boost"]["MAPE"].values[0], 4))
col3.metric("MSE", round(df[df["Model"] == "XG Boost"]["MSE"].values[0], 4))
col4.metric("RMSE", round(df[df["Model"] == "XG Boost"]["RMSE"].values[0], 4))
col5.metric("R²", round(df[df["Model"] == "XG Boost"]["R2"].values[0], 4))

st.markdown("# **Error metrics of models**")

# fig, ax = plt.subplots(figsize=(10, 6))

# rf = df[df['Model']=="Random Forest"]

# rf.boxplot()
# plt.title('Distribution of RMSE for Different Models', fontsize=14)
# plt.xlabel('Models', fontsize=12)
# plt.ylabel('RMSE', fontsize=12)
# plt.grid(False)
# st.pyplot(fig)

fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(x='Model', y='RMSE', data=df, palette='viridis', ax=ax1)

# Customize the plot
ax1.set_title('Model RMSE Comparison', fontsize=16)
ax1.set_xlabel('Models', fontsize=14)
ax1.set_ylabel('RMSE Values', fontsize=14)
ax1.tick_params(axis='x', rotation=45, labelsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Render the plot in Streamlit
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(x='Model', y='MAE', data=df, palette='viridis', ax=ax2)

# Customize the plot
ax2.set_title('Model MAE Comparison', fontsize=16)
ax2.set_xlabel('Models', fontsize=14)
ax2.set_ylabel('MAE Values', fontsize=14)
ax2.tick_params(axis='x', rotation=45, labelsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Render the plot in Streamlit
st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.barplot(x='Model', y='R2', data=df, palette='viridis', ax=ax3)

# Customize the plot
ax3.set_title('Model R2 Comparison', fontsize=16)
ax3.set_xlabel('Models', fontsize=14)
ax3.set_ylabel('R2 Values', fontsize=14)
ax3.tick_params(axis='x', rotation=45, labelsize=12)
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# Render the plot in Streamlit
st.pyplot(fig3)

fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.barplot(x='Model', y='MAPE', data=df, palette='viridis', ax=ax4)

# Customize the plot
ax4.set_title('Model MAPE Comparison', fontsize=16)
ax4.set_xlabel('Models', fontsize=14)
ax4.set_ylabel('MAPE Values', fontsize=14)
ax4.tick_params(axis='x', rotation=45, labelsize=12)
ax4.grid(axis='y', linestyle='--', alpha=0.7)

# Render the plot in Streamlit
st.pyplot(fig4)

fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.barplot(x='Model', y='MSE', data=df, palette='viridis', ax=ax5)

# Customize the plot
ax5.set_title('Model MSE Comparison', fontsize=16)
ax5.set_xlabel('Models', fontsize=14)
ax5.set_ylabel('MSE Values', fontsize=14)
ax5.tick_params(axis='x', rotation=45, labelsize=12)
ax5.grid(axis='y', linestyle='--', alpha=0.7)

# Render the plot in Streamlit
st.pyplot(fig5)
