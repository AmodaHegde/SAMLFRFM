#imports

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helper import *

st.set_page_config(
    page_title="Model Comparison",
)

df = pd.read_csv("D:/SAMLFRFM/notebooks/model_metrics.csv")

st.markdown("# **Error metrics of models**")

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
