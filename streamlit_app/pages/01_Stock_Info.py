# Import streamlit
import streamlit as st

# Import functions
from helper import *
from randomforest import update_to_csv

# Configure the page
st.set_page_config(
    page_title="Stock Info",
)

#Sidebar Start

# Add a sidebar
# Fetch and store the stock data
stock_dict = fetch_stocks()

# Add a dropdown for selecting the stock
st.sidebar.markdown("## **Select stock**")
stock = st.sidebar.selectbox("Choose a stock", list(stock_dict.keys()))

#Sidebar End

# Fetch the info of the stock
try:
    stock_data_info = fetch_stock_info(stock)
except:
    st.error("Error: Unable to fetch the stock data. Please try again later.")
    st.stop()

#Title

# Add title to the app
st.markdown("# **Stock Info**")

#Title End

#Basic Information

# Add heading
st.markdown("## **Basic Information**")

# Create 2 columns
col1, col2 = st.columns(2)

# Row 1
col1.dataframe(
    pd.DataFrame({"Issuer Name": [stock_data_info["Basic Information"]["longName"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame({"Symbol": [stock]}),
    hide_index=True,
    width=500,
)

# Row 2
col1.dataframe(
    pd.DataFrame({"Currency": [stock_data_info["Basic Information"]["currency"]]}),
    hide_index=True,
    width=500,
)

#Basic Information End

#Market Data

# Add a heading
st.markdown("## **Market Data**")

# Create 2 columns
col1, col2 = st.columns(2)

# Row 1
col1.dataframe(
    pd.DataFrame({"Current Price": [stock_data_info["Market Data"]["currentPrice"]]}),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame({"Previous Close": [stock_data_info["Market Data"]["previousClose"]]}),
    hide_index=True,
    width=500,
)

# Create 3 columns
col1, col2, col3 = st.columns(3)

# Row 1
col1.dataframe(
    pd.DataFrame({"Open": [stock_data_info["Market Data"]["open"]]}),
    hide_index=True,
    width=300,
)
col2.dataframe(
    pd.DataFrame({"Day Low": [stock_data_info["Market Data"]["dayLow"]]}),
    hide_index=True,
    width=300,
)
col3.dataframe(
    pd.DataFrame({"Open": [stock_data_info["Market Data"]["dayHigh"]]}),
    hide_index=True,
    width=300,
)

# Create 2 columns
col1, col2 = st.columns(2)

# Row 1
col1.dataframe(
    pd.DataFrame(
        {
            "Regular Market Previous Close": [
                stock_data_info["Market Data"]["regularMarketPreviousClose"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {"Regular Market Open": [stock_data_info["Market Data"]["regularMarketOpen"]]}
    ),
    hide_index=True,
    width=500,
)

# Row 2
col1.dataframe(
    pd.DataFrame(
        {
            "Regular Market Day Low": [
                stock_data_info["Market Data"]["regularMarketDayLow"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)
col2.dataframe(
    pd.DataFrame(
        {
            "Regular Market Day High": [
                stock_data_info["Market Data"]["regularMarketDayHigh"]
            ]
        }
    ),
    hide_index=True,
    width=500,
)

# Create 3 columns
col1, col2, col3 = st.columns(3)

# Row 1
col1.dataframe(
    pd.DataFrame(
        {"Fifty-Two Week Low": [stock_data_info["Market Data"]["fiftyTwoWeekLow"]]}
    ),
    hide_index=True,
    width=300,
)
col2.dataframe(
    pd.DataFrame(
        {"Fifty-Two Week High": [stock_data_info["Market Data"]["fiftyTwoWeekHigh"]]}
    ),
    hide_index=True,
    width=300,
)
col3.dataframe(
    pd.DataFrame(
        {"Fifty-Day Average": [stock_data_info["Market Data"]["fiftyDayAverage"]]}
    ),
    hide_index=True,
    width=300,
)

#Market Data End