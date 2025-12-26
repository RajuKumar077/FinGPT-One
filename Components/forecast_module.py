import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

# --- Data Fetching ---
@st.cache_data(ttl=3600, show_spinner="Fetching stock data...")
def fetch_stock_data(ticker: str, period="2y") -> pd.DataFrame:
    try:
        data = yf.download(ticker, period=period)
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# --- Metrics ---
@st.cache_data(ttl=3600)
def calculate_metrics(y_true, y_pred):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).dropna()
    if df.empty:
        return float('inf'), float('inf'), float('inf')
    rmse = np.sqrt(mean_squared_error(df['y_true'], df['y_pred']))
    mae = mean_absolute_error(df['y_true'], df['y_pred'])
    mape = np.mean(np.abs((df['y_true'] - df['y_pred']) / df['y_true'].replace(0, np.nan))) * 100
    return rmse, mae, mape

# --- Forecasting Display ---
def display_forecasting(hist_data, ticker):
    st.subheader(f"📊 Forecasting for {ticker}")
    st.info("Currently using ARIMA + Prophet. LSTM will be added soon.")
    
    # ARIMA example
    series = hist_data['Close']
    from statsmodels.tsa.arima.model import ARIMA
    try:
        model = ARIMA(series, order=(5,1,0)).fit()
        forecast = model.forecast(30)
        forecast.index = pd.date_range(series.index[-1]+pd.Timedelta(days=1), periods=30)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Historical", line=dict(color='lightgray')))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="ARIMA Forecast", line=dict(color='red', dash='dash')))
        fig.update_layout(title=f"ARIMA Forecast for {ticker}", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"ARIMA forecast failed: {e}")
