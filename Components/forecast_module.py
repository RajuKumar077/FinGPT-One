# Components/forecast_module.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.style.use('dark_background')
plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})

# ----------------- LSTM Model Class -----------------
class LSTMRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ----------------- Forecasting Interface -----------------
def display_forecasting(hist_data, ticker):
    if hist_data is None or hist_data.empty:
        st.error(f"❌ No historical data available for {ticker}.")
        return

    st.markdown(f"<h3>📈 Forecasting for {ticker}</h3>", unsafe_allow_html=True)

    # Prepare Prophet data
    df_prophet = hist_data.reset_index()[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    days_to_forecast = st.slider("Days to forecast", min_value=7, max_value=365, value=30, step=7, key=f"forecast_{ticker}")

    # ----------------- ARIMA Forecast -----------------
    try:
        arima_model = ARIMA(df_prophet['y'], order=(5,1,0)).fit()
        arima_forecast = arima_model.forecast(steps=days_to_forecast)
        arima_index = pd.date_range(df_prophet['ds'].iloc[-1]+pd.Timedelta(days=1), periods=days_to_forecast)
        arima_series = pd.Series(arima_forecast, index=arima_index, name='ARIMA Forecast')
    except:
        arima_series = None

    # ----------------- Prophet Forecast -----------------
    try:
        prophet_model = Prophet(daily_seasonality=True)
        prophet_model.fit(df_prophet)
        future = prophet_model.make_future_dataframe(periods=days_to_forecast, include_history=False)
        prophet_forecast = prophet_model.predict(future).set_index('ds')['yhat']
    except:
        prophet_forecast = None

    # ----------------- XGBoost Forecast -----------------
    df_xgb = df_prophet.copy()
    for i in range(1,11):
        df_xgb[f'lag_{i}'] = df_xgb['y'].shift(i)
    df_xgb['rolling_mean_5'] = df_xgb['y'].rolling(5).mean()
    df_xgb['rolling_std_5'] = df_xgb['y'].rolling(5).std()
    df_xgb.dropna(inplace=True)
    features = [c for c in df_xgb.columns if c not in ['y','ds']]
    xgb_series = None
    if not df_xgb.empty and len(df_xgb) > 10:
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05)
        xgb_model.fit(df_xgb[features], df_xgb['y'])
        forecast_vals = []
        temp_df = df_xgb.copy()
        last_date = temp_df.index[-1]
        for _ in range(days_to_forecast):
            temp_df_row = temp_df.iloc[-1:].copy()
            for i in range(1,11):
                temp_df_row[f'lag_{i}'] = temp_df['y'].iloc[-i]
            temp_df_row['rolling_mean_5'] = temp_df['y'].rolling(5).mean().iloc[-1]
            temp_df_row['rolling_std_5'] = temp_df['y'].rolling(5).std().iloc[-1]
            pred = xgb_model.predict(temp_df_row[features])[0]
            forecast_vals.append(pred)
            temp_df = temp_df.append({'y':pred}, ignore_index=True)
            last_date += pd.Timedelta(days=1)
        xgb_series = pd.Series(forecast_vals, index=pd.date_range(df_prophet['ds'].iloc[-1]+pd.Timedelta(days=1), periods=len(forecast_vals)), name='XGBoost Forecast')

    # ----------------- Plot -----------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name='Historical', line=dict(color='lightgray')))
    if arima_series is not None:
        fig.add_trace(go.Scatter(x=arima_series.index, y=arima_series.values, name='ARIMA Forecast', line=dict(color='red', dash='dash')))
    if prophet_forecast is not None:
        fig.add_trace(go.Scatter(x=prophet_forecast.index, y=prophet_forecast.values, name='Prophet Forecast', line=dict(color='blue', dash='dot')))
    if xgb_series is not None:
        fig.add_trace(go.Scatter(x=xgb_series.index, y=xgb_series.values, name='XGBoost Forecast', line=dict(color='green', dash='dashdot')))

    fig.update_layout(title=f"{ticker} Forecasts", xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
