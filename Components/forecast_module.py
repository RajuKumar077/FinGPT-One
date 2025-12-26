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
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
plt.style.use('dark_background')
plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})

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

# --- Data Preparation ---
@st.cache_data(ttl=3600)
def prepare_data_for_forecasting(df):
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    df.index = pd.to_datetime(df.index)
    df_prophet = df[['Close']].rename(columns={'Close': 'y'})
    df_prophet['ds'] = df_prophet.index
    return df_prophet

# --- ARIMA ---
@st.cache_resource
def train_and_forecast_arima(series, days=30, order=(5,1,0)):
    try:
        model = ARIMA(series, order=order).fit()
        future = model.forecast(steps=days)
        future_index = pd.date_range(series.index[-1]+pd.Timedelta(days=1), periods=days)
        return model, pd.Series(future, index=future_index, name='yhat')
    except:
        return None, None

# --- Prophet ---
@st.cache_resource
def train_and_forecast_prophet(df, days=30):
    try:
        model = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=0.05,
                        yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(df)
        future = model.make_future_dataframe(periods=days, include_history=False)
        forecast = model.predict(future)
        return model, forecast.set_index('ds')['yhat']
    except:
        return None, None

# --- XGBoost ---
@st.cache_resource
def train_xgboost(df):
    df = df.copy()
    for i in range(1, 11):
        df[f'lag_{i}'] = df['y'].shift(i)
    df['rolling_mean_5'] = df['y'].rolling(5).mean()
    df['rolling_std_5'] = df['y'].rolling(5).std()
    df.dropna(inplace=True)
    features = [c for c in df.columns if c not in ['y','ds']]
    if df.empty or len(df) < 10:
        return None, None
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, n_jobs=-1)
    model.fit(df[features], df['y'])
    return model, features

def forecast_xgboost(model, features, df, days=30):
    if model is None or features is None or df.empty:
        return None
    forecast_df = df.copy()
    predictions = []
    last_date = forecast_df.index[-1]
    for _ in range(days):
        new_row = pd.DataFrame({'y':[np.nan]}, index=[last_date+pd.Timedelta(days=1)])
        forecast_df = pd.concat([forecast_df, new_row])
        for i in range(1, 11):
            forecast_df[f'lag_{i}'] = forecast_df['y'].shift(i)
        forecast_df['rolling_mean_5'] = forecast_df['y'].rolling(5).mean()
        forecast_df['rolling_std_5'] = forecast_df['y'].rolling(5).std()
        X_pred = forecast_df[features].iloc[[-1]].dropna()
        if X_pred.empty:
            break
        pred = model.predict(X_pred)[0]
        predictions.append(pred)
        forecast_df.iloc[-1, forecast_df.columns.get_loc('y')] = pred
        last_date += pd.Timedelta(days=1)
    return pd.Series(predictions, index=pd.date_range(df.index[-1]+pd.Timedelta(days=1), periods=len(predictions)), name='yhat')

# --- LSTM ---
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
        out,_ = self.lstm(x, (h0,c0))
        out = self.fc(out[:,-1,:])
        return out

@st.cache_resource
def prepare_lstm_data(df, seq_len=20):
    data = df['y'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled)-seq_len):
        X.append(scaled[i:i+seq_len,0])
        y.append(scaled[i+seq_len,0])
    return np.array(X), np.array(y), scaler

@st.cache_resource
def train_lstm(X, y, seq_len=20, hidden_size=50, num_layers=2, epochs=50):
    if X.shape[0]==0:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = LSTMRegressionModel(1, hidden_size, num_layers, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model

def forecast_lstm(model, scaler, df, days=30, seq_len=20):
    if model is None or df.empty:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = df['y'].values.reshape(-1,1)
    scaled = scaler.transform(data)
    seq = list(scaled[-seq_len:].flatten())
    preds = []
    last_date = df.index[-1]
    model.eval()
    with torch.no_grad():
        for _ in range(days):
            input_tensor = torch.tensor(seq[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            pred_scaled = model(input_tensor).cpu().numpy()[0][0]
            preds.append(pred_scaled)
            seq.append(pred_scaled)
            last_date += pd.Timedelta(days=1)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    return pd.Series(preds, index=pd.date_range(df.index[-1]+pd.Timedelta(days=1), periods=len(preds)), name='yhat')

# --- Plotting ---
def plot_forecast(hist, forecast, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name='Historical', line=dict(color='lightgray')))
    if forecast is not None:
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name=f'{model_name} Forecast', line=dict(color='red', dash='dash')))
    fig.update_layout(title=f'{model_name} Forecast', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
    return fig
