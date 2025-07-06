import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import ta  # For technical analysis indicators
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb

# Suppress Prophet's verbose logging
import logging

logging.getLogger('prophet').setLevel(logging.WARNING)

# Conditional import for Prophet
PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    st.warning("Prophet library not found. Please install it (`pip install prophet`) to use Prophet forecasting.")

# Conditional imports for LSTM and SHAP
LSTM_AVAILABLE = False  # Initialize to False
try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn # Import nn here, only if LSTM_AVAILABLE is True
    LSTM_AVAILABLE = True
except ImportError as e:
    st.warning(
        f"PyTorch library not found or failed to load: {e}. Please install it (`pip install torch`) to use LSTM forecasting.")
    LSTM_AVAILABLE = False
except Exception as e:  # Catch any other unexpected errors during import
    st.warning(f"An unexpected error occurred during PyTorch import: {e}. Please check your installation.")
    LSTM_AVAILABLE = False

SHAP_AVAILABLE = False
try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    st.warning("SHAP library not found. Please install it (`pip install shap matplotlib`) for model explainability.")


# --- Helper functions for feature engineering ---
@st.cache_data(ttl=3600, show_spinner=False)
def compute_rsi(series, period=14):
    """Computes the Relative Strength Index (RSI) using 'ta' library."""
    return ta.momentum.RSIIndicator(series, window=period).rsi()


@st.cache_data(ttl=3600, show_spinner=False)
def prepare_forecast_data(hist_data, ticker):
    """
    Prepares historical data for forecasting by computing features.
    Ensures data is ready for Linear Regression, Prophet, and LSTM.
    """
    if hist_data.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "No historical data provided."

    df = hist_data.copy()
    required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Missing required columns: {', '.join(missing_cols)}."

    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().all():
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"All '{col}' values are invalid or missing."

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "Could not convert index to datetime format."

    df = df.sort_index()

    df['Date_Ordinal'] = df.index.map(lambda x: x.toordinal())
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag5'] = df['Close'].shift(5)

    df['Return_1d'] = df['Close'].pct_change()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['High_Low_Diff'] = (df['High'] - df['Low']) / df['Close']
    df['Open_Close_Diff'] = (df['Close'] - df['Open']) / df['Open']

    df.dropna(inplace=True)

    min_data_points = 60
    if df.empty or len(df) < min_data_points:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Not enough data after feature engineering (minimum {min_data_points} data points required)."

    df_prophet = df[['Close']].copy()
    df_prophet.index.name = 'ds'
    df_prophet = df_prophet.reset_index()
    df_prophet = df_prophet.rename(columns={'Close': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    df_lstm = df.copy()
    df_xgb = df.copy()

    return df, df_prophet, df_lstm, df_xgb, None


# --- Deep Learning Model (LSTM) ---
if LSTM_AVAILABLE:
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out


    @st.cache_resource(show_spinner=False)
    def prepare_lstm_data_for_forecast(df, features, sequence_length):
        """Prepares data for LSTM model with sequence input for regression."""
        if df.empty or len(df) < sequence_length:
            return np.array([]), np.array([]), None, "Not enough data for LSTM sequences."

        data = df[features].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i + sequence_length])
            y.append(df['Close'].iloc[i + sequence_length])

        return np.array(X), np.array(y), scaler, None


    @st.cache_resource(show_spinner=False)
    def train_lstm_model_for_forecast(X_train, y_train, X_test, y_test, input_size, hidden_size=50, num_layers=2,
                                     epochs=50, batch_size=32):
        """Trains an LSTM model for regression forecasting."""
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            return None, np.array([]), np.array([]), "Insufficient data for LSTM training/testing split."

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = LSTMModel(input_size, hidden_size, num_layers, 1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_tensor).numpy().flatten()

        return model, test_preds, y_test.flatten(), None


# --- Traditional ML Models (Linear Regression, ARIMA, XGBoost) ---
@st.cache_data(ttl=3600, show_spinner=False)
def run_linear_regression_forecast(df, forecast_days):
    """
    Trains and forecasts using Linear Regression with time series cross-validation.
    """
    features = [
        'Date_Ordinal', 'MA5', 'MA10', 'MA20', 'MA50',
        'Close_Lag1', 'Close_Lag5', 'Return_1d', 'Volatility', 'RSI',
        'Volume_MA5', 'High_Low_Diff', 'Open_Close_Diff'
    ]

    available_features = [f for f in features if f in df.columns and not df[f].isnull().all()]
    if not available_features:
        return None, "No valid features for Linear Regression.", None, None, None, None, np.nan, np.nan, np.nan

    X = df[available_features]
    y = df['Close']

    if y.nunique() <= 1:
        return None, "All closing prices are identical for Linear Regression.", None, None, None, None, np.nan, np.nan, np.nan

    tscv = TimeSeriesSplit(n_splits=5)

    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    y_preds_test_cv = []
    y_actuals_test_cv = []
    test_dates_cv = []

    cv_rmse_scores = []
    cv_mae_scores = []
    cv_mape_scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        if len(X_train) == 0 or len(X_test_fold) == 0:
            continue

        model_pipeline.fit(X_train, y_train)
        y_pred_fold = model_pipeline.predict(X_test_fold)

        y_preds_test_cv.extend(y_pred_fold)
        y_actuals_test_cv.extend(y_test_fold)
        test_dates_cv.extend(df.index[test_index])

        rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred_fold))
        mae = mean_absolute_error(y_test_fold, y_pred_fold)
        mape = np.mean(np.abs((y_test_fold - y_pred_fold) / y_test_fold)) * 100 if np.all(y_test_fold != 0) else np.nan

        cv_rmse_scores.append(rmse)
        cv_mae_scores.append(mae)
        if not np.isnan(mape):
            cv_mape_scores.append(mape)

    if not cv_rmse_scores:
        return None, "Not enough data for time series cross-validation.", None, None, None, None, np.nan, np.nan, np.nan

    avg_rmse = np.mean(cv_rmse_scores)
    avg_mae = np.mean(cv_mae_scores)
    avg_mape = np.mean(cv_mape_scores) if cv_mape_scores else np.nan

    # Retrain on full data for future prediction
    model_pipeline.fit(X, y)

    last_date_ordinal = df['Date_Ordinal'].iloc[-1]

    # Dynamically extrapolate future features
    future_features_data = {}
    for feature in available_features:
        if feature == 'Date_Ordinal':
            future_features_data['Date_Ordinal'] = [last_date_ordinal + i for i in range(1, forecast_days + 1)]
            continue

        # Extrapolate other features using a simple linear trend from recent data
        recent_data_for_extrapolation = df[feature].tail(min(len(df[feature]), 20)) # Use up to 20 recent points
        if len(recent_data_for_extrapolation) >= 2 and feature != 'Close_Lag1' and feature != 'Close_Lag5': # Lag features will be updated iteratively
            ma_X = np.arange(len(recent_data_for_extrapolation)).reshape(-1, 1)
            ma_y = recent_data_for_extrapolation.values
            try:
                ma_model = LinearRegression().fit(ma_X, ma_y)
                future_steps = np.arange(len(recent_data_for_extrapolation), len(recent_data_for_extrapolation) + forecast_days).reshape(-1, 1)
                future_features_data[feature] = ma_model.predict(future_steps)
                future_features_data[feature] = np.maximum(future_features_data[feature], 0) # Prices/volumes can't be negative
            except Exception: # Fallback if linear regression fails
                future_features_data[feature] = [df[feature].iloc[-1]] * forecast_days
        else:
            future_features_data[feature] = [df[feature].iloc[-1]] * forecast_days # Use last known value

    # Initialize future predictions and iteratively update lag features
    future_predictions_list = []
    current_close_for_lag1 = df['Close'].iloc[-1]
    current_close_for_lag5 = df['Close'].iloc[-5] if len(df) >= 5 else df['Close'].iloc[-1] # Fallback if not enough history

    for i in range(forecast_days):
        # Prepare the row for the next prediction
        next_day_features = {f: future_features_data[f][i] for f in available_features if f not in ['Close_Lag1', 'Close_Lag5']}
        next_day_features['Close_Lag1'] = current_close_for_lag1
        next_day_features['Close_Lag5'] = current_close_for_lag5 # This one won't update for first 4 days

        # Ensure order matches training features
        next_day_X = pd.DataFrame([next_day_features], columns=available_features)
        
        predicted_close = model_pipeline.predict(next_day_X)[0]
        predicted_close = max(predicted_close, 0) # Ensure no negative predictions

        future_predictions_list.append(predicted_close)

        # Update lag features for the *next* iteration
        current_close_for_lag5 = current_close_for_lag1 # Shift lag1 to lag5
        current_close_for_lag1 = predicted_close # New prediction becomes lag1 for next day

    forecast_df = pd.DataFrame({
        'Date': [pd.Timestamp.fromordinal(int(date)).date() for date in future_features_data['Date_Ordinal']],
        'Predicted Close': np.array(future_predictions_list).round(2)
    })

    return model_pipeline, forecast_df, test_dates_cv, y_preds_test_cv, y_actuals_test_cv, None, avg_rmse, avg_mae, avg_mape

@st.cache_data(ttl=3600, show_spinner=False)
def run_arima_forecast(df, forecast_days, p, d, q):
    """
    Trains and forecasts using ARIMA model.
    """
    y = df['Close']

    if len(y) < max(p, d, q) + 10: # Ensure enough data for ARIMA order
        return None, f"Not enough data for ARIMA({p},{d},{q}). Need at least {max(p, d, q) + 10} data points.", None, None, None, np.nan, np.nan, np.nan

    # Splitting data for in-sample evaluation
    train_size = int(len(y) * 0.8)
    train_data, test_data = y[0:train_size], y[train_size:]

    try:
        # Fit ARIMA model on training data
        model = ARIMA(train_data, order=(p, d, q))
        model_fit = model.fit()

        # In-sample prediction and evaluation
        start_index = len(train_data)
        end_index = len(y) - 1 # Corresponds to the last point of the original data

        # Ensure prediction range is within bounds for in-sample
        if start_index >= end_index: # Handle cases where test set is too small or non-existent
             y_pred_hist = pd.Series([])
             y_true_hist = pd.Series([])
        else:
            y_pred_hist = model_fit.predict(start=start_index, end=end_index, typ='levels')
            y_true_hist = test_data

        rmse, mae, mape = np.nan, np.nan, np.nan

        if len(y_true_hist) > 0:
            valid_indices = y_true_hist.notna() & y_pred_hist.notna()
            y_true_hist_filtered = y_true_hist[valid_indices]
            y_pred_hist_filtered = y_pred_hist[valid_indices]

            if len(y_true_hist_filtered) > 0:
                rmse = np.sqrt(mean_squared_error(y_true_hist_filtered, y_pred_hist_filtered))
                mae = mean_absolute_error(y_true_hist_filtered, y_pred_hist_filtered)
                mape = np.mean(np.abs((y_true_hist_filtered - y_pred_hist_filtered) / y_true_hist_filtered)) * 100 if np.all(y_true_hist_filtered != 0) else np.nan
            else:
                st.info("No valid data points for ARIMA in-sample performance evaluation after filtering.")
        else:
            st.info("Not enough historical data for in-sample ARIMA performance evaluation.")

        st.write(f"**ARIMA In-Sample Performance:**")
        st.write(f"  - **RMSE:** `{rmse:.2f}`")
        st.write(f"  - **MAE:** `{mae:.2f}`")
        if not np.isnan(mape):
            st.write(f"  - **MAPE:** `{mape:.2f}%`")
        else:
            st.write("  - **MAPE:** `N/A (due to zero actual values or no valid predictions)`")

        # Forecast future values from the full dataset
        full_model = ARIMA(y, order=(p, d, q))
        full_model_fit = full_model.fit()
        
        forecast_results = full_model_fit.forecast(steps=forecast_days)
        
        forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B') # Business days
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted Close': forecast_results.values.round(2)
        })

        # Capture actuals and predictions for plotting
        test_dates_plot = test_data.index
        test_preds_plot = y_pred_hist.values if len(y_pred_hist) > 0 else np.array([])
        test_actuals_plot = test_data.values
        
        return full_model, forecast_df, test_dates_plot, test_preds_plot, test_actuals_plot, rmse, mae, mape

    except Exception as e:
        return None, f"ARIMA model training or forecasting failed: {e}", None, None, None, np.nan, np.nan, np.nan

@st.cache_data(ttl=3600, show_spinner=False)
def run_xgboost_forecast(df_xgb, forecast_days):
    """
    Trains and forecasts using XGBoost.
    """
    features = [
        'Date_Ordinal', 'MA5', 'MA10', 'MA20', 'MA50',
        'Close_Lag1', 'Close_Lag5', 'Return_1d', 'Volatility', 'RSI',
        'Volume_MA5', 'High_Low_Diff', 'Open_Close_Diff'
    ]

    available_features = [f for f in features if f in df_xgb.columns and not df_xgb[f].isnull().all()]
    if not available_features:
        return None, "No valid features for XGBoost.", None, None, None, None, np.nan, np.nan, np.nan

    X = df_xgb[available_features]
    y = df_xgb['Close']

    if y.nunique() <= 1:
        return None, "All closing prices are identical for XGBoost.", None, None, None, None, np.nan, np.nan, np.nan

    tscv = TimeSeriesSplit(n_splits=5)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    y_preds_test_cv = []
    y_actuals_test_cv = []
    test_dates_cv = []

    cv_rmse_scores = []
    cv_mae_scores = []
    cv_mape_scores = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        if len(X_train) == 0 or len(X_test_fold) == 0:
            continue
        
        # Scaling within CV loop for robust evaluation
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

        model.fit(X_train_scaled, y_train_scaled)
        
        X_test_scaled = scaler_X.transform(X_test_fold)
        y_pred_scaled = model.predict(X_test_scaled)
        
        y_pred_fold = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        y_preds_test_cv.extend(y_pred_fold)
        y_actuals_test_cv.extend(y_test_fold)
        test_dates_cv.extend(df_xgb.index[test_index])

        rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred_fold))
        mae = mean_absolute_error(y_test_fold, y_pred_fold)
        mape = np.mean(np.abs((y_test_fold - y_pred_fold) / y_test_fold)) * 100 if np.all(y_test_fold != 0) else np.nan

        cv_rmse_scores.append(rmse)
        cv_mae_scores.append(mae)
        if not np.isnan(mape):
            cv_mape_scores.append(mape)

    if not cv_rmse_scores:
        return None, "Not enough data for time series cross-validation.", None, None, None, None, np.nan, np.nan, np.nan

    avg_rmse = np.mean(cv_rmse_scores)
    avg_mae = np.mean(cv_mae_scores)
    avg_mape = np.mean(cv_mape_scores) if cv_mape_scores else np.nan

    st.write(f"**XGBoost Cross-Validation (5-fold):**")
    st.write(f"  - **Avg RMSE:** `{avg_rmse:.2f} ± {np.std(cv_rmse_scores):.2f}`")
    st.write(f"  - **Avg MAE:** `{avg_mae:.2f} ± {np.std(cv_mae_scores):.2f}`")
    if not np.isnan(avg_mape):
        st.write(f"  - **Avg MAPE:** `{avg_mape:.2f}% ± {np.std(cv_mape_scores):.2f}%`")
    else:
        st.write("  - **Avg MAPE:** `N/A (due to zero actual values)`")

    # Retrain on full data for future prediction
    scaler_X_full = StandardScaler()
    scaler_y_full = StandardScaler()
    X_scaled_full = scaler_X_full.fit_transform(X)
    y_scaled_full = scaler_y_full.fit_transform(y.values.reshape(-1, 1)).flatten()
    model.fit(X_scaled_full, y_scaled_full)


    last_data_point_features = X.iloc[-1].copy()
    future_predictions_list = []
    future_dates = [df_xgb.index[-1] + pd.Timedelta(days=i + 1) for i in range(forecast_days)]

    # We need to iteratively predict and update features for future steps
    # For simplicity, we'll use the last known values for most features and extrapolate Date_Ordinal
    # For lag features, we'll update them with the predicted value
    current_close_for_lag1 = df_xgb['Close'].iloc[-1]
    current_close_for_lag5 = df_xgb['Close'].iloc[-5] if len(df_xgb) >= 5 else df_xgb['Close'].iloc[-1]

    for i in range(forecast_days):
        # Update Date_Ordinal
        last_data_point_features['Date_Ordinal'] = last_data_point_features['Date_Ordinal'] + 1
        
        # Update lag features
        if 'Close_Lag1' in last_data_point_features:
            last_data_point_features['Close_Lag1'] = current_close_for_lag1
        if 'Close_Lag5' in last_data_point_features:
            last_data_point_features['Close_Lag5'] = current_close_for_lag5

        # For other features, for simplicity, we'll hold them constant or do a very simple extrapolation
        # More sophisticated methods would involve time-series forecasting for each feature
        # Here, we'll just use the last known values for MA, RSI, Volatility, etc.
        # as a pragmatic approach for forecasting future exogenous variables.
        # A more robust solution would involve forecasting these features too.

        # Prepare the feature vector for prediction
        next_day_X_raw = pd.DataFrame([last_data_point_features.to_dict()], columns=available_features)
        next_day_X_scaled = scaler_X_full.transform(next_day_X_raw)

        predicted_scaled_close = model.predict(next_day_X_scaled)[0]
        predicted_close = scaler_y_full.inverse_transform(predicted_scaled_close.reshape(-1, 1))[0, 0]
        predicted_close = max(predicted_close, 0) # Ensure non-negative price

        future_predictions_list.append(predicted_close)

        # Update current_close_for_lag1 and current_close_for_lag5 for the next iteration
        current_close_for_lag5 = current_close_for_lag1
        current_close_for_lag1 = predicted_close
        
        # Update the features for the next iteration based on prediction for recursive forecasting of features
        if 'MA5' in last_data_point_features: # Simple update for MAs, would need more sophisticated logic
            last_data_point_features['MA5'] = (last_data_point_features['MA5'] * 4 + predicted_close) / 5 # crude MA update

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close': np.array(future_predictions_list).round(2)
    })

    return model, forecast_df, test_dates_cv, y_preds_test_cv, y_actuals_test_cv, None, avg_rmse, avg_mae, avg_mape

@st.cache_data(ttl=3600, show_spinner=False)
def run_prophet_forecast(df_prophet, forecast_days):
    """
    Trains and forecasts using Prophet.
    """
    if not PROPHET_AVAILABLE:
        return None, "Prophet library is not installed.", None, None, None, None, np.nan, np.nan, np.nan

    m = Prophet(
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,
        interval_width=0.95 # Added interval width for confidence bands
    )

    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=forecast_days)

    forecast = m.predict(future)

    forecast_with_actuals = pd.merge(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        df_prophet[['ds', 'y']],
        on='ds',
        how='left'
    )

    historical_predictions = forecast_with_actuals.dropna(subset=['y'])

    y_true_hist = historical_predictions['y']
    y_pred_hist = historical_predictions['yhat']

    valid_indices = y_true_hist.notna() & y_pred_hist.notna()
    y_true_hist = y_true_hist[valid_indices]
    y_pred_hist = y_pred_hist[valid_indices]

    rmse, mae, mape = np.nan, np.nan, np.nan

    if len(y_true_hist) > 0:
        rmse = np.sqrt(mean_squared_error(y_true_hist, y_pred_hist))
        mae = mean_absolute_error(y_true_hist, y_pred_hist)
        mape = np.mean(np.abs((y_true_hist - y_pred_hist) / y_true_hist)) * 100 if np.all(y_true_hist != 0) else np.nan
    else:
        st.info("Not enough historical data for in-sample Prophet performance evaluation.")

    return m, forecast_with_actuals, df_prophet['ds'], df_prophet['y'], None, m.plot_components, rmse, mae, mape


# --- SHAP Explainability ---
def explain_with_shap(model, X_sample, feature_names, model_type="linear"):
    """Generates and plots SHAP values for model explainability."""
    if not SHAP_AVAILABLE:
        st.warning("SHAP library not available for explainability.")
        return None
    try:
        if model_type == "linear":
            scaler = model.named_steps['scaler']
            regressor = model.named_steps['regressor']

            X_sample_scaled = scaler.transform(X_sample)

            explainer = shap.LinearExplainer(regressor, X_sample_scaled)
            shap_values = explainer.shap_values(X_sample_scaled)

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, feature_names=feature_names)
            plt.title(f'SHAP Feature Importance - Linear Regression', fontsize=14) # Use plt.title directly
            plt.tight_layout()
            return fig
        elif model_type == "xgboost":
            # For XGBoost, you explain the raw model (scaler applied externally)
            # Ensure X_sample here is already scaled if the model was trained on scaled data
            # Or, use a KernelExplainer if the model is complex
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, feature_names=feature_names)
            plt.title(f'SHAP Feature Importance - XGBoost', fontsize=14)
            plt.tight_layout()
            return fig
        elif model_type == "lstm":
            st.info(
                "SHAP for LSTM models (especially with sequential data) is complex and computationally intensive. Skipping for this demo.")
            return None
        else:
            st.warning(f"Unsupported model type '{model_type}' for SHAP explanation.")
            return None
    except Exception as e:
        st.warning(f"Could not generate SHAP plot for {model_type} model: {e}")
        return None


def display_forecasting(hist_data, ticker):
    """Main function to display stock price forecasting."""
    if not ticker or not isinstance(ticker, str):
        st.error("❌ Invalid ticker symbol. Please enter a valid ticker (e.g., 'AAPL').")
        return

    ticker = ticker.strip().upper()
    st.markdown(f"<h3 class='section-title'>Stock Price Forecasting for {ticker}</h3>", unsafe_allow_html=True)
    st.markdown("---") # Separator

    if hist_data.empty:
        st.warning("⚠️ No historical data provided. Please select a stock and load historical data.")
        return

    st.markdown(
        "<p>This section uses advanced time series models to forecast stock prices based on historical trends. Remember, financial forecasting is complex and inherently uncertain.</p>",
        unsafe_allow_html=True)

    with st.spinner("⏳ Preparing data for forecasting..."):
        df, df_prophet, df_lstm, df_xgb, error = prepare_forecast_data(hist_data, ticker)
        if error:
            st.error(f"❌ Data preparation failed: {error}")
            return
    st.success("✅ Data prepared successfully!")
    st.markdown("---")

    # Sidebar for Model Configuration
    st.sidebar.subheader("⚙️ Forecasting Configuration")

    model_options = ["Linear Regression", "ARIMA", "XGBoost"]
    if PROPHET_AVAILABLE:
        model_options.append("Prophet")
    if LSTM_AVAILABLE:
        model_options.append("LSTM")

    selected_model = st.sidebar.selectbox("🎯 Choose Forecasting Model:", model_options)
    
    # Use st.columns for forecast days for better layout
    col_fdays1, col_fdays2 = st.sidebar.columns([0.6, 0.4])
    with col_fdays1:
        forecast_days = st.slider("📆 Days to forecast:", min_value=7, max_value=90, value=30, step=7,
                                    key=f"forecast_days_slider_{ticker}")
    with col_fdays2:
        st.write(f"<br><br><b>{forecast_days} days</b>", unsafe_allow_html=True)

    # Model-specific parameters
    if selected_model == "LSTM":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧠 LSTM Model Parameters")
        sequence_length = st.sidebar.slider("Sequence Length (Timesteps):", min_value=5, max_value=30, value=20, step=1)
        lstm_hidden_size = st.sidebar.slider("LSTM Hidden Size:", min_value=10, max_value=100, value=50, step=10)
        lstm_num_layers = st.sidebar.slider("LSTM Number of Layers:", min_value=1, max_value=4, value=2, step=1)
        lstm_epochs = st.sidebar.slider("LSTM Training Epochs:", min_value=10, max_value=100, value=50, step=10)
        lstm_batch_size = st.sidebar.slider("LSTM Batch Size:", min_value=16, max_value=64, value=32, step=16)
    elif selected_model == "ARIMA":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔢 ARIMA Model Order (p, d, q)")
        p_order = st.sidebar.slider("P (AR order):", min_value=0, max_value=5, value=5, step=1)
        d_order = st.sidebar.slider("D (Differencing order):", min_value=0, max_value=2, value=1, step=1)
        q_order = st.sidebar.slider("Q (MA order):", min_value=0, max_value=5, value=0, step=1)
        arima_order = (p_order, d_order, q_order)
    else: # Default values if not LSTM/ARIMA
        sequence_length = 20
        lstm_hidden_size = 50
        lstm_num_layers = 2
        lstm_epochs = 50
        lstm_batch_size = 32
        arima_order = (5,1,0) # Default for other models

    model = None
    forecast_df = pd.DataFrame()
    test_dates = None
    test_preds = None
    test_actuals = None
    prophet_components_plot = None
    display_rmse, display_mae, display_mape = np.nan, np.nan, np.nan
    shap_X_explain = pd.DataFrame() # To store data for SHAP
    shap_feature_names = []
    shap_model_type = ""

    st.markdown("---")
    st.markdown(f"<h4><i class='fas fa-chart-line'></i> Running {selected_model} Model...</h4>", unsafe_allow_html=True)
    
    # Model Execution
    if selected_model == "Linear Regression":
        with st.spinner("📈 Training and forecasting with Linear Regression..."):
            model, lr_forecast_df, test_dates, test_preds, test_actuals, _, display_rmse, display_mae, display_mape = \
                run_linear_regression_forecast(df, forecast_days)
            if model is None:
                st.error(f"❌ Linear Regression failed: {lr_forecast_df}") # lr_forecast_df contains error message
                return
            forecast_df = lr_forecast_df
            shap_X_explain = df[model.named_steps['scaler'].feature_names_in_].tail(min(len(df), 200))
            shap_feature_names = model.named_steps['scaler'].feature_names_in_
            shap_model_type = "linear"

    elif selected_model == "Prophet":
        if not PROPHET_AVAILABLE:
            st.error("❌ Prophet library is not installed. Please select 'Linear Regression' or install Prophet.")
            return
        with st.spinner("⏳ Training and forecasting with Prophet..."):
            model, forecast_df, hist_dates, hist_actuals, _, prophet_components_plot, display_rmse, display_mae, display_mape = \
                run_prophet_forecast(df_prophet, forecast_days)
            if model is None:
                st.error(f"❌ Prophet failed: {forecast_df}") # forecast_df contains error message
                return
            test_dates = forecast_df['ds'][forecast_df['y'].notna()]
            test_preds = forecast_df['yhat'][forecast_df['y'].notna()]
            test_actuals = forecast_df['y'][forecast_df['y'].notna()]

    elif selected_model == "ARIMA":
        with st.spinner(f"⏳ Training and forecasting with ARIMA {arima_order}..."):
            model, forecast_df, test_dates, test_preds, test_actuals, display_rmse, display_mae, display_mape = \
                run_arima_forecast(df, forecast_days, *arima_order)
            if model is None:
                st.error(f"❌ ARIMA failed: {forecast_df}") # forecast_df contains error message
                return
    
    elif selected_model == "XGBoost":
        with st.spinner("⚡ Training and forecasting with XGBoost..."):
            model, xgb_forecast_df, test_dates, test_preds, test_actuals, _, display_rmse, display_mae, display_mape = \
                run_xgboost_forecast(df_xgb, forecast_days)
            if model is None:
                st.error(f"❌ XGBoost failed: {xgb_forecast_df}")
                return
            forecast_df = xgb_forecast_df
            # For SHAP, use the raw features, as XGBoost handles scaling internally (or we scaled them before training)
            shap_X_explain = df_xgb[model.feature_names_in_].tail(min(len(df_xgb), 200)) # Use features model was trained on
            shap_feature_names = model.feature_names_in_
            shap_model_type = "xgboost"

    elif selected_model == "LSTM":
        if not LSTM_AVAILABLE:
            st.error("❌ PyTorch library is not installed. Please select another model or install PyTorch.")
            return
        with st.spinner("🚀 Preparing data and training LSTM model..."):
            lstm_features = [
                'MA5', 'MA10', 'MA20', 'MA50', 'Close_Lag1', 'Close_Lag5',
                'Return_1d', 'Volatility', 'RSI', 'Volume_MA5', 'High_Low_Diff', 'Open_Close_Diff'
            ]
            lstm_available_features = [f for f in lstm_features if
                                       f in df_lstm.columns and not df_lstm[f].isnull().all()]

            X_lstm, y_lstm, scaler_lstm, error_msg_lstm_prep = prepare_lstm_data_for_forecast(
                df_lstm, lstm_available_features, sequence_length
            )

            if error_msg_lstm_prep:
                st.warning(f"⚠️ {error_msg_lstm_prep}")
                return

            if X_lstm.shape[0] == 0 or len(y_lstm) == 0:
                st.warning("⚠️ Not enough data to prepare LSTM sequences.")
                return

            train_size_lstm = int(len(X_lstm) * 0.8)
            X_train_lstm, X_test_lstm = X_lstm[:train_size_lstm], X_lstm[train_size_lstm:]
            y_train_lstm, y_test_lstm = y_lstm[:train_size_lstm], y_lstm[train_size_lstm:]

            model, test_preds, test_actuals, error_msg_lstm_train = train_lstm_model_for_forecast(
                X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm,
                input_size=X_lstm.shape[2], hidden_size=lstm_hidden_size, num_layers=lstm_num_layers,
                epochs=lstm_epochs, batch_size=lstm_batch_size
            )

            if error_msg_lstm_train:
                st.warning(f"⚠️ {error_msg_lstm_train}")
                return

            if model is None:
                st.warning("⚠️ LSTM model training failed.")
                return

            if len(test_actuals) > 0:
                display_rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
                display_mae = mean_absolute_error(test_actuals, test_preds)
                display_mape = np.mean(np.abs((test_actuals - test_preds) / test_actuals)) * 100 if np.all(
                    test_actuals != 0) else np.nan

            future_predictions_lstm = []
            # Start with the last sequence from the original data (X_lstm[-1])
            current_sequence_scaled = torch.tensor(X_lstm[-1:], dtype=torch.float32)

            # Get the raw last data point to help update features
            last_actual_features = df_lstm[lstm_available_features].iloc[-1].values
            close_feature_idx = lstm_available_features.index('Close') if 'Close' in lstm_available_features else -1

            for i in range(forecast_days):
                with torch.no_grad():
                    next_scaled_close_pred_tensor = model(current_sequence_scaled)
                    next_scaled_close_pred = next_scaled_close_pred_tensor.numpy().flatten()[0]

                # Create a dummy row for inverse transformation.
                # Only the 'Close' feature is set to the predicted value.
                # Other features are placeholder values that will be replaced.
                dummy_scaled_row = np.zeros((1, len(lstm_available_features)))
                if close_feature_idx != -1:
                    dummy_scaled_row[0, close_feature_idx] = next_scaled_close_pred

                # Inverse transform to get the actual predicted close price
                next_actual_close_pred = scaler_lstm.inverse_transform(dummy_scaled_row)[0, close_feature_idx]
                future_predictions_lstm.append(next_actual_close_pred)

                # Now, prepare the 'next' feature vector for the sliding window
                # This is a crucial and often complex part for LSTM forecasting with multiple features.
                # For simplicity, we'll assume other features mostly stay the same or use simple extrapolation
                # for the purpose of generating the next sequence's input.
                # A more accurate approach would involve forecasting each feature separately or
                # using a multi-variate LSTM if features are highly dependent.

                # We'll re-scale the *predicted* actual close and append it to the current sequence,
                # while keeping other features static for this simplified example.
                
                # To create the next sequence, we need to shift the window and add a new "day"
                # This new "day" must contain scaled values for all features.
                
                # We need to create a new *full* feature vector for the next time step.
                # This is an approximation:
                next_feature_vector_for_scaling = last_actual_features.copy()
                if close_feature_idx != -1:
                    next_feature_vector_for_scaling[close_feature_idx] = next_actual_close_pred
                
                # For other features, we're simply carrying forward their last known value,
                # or you could apply a simple trend extrapolation if preferred.
                # For example, for MAs, you'd calculate them based on `predicted_close` and old values.
                # This example keeps them static.
                
                next_feature_vector_scaled = scaler_lstm.transform(next_feature_vector_for_scaling.reshape(1, -1))
                
                # Create the new sequence by removing the oldest step and adding the newest.
                # current_sequence_scaled is (1, sequence_length, num_features)
                new_sequence_data = current_sequence_scaled.numpy().squeeze(0) # (sequence_length, num_features)
                new_sequence_data = np.roll(new_sequence_data, -1, axis=0) # Shift elements up
                new_sequence_data[-1, :] = next_feature_vector_scaled[0, :] # Place new data at the end

                current_sequence_scaled = torch.tensor(new_sequence_data.reshape(1, sequence_length, -1), dtype=torch.float32)

            forecast_dates_lstm = [df_lstm.index[-1] + pd.Timedelta(days=i + 1) for i in range(forecast_days)]
            forecast_df = pd.DataFrame({
                'Date': forecast_dates_lstm,
                'Predicted Close': np.array(future_predictions_lstm).round(2)
            })
            # Adjust test_dates for LSTM due to sequence length offset
            if len(df_lstm.index) > train_size_lstm + sequence_length:
                test_dates = df_lstm.index[train_size_lstm + sequence_length:]
            else:
                test_dates = pd.DatetimeIndex([]) # Empty if not enough data for test sequence


    if model is None:
        st.error("Something went wrong. Forecasting model could not be trained or retrieved.")
        return

    st.markdown("---")
    st.markdown("<h4>📊 Model Performance Metrics:</h4>", unsafe_allow_html=True)
    
    # Using st.columns for better layout of metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Root Mean Squared Error (RMSE)", 
                  value=f"{display_rmse:.2f}" if not np.isnan(display_rmse) else "N/A",
                  help="Measures the average magnitude of the errors. Lower is better.")
    with col2:
        st.metric(label="Mean Absolute Error (MAE)", 
                  value=f"{display_mae:.2f}" if not np.isnan(display_mae) else "N/A",
                  help="Average of the absolute errors. Less sensitive to outliers than RMSE.")
    with col3:
        st.metric(label="Mean Absolute Percentage Error (MAPE)", 
                  value=f"{display_mape:.2f}%" if not np.isnan(display_mape) else "N/A",
                  help="Average of the absolute percentage errors. Easy to interpret, but sensitive to zero/near-zero actuals.")
    
    st.markdown("---")
    st.markdown("<h4>🔮 Future Price Forecast:</h4>", unsafe_allow_html=True)
    
    # Display the last few days of forecast in a clean table
    if selected_model == "Prophet":
        st.dataframe(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).round(2).rename(
            columns={'ds': 'Date', 'yhat': 'Predicted Close', 'yhat_lower': 'Lower Bound',
                     'yhat_upper': 'Upper Bound'}), use_container_width=True)
    else:
        st.dataframe(forecast_df.tail(forecast_days), use_container_width=True)

    st.markdown("---")
    st.markdown("<h4>📈 Forecast Visualization:</h4>", unsafe_allow_html=True)

    fig = go.Figure()

    # Historical Close Price
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Historical Close Price',
        line=dict(color='cyan', width=2)
    ))

    # Test Set Predictions (if available)
    if test_dates is not None and len(test_dates) > 0 and test_preds is not None and len(test_preds) > 0:
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_preds,
            mode='lines',
            name='Predicted (In-Sample Test)',
            line=dict(color='orange', dash='dot', width=1.5)
        ))
        if selected_model != "Prophet": # Prophet's in-sample is typically its yhat, actuals overlaid
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=test_actuals,
                mode='markers',
                name='Actual (In-Sample Test)',
                marker=dict(color='red', size=4, symbol='circle-open')
            ))
    elif selected_model == "Prophet" and hist_dates is not None and hist_actuals is not None:
         fig.add_trace(go.Scatter(
            x=hist_dates,
            y=forecast_df['yhat'][forecast_df['y'].notna()], # Prophet's in-sample predictions
            mode='lines',
            name='Predicted (Historical In-Sample)',
            line=dict(color='orange', dash='dot', width=1.5)
        ))


    # Future Forecasted Prices
    if selected_model == "Prophet":
        # Add forecasted line
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'].tail(forecast_days),
            y=forecast_df['yhat'].tail(forecast_days),
            mode='lines',
            name=f'Forecasted ({forecast_days} Days)',
            line=dict(color='lime', dash='dash', width=2)
        ))
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['ds'].tail(forecast_days), forecast_df['ds'].tail(forecast_days).iloc[::-1]]),
            y=pd.concat([forecast_df['yhat_upper'].tail(forecast_days),
                         forecast_df['yhat_lower'].tail(forecast_days).iloc[::-1]]),
            fill='toself',
            fillcolor='rgba(0,255,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
    else:  # Linear Regression, ARIMA, LSTM, XGBoost
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Predicted Close'],
            mode='lines',
            name=f'Forecasted ({forecast_days} Days)',
            line=dict(color='lime', dash='dash', width=2)
        ))

    fig.update_layout(
        title=f'📈 Stock Price Forecast for {ticker} ({selected_model})',
        xaxis_title='Date',
        yaxis_title='Close Price',
        template='plotly_dark',
        height=600,
        showlegend=True,
        font=dict(family="Inter", size=12, color="#E0E0E0"),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model Specific Visualizations / Explainability
    st.markdown("---")
    if selected_model == "Prophet" and prophet_components_plot:
        st.markdown("<h4>📉 Prophet Trend and Seasonality Components:</h4>", unsafe_allow_html=True)
        # Prophet's plot_components returns a matplotlib figure object
        st.pyplot(prophet_components_plot.figure)

    if (selected_model == "Linear Regression" or selected_model == "XGBoost") and SHAP_AVAILABLE and model is not None and not shap_X_explain.empty:
        st.markdown("<h4>🔍 Model Explainability (SHAP):</h4>", unsafe_allow_html=True)
        st.info("SHAP values indicate how much each feature contributes to the model's output (positive or negative).")
        
        # Ensure X_explain and feature_names are passed correctly based on the model
        if selected_model == "Linear Regression":
            X_shap_data = df[model.named_steps['scaler'].feature_names_in_].tail(min(len(df), 200))
            current_feature_names = model.named_steps['scaler'].feature_names_in_
            current_model_for_shap = model.named_steps['regressor'] # Pass the regressor, not the pipeline
            shap_plot = explain_with_shap(current_model_for_shap, X_shap_data, current_feature_names, "linear")
        elif selected_model == "XGBoost":
            X_shap_data = df_xgb[model.feature_names_in_].tail(min(len(df_xgb), 200))
            current_feature_names = model.feature_names_in_
            current_model_for_shap = model
            shap_plot = explain_with_shap(current_model_for_shap, X_shap_data, current_feature_names, "xgboost")
        else:
            shap_plot = None # Should not happen with the if condition above

        if shap_plot:
            st.pyplot(shap_plot)
        else:
            st.warning("SHAP plot could not be generated for this model or data.")

    st.markdown("---")
    st.warning("""
    **Disclaimer for Forecasting:**
    Stock price forecasting is inherently challenging and subject to numerous unpredictable factors.
    - **Model Limitations:** While advanced models like Prophet, LSTM, ARIMA, and XGBoost account for trends and seasonality, they do not fully incorporate real-world events, breaking news, macroeconomic shifts, company-specific announcements, or significant shifts in investor sentiment. The future is uncertain.
    - **Accuracy:** Forecasts are based on historical patterns and computational algorithms; they **do not guarantee future performance.** Past performance is not indicative of future results.
    - **Risk:** **These forecasts are for informational and educational purposes only and should NOT be used for actual investment decisions.** Always conduct your own thorough research, consider multiple sources of information, and consult with a qualified financial advisor before making any investment choices.
    """)

    st.markdown("---")
    st.info("💡 **Tip:** Experiment with different models and 'Days to Forecast' in the sidebar to see how they impact the predictions and performance metrics!")
