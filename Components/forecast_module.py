import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Added MinMaxScaler for LSTM
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit  # For time series specific cross-validation
import ta  # For technical analysis indicators

# Suppress Prophet's verbose logging
import logging

logging.getLogger('prophet').setLevel(logging.WARNING)

# Conditional import for Prophet to avoid errors if not installed during initial checks
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    # st.warning("Prophet library not found. Please install it (`pip install prophet`) to use Prophet forecasting.")

# Imports for LSTM and SHAP
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    # st.warning("PyTorch library not found. Please install it (`pip install torch`) to use LSTM forecasting.")

try:
    import shap
    import matplotlib.pyplot as plt

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    # st.warning("SHAP library not found. Please install it (`pip install shap matplotlib`) for model explainability.")


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
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "No historical data provided."

    df = hist_data.copy()
    required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Missing required columns: {', '.join(missing_cols)}."

    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().all():
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"All '{col}' values are invalid or missing."

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "Could not convert index to datetime format."

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
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"Not enough data after feature engineering (minimum {min_data_points} data points required)."

    df_prophet = df[['Close']].copy()
    df_prophet.index.name = 'ds'
    df_prophet = df_prophet.reset_index()
    df_prophet = df_prophet.rename(columns={'Close': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    df_lstm = df.copy()

    return df, df_prophet, df_lstm, None


# --- Deep Learning Model (LSTM) ---
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


# --- Traditional ML Models (Linear Regression) ---
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

    st.write(f"**Linear Regression Cross-Validation (5-fold):**")
    st.write(f"  - **Avg RMSE:** `{avg_rmse:.2f} ± {np.std(cv_rmse_scores):.2f}`")
    st.write(f"  - **Avg MAE:** `{avg_mae:.2f} ± {np.std(cv_mae_scores):.2f}`")
    if not np.isnan(avg_mape):
        st.write(f"  - **Avg MAPE:** `{avg_mape:.2f}% ± {np.std(cv_mape_scores):.2f}%`")
    else:
        st.write("  - **Avg MAPE:** `N/A (due to zero actual values)`")

    model_pipeline.fit(X, y)

    last_date_ordinal = df['Date_Ordinal'].iloc[-1]

    future_features_data = {}
    for feature in ['MA5', 'MA10', 'MA20', 'MA50', 'Return_1d', 'Volatility', 'RSI', 'Volume_MA5', 'High_Low_Diff',
                    'Open_Close_Diff']:
        if feature in df.columns:
            recent_data_for_extrapolation = df[feature].tail(20)
            if len(recent_data_for_extrapolation) >= 2:
                ma_X = np.arange(len(recent_data_for_extrapolation)).reshape(-1, 1)
                ma_y = recent_data_for_extrapolation.values
                ma_model = LinearRegression().fit(ma_X, ma_y)
                future_steps = np.arange(1, forecast_days + 1).reshape(-1, 1)
                future_features_data[feature] = ma_model.predict(future_steps)
                future_features_data[feature] = np.maximum(future_features_data[feature], 0)
            else:
                future_features_data[feature] = [df[feature].iloc[-1]] * forecast_days
        else:
            future_features_data[feature] = [0.0] * forecast_days

    future_features_data['Close_Lag1'] = [df['Close'].iloc[-1]] * forecast_days
    future_features_data['Close_Lag5'] = [df['Close'].iloc[-1]] * forecast_days

    future_dates_ordinal = [last_date_ordinal + i for i in range(1, forecast_days + 1)]
    future_features_data['Date_Ordinal'] = future_dates_ordinal

    X_future = pd.DataFrame(future_features_data, columns=available_features)
    future_predictions = model_pipeline.predict(X_future)
    future_predictions = np.maximum(future_predictions, 0)

    forecast_df = pd.DataFrame({
        'Date': [pd.Timestamp.fromordinal(int(date)).date() for date in future_dates_ordinal],
        'Predicted Close': future_predictions.round(2)
    })

    return model_pipeline, forecast_df, test_dates_cv, y_preds_test_cv, y_actuals_test_cv, None, avg_rmse, avg_mae, avg_mape


@st.cache_data(ttl=3600, show_spinner=False)
def run_prophet_forecast(df_prophet, forecast_days):
    """
    Trains and forecasts using Prophet.
    """
    if not PROPHET_AVAILABLE:
        return None, "Prophet library is not installed.", None, None, None, None, np.nan, np.nan, np.nan

    m = Prophet(
        seasonality_mode='additive',
        changepoint_prior_scale=0.05
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

        st.write(f"**Prophet In-Sample Performance:**")
        st.write(f"  - **RMSE:** `{rmse:.2f}`")
        st.write(f"  - **MAE:** `{mae:.2f}`")
        if not np.isnan(mape):
            st.write(f"  - **MAPE:** `{mape:.2f}%`")
        else:
            st.write("  - **MAPE:** `N/A (due to zero actual values)`")
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
            # Removed ax=ax from shap.summary_plot as it can cause issues with some versions/plots
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, feature_names=feature_names)
            ax.set_title(f'SHAP Feature Importance - {model.__class__.__name__}', fontsize=14)
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
        st.warning(f"Could not generate SHAP plot for {model.__class__.__name__}: {e}")
        return None


def display_forecasting(hist_data, ticker):
    """Main function to display stock price forecasting."""
    if not ticker or not isinstance(ticker, str):
        st.error("❌ Invalid ticker symbol. Please enter a valid ticker (e.g., 'AAPL').")
        return

    ticker = ticker.strip().upper()
    st.markdown(f"<h3 class='section-title'>Stock Price Forecasting for {ticker}</h3>", unsafe_allow_html=True)

    if hist_data.empty:
        st.warning("⚠️ No historical data provided. Please select a stock and load historical data.")
        return

    st.markdown(
        "<p>This section uses advanced time series models to forecast stock prices based on historical trends. Remember, financial forecasting is complex and inherently uncertain.</p>",
        unsafe_allow_html=True)

    with st.spinner("Preparing data for forecasting..."):
        df, df_prophet, df_lstm, error = prepare_forecast_data(hist_data, ticker)
        if error:
            st.warning(f"⚠️ {error} Cannot perform forecasting.")
            return

    st.sidebar.subheader("Forecasting Configuration")

    model_options = ["Linear Regression"]
    if PROPHET_AVAILABLE:
        model_options.append("Prophet")
    if LSTM_AVAILABLE:
        model_options.append("LSTM")

    selected_model = st.sidebar.selectbox("Choose Forecasting Model:", model_options)
    forecast_days = st.sidebar.slider("Number of days to forecast:", min_value=7, max_value=90, value=30, step=7,
                                      key=f"forecast_days_slider_{ticker}")

    if selected_model == "LSTM":
        st.sidebar.markdown("---")
        st.sidebar.subheader("LSTM Model Parameters")
        sequence_length = st.sidebar.slider("Sequence Length (Timesteps):", min_value=5, max_value=30, value=20, step=1)
        lstm_hidden_size = st.sidebar.slider("LSTM Hidden Size:", min_value=10, max_value=100, value=50, step=10)
        lstm_num_layers = st.sidebar.slider("LSTM Number of Layers:", min_value=1, max_value=4, value=2, step=1)
        lstm_epochs = st.sidebar.slider("LSTM Training Epochs:", min_value=10, max_value=100, value=50, step=10)
        lstm_batch_size = st.sidebar.slider("LSTM Batch Size:", min_value=16, max_value=64, value=32, step=16)
    else:
        sequence_length = 20
        lstm_hidden_size = 50
        lstm_num_layers = 2
        lstm_epochs = 50
        lstm_batch_size = 32

    model = None
    forecast_df = pd.DataFrame()
    test_dates = None
    test_preds = None
    test_actuals = None
    prophet_components_plot = None

    display_rmse, display_mae, display_mape = np.nan, np.nan, np.nan

    if selected_model == "Linear Regression":
        with st.spinner("Training and forecasting with Linear Regression..."):
            model, lr_forecast_df, test_dates, test_preds, test_actuals, _, display_rmse, display_mae, display_mape = run_linear_regression_forecast(
                df, forecast_days)
            if lr_forecast_df is None and isinstance(model, str):
                st.warning(f"⚠️ {model}")
                return
            elif lr_forecast_df is None:
                st.warning("⚠️ Linear Regression forecasting failed with an unknown error.")
                return
            forecast_df = lr_forecast_df

    elif selected_model == "Prophet":
        if not PROPHET_AVAILABLE:
            st.error("Prophet library is not installed. Please select 'Linear Regression' or install Prophet.")
            return
        with st.spinner("Training and forecasting with Prophet..."):
            model, forecast_df, hist_dates, hist_actuals, _, prophet_components_plot, display_rmse, display_mae, display_mape = run_prophet_forecast(
                df_prophet, forecast_days)
            if model is None:
                st.warning(f"⚠️ {forecast_df}")
                return
            test_dates = forecast_df['ds'][forecast_df['y'].notna()]
            test_preds = forecast_df['yhat'][forecast_df['y'].notna()]
            test_actuals = forecast_df['y'][forecast_df['y'].notna()]

    elif selected_model == "LSTM":
        if not LSTM_AVAILABLE:
            st.error("PyTorch library is not installed. Please select another model or install PyTorch.")
            return
        with st.spinner("Preparing data and training LSTM model..."):
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
            current_sequence_scaled = torch.tensor(X_lstm[-1:], dtype=torch.float32)

            last_actual_features = df_lstm[lstm_available_features].iloc[-1].values
            close_feature_idx = lstm_available_features.index('Close') if 'Close' in lstm_available_features else -1

            for i in range(forecast_days):
                with torch.no_grad():
                    next_scaled_close_pred = model(current_sequence_scaled).numpy().flatten()[0]

                dummy_scaled_row = np.zeros((1, len(lstm_available_features)))
                if close_feature_idx != -1:
                    dummy_scaled_row[0, close_feature_idx] = next_scaled_close_pred

                next_actual_close_pred = scaler_lstm.inverse_transform(dummy_scaled_row)[0, close_feature_idx]
                future_predictions_lstm.append(next_actual_close_pred)

                next_feature_vector = last_actual_features.copy()
                if close_feature_idx != -1:
                    next_feature_vector[close_feature_idx] = next_actual_close_pred

                next_feature_vector_scaled = scaler_lstm.transform(next_feature_vector.reshape(1, -1))

                new_sequence_scaled = np.roll(current_sequence_scaled.numpy(), -1, axis=1)
                new_sequence_scaled[0, -1, :] = next_feature_vector_scaled
                current_sequence_scaled = torch.tensor(new_sequence_scaled, dtype=torch.float32)

            forecast_dates_lstm = [df_lstm.index[-1] + pd.Timedelta(days=i + 1) for i in range(forecast_days)]
            forecast_df = pd.DataFrame({
                'Date': forecast_dates_lstm,
                'Predicted Close': np.array(future_predictions_lstm).round(2)
            })
            test_dates = df_lstm.index[train_size_lstm + sequence_length:]

    if model is None:
        st.error("Failed to train or retrieve forecasting model.")
        return

    st.markdown("<h5>Model Performance Metrics:</h5>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="RMSE", value=f"{display_rmse:.2f}" if not np.isnan(display_rmse) else "N/A")
    with col2:
        st.metric(label="MAE", value=f"{display_mae:.2f}" if not np.isnan(display_mae) else "N/A")
    with col3:
        st.metric(label="MAPE", value=f"{display_mape:.2f}%" if not np.isnan(display_mape) else "N/A")

    st.markdown("<h5>Future Price Forecast:</h5>", unsafe_allow_html=True)
    if selected_model == "Prophet":
        st.dataframe(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).round(2).rename(
            columns={'ds': 'Date', 'yhat': 'Predicted Close', 'yhat_lower': 'Lower Bound',
                     'yhat_upper': 'Upper Bound'}), use_container_width=True)
    else:
        st.dataframe(forecast_df.tail(forecast_days), use_container_width=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Historical Close Price',
        line=dict(color='cyan', width=2)
    ))

    if test_dates is not None and len(test_dates) > 0:
        if selected_model == "Prophet":
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=test_preds,
                mode='lines',
                name='Predicted (Historical In-Sample)',
                line=dict(color='orange', dash='dot', width=1.5)
            ))
        elif selected_model == "LSTM":
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=test_preds,
                mode='lines',
                name='Predicted (Test Set)',
                line=dict(color='orange', dash='dot', width=1.5)
            ))
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=test_actuals,
                mode='markers',
                name='Actual (Test Set)',
                marker=dict(color='red', size=4, symbol='circle-open')
            ))
        else:  # Linear Regression
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=test_preds,
                mode='lines',
                name='Predicted (Test Set)',
                line=dict(color='orange', dash='dot', width=1.5)
            ))
            fig.add_trace(go.Scatter(
                x=test_dates,
                y=test_actuals,
                mode='markers',
                name='Actual (Test Set)',
                marker=dict(color='red', size=4, symbol='circle-open')
            ))

    if selected_model == "Prophet":
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'].tail(forecast_days),
            y=forecast_df['yhat'].tail(forecast_days),
            mode='lines',
            name=f'Forecasted ({forecast_days} Days)',
            line=dict(color='lime', dash='dash', width=2)
        ))
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
    else:  # Linear Regression or LSTM
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

    if selected_model == "Prophet" and prophet_components_plot:
        st.markdown("<h5>Prophet Trend and Seasonality Components:</h5>", unsafe_allow_html=True)
        st.pyplot(prophet_components_plot.figure)

    if selected_model == "Linear Regression" and SHAP_AVAILABLE and model is not None:
        st.markdown("<h5>Model Explainability (SHAP):</h5>", unsafe_allow_html=True)
        # Use a subset of the data for SHAP explanation for performance
        X_explain = df[model.named_steps['scaler'].feature_names_in_].tail(200)  # Use features model was trained on
        shap_plot = explain_with_shap(model, X_explain, model.named_steps['scaler'].feature_names_in_, "linear")
        if shap_plot:
            st.pyplot(shap_plot)

    st.warning("""
    **Disclaimer for Forecasting:**
    Stock price forecasting is inherently challenging and subject to numerous unpredictable factors.
    - **Model Limitations:** While advanced models like Prophet and LSTM account for trends and seasonality, they do not incorporate real-world events, news, macroeconomic shifts, or company-specific announcements. Linear Regression assumes a linear relationship, which is often not the case in financial markets.
    - **Accuracy:** Forecasts are based on historical patterns and do not guarantee future performance. Past performance is not indicative of future results.
    - **Risk:** **These forecasts are for informational and educational purposes only and should NOT be used for actual investment decisions.** Always conduct your own thorough research, consider multiple sources of information, and consult with a qualified financial advisor before making any investment choices.
    """)
