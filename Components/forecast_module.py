import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from prophet.plot import plot_plotly # Keeping it for potential use, but not strictly necessary for this logic
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shap
import matplotlib.pyplot as plt
import warnings
# import yfinance as yf # Not directly used in this module now, hist_data is passed in

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set Matplotlib style for consistency with Streamlit dark theme
plt.style.use('dark_background')
plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})


# --- Utility Functions (Cached) ---
@st.cache_data(ttl=3600, show_spinner=False)
def calculate_metrics(y_true, y_pred):
    """Calculates RMSE, MAE, MAPE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # Avoid division by zero in MAPE for 0 or negative actual values, handle NaN/Inf
    y_true_clean = y_true[y_true != 0] # Filter out zeros for MAPE calculation
    y_pred_clean = y_pred[y_true != 0]

    if not y_true_clean.empty:
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    else:
        mape = float('inf') # Or 0 if all true values are 0

    return rmse, mae, mape

@st.cache_data(ttl=3600, show_spinner=False)
def prepare_data_for_forecasting(hist_data):
    """Prepares data for various forecasting models."""
    df = hist_data.copy()
    # Ensure 'Date' column exists and is datetime, then set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    df.index = pd.to_datetime(df.index) # Ensure index is datetime even if already set

    # Some models might need `ds` and `y` columns. Create them for consistency.
    df_prophet_format = df[['Close']].rename(columns={'Close': 'y'})
    df_prophet_format['ds'] = df_prophet_format.index
    return df_prophet_format


# --- ARIMA Model ---
@st.cache_resource(show_spinner=False) # Cache the trained model
def train_arima_model(train_data_series, order=(5, 1, 0)):
    """Trains an ARIMA model."""
    try:
        model = ARIMA(train_data_series, order=order)
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        st.error(f"ARIMA model training failed: {e}")
        return None

# --- Prophet Model ---
@st.cache_resource(show_spinner=False) # Cache the trained model
def train_prophet_model(train_data_df):
    """Trains a Prophet model."""
    m = Prophet(seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False)
    m.fit(train_data_df)
    return m


# --- XGBoost Model (Regression) ---
@st.cache_resource(show_spinner=False) # Cache the trained model and data for prediction
def train_xgboost_model(df_full, days_to_forecast):
    """Prepares data and trains an XGBoost Regressor model."""
    df_xgb = df_full.copy()

    # Create lag features for 'y' (Close price)
    for i in range(1, 11): # Lag features for past 10 days
        df_xgb[f'lag_{i}'] = df_xgb['y'].shift(i)
    # Moving averages
    df_xgb['rolling_mean_5'] = df_xgb['y'].rolling(window=5).mean()
    df_xgb['rolling_std_5'] = df_xgb['y'].rolling(window=5).std()

    # Drop rows with NaN created by lag/rolling features
    df_xgb.dropna(inplace=True)

    # Features and Target
    # Exclude 'ds' (datetime) and 'y' (target) from features
    features = [col for col in df_xgb.columns if col not in ['ds', 'y']]
    X = df_xgb[features]
    y = df_xgb['y']

    if X.empty or len(X) < 2:
        return None, None, None, None

    # Split data: Use the last `days_to_forecast` as test data, rest as train
    # Ensure there's enough data for both train and test
    if len(X) <= days_to_forecast + 1: # Need at least one sample for train
        st.warning(f"Not enough data for XGBoost (effective data points: {len(X)}, forecast days: {days_to_forecast}). Skipping XGBoost.")
        return None, None, None, None

    train_size = len(X) - days_to_forecast
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = xgb.XGBRegressor(objective='reg:squarederror',
                             n_estimators=200, # Increased estimators for potentially better performance
                             learning_rate=0.05, # Slightly reduced learning rate
                             random_state=42,
                             n_jobs=-1) # Use all available cores

    model.fit(X_train, y_train)

    # Return model, the actual X_test for prediction/SHAP, y_test for evaluation, and feature names
    return model, features, X_test, y_test


# --- LSTM Model (Regression) ---
class LSTMRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegressionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # input_size: number of features at each time step (which is 1 for univariate time series)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Take the output from the last time step
        return out

@st.cache_resource(show_spinner=False)
def prepare_lstm_data_regression(df_full, sequence_length=20):
    """
    Prepares data for LSTM regression.
    Returns X (sequences), y (next value), and the scaler.
    """
    data = df_full['y'].values.reshape(-1, 1) # Ensure it's 2D for scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i : (i + sequence_length), 0]) # Extract a 1D sequence
        y.append(scaled_data[i + sequence_length, 0]) # The value to predict

    return np.array(X), np.array(y), scaler

@st.cache_resource(show_spinner=False)
def train_lstm_model_regression(X_train, y_train, sequence_length, hidden_size=50, num_layers=2, epochs=50, batch_size=32):
    """Trains an LSTM regression model."""
    if X_train.shape[0] == 0:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # X_train_tensor needs to be (batch_size, sequence_length, num_features)
    # num_features is 1 for univariate time series
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # input_size should be 1 as it's a univariate time series
    model = LSTMRegressionModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    return model


# --- Plotting Functions ---
def plot_forecast(actual, forecast, model_name, historical_data_plot=None):
    """Plots actual vs. forecast using Plotly."""
    fig = go.Figure()

    if historical_data_plot is not None and not historical_data_plot.empty:
        # Plot historical data up to the last known point before forecast
        fig.add_trace(go.Scatter(x=historical_data_plot.index, y=historical_data_plot['Close'],
                                 mode='lines', name='Historical Close',
                                 line=dict(color='lightgray')))

    fig.add_trace(go.Scatter(x=actual.index, y=actual, mode='lines', name='Actual Future Prices',
                             line=dict(color='blue', width=2))) # Made lines a bit thicker
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name=f'{model_name} Forecast',
                             line=dict(color='red', dash='dash', width=2)))

    fig.update_layout(
        title=f'Stock Price Forecast ({model_name})',
        xaxis_title='Date',
        yaxis_title='Price',
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Inter", size=12, color="#E0E0E0")
    )
    return fig

def plot_shap_summary(model, X_data_for_shap, model_name="Model"):
    """Generates and plots SHAP values."""
    if X_data_for_shap.empty:
        st.warning(f"No data available for SHAP explanation for {model_name}.")
        return False

    try:
        # Determine explainer based on model type
        if isinstance(model, (xgb.XGBRegressor, xgb.XGBClassifier)):
            explainer = shap.TreeExplainer(model)
        else: # Fallback to KernelExplainer for other models (less efficient for tree models)
            # KernelExplainer needs a background dataset; use a sample of X_data_for_shap
            # Using a smaller sample for background data for speed
            background_data = X_data_for_shap.sample(min(50, len(X_data_for_shap)), random_state=42)
            explainer = shap.KernelExplainer(model.predict, background_data)

        # Calculate SHAP values - use a sample of X_data_for_shap for summary plot for speed
        shap_values = explainer.shap_values(X_data_for_shap.sample(min(200, len(X_data_for_shap)), random_state=42))

        fig, ax = plt.subplots(figsize=(10, 6))
        # Ensure SHAP plot is correctly displayed for regression (shap_values is not a list)
        if isinstance(shap_values, list): # For multi-output models, take the first output
             shap.summary_plot(shap_values[0], X_data_for_shap.columns, plot_type="bar", show=False, ax=ax,
                              color_bar=True, cmap='coolwarm', max_display=10)
        else:
             shap.summary_plot(shap_values, X_data_for_shap.columns, plot_type="bar", show=False, ax=ax,
                              color_bar=True, cmap='coolwarm', max_display=10)

        ax.set_title(f'SHAP Feature Importance - {model_name}', fontsize=14, color='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig) # Close the figure to free memory
        return True
    except Exception as e:
        st.error(f"Error generating SHAP plot for {model_name}: {e}")
        st.info("SHAP explanation might fail if the model has no features, or if data is unsuitable.")
        return False


# --- Main Forecasting Display Function ---
def display_forecasting(hist_data, ticker):
    st.markdown(f"<h3 class='section-title'>📈 Stock Price Forecasting for {ticker}</h3>", unsafe_allow_html=True)
    st.markdown("""
        <p>This section uses advanced time series models to forecast stock prices based on historical trends.
        Remember, financial forecasting is complex and inherently uncertain.</p>
        """, unsafe_allow_html=True)

    if hist_data is None or hist_data.empty:
        st.info("📊 Please select a stock and date range in the sidebar to load historical data for forecasting.")
        return

    # User Configuration
    st.subheader("⚙️ Forecasting Configuration")
    days_to_forecast = st.slider("Days to Forecast", min_value=5, max_value=60, value=30)
    selected_model = st.selectbox(
        "Choose Forecasting Model",
        ["Prophet", "XGBoost (Regression)", "LSTM (Regression)", "ARIMA"]
    )
    st.markdown("---")

    # Prepare data (cached for efficiency)
    with st.spinner("Preparing data for forecasting... ⏳"):
        df_forecast = prepare_data_for_forecasting(hist_data)
        # Ensure df_forecast index is unique and sorted for reliable splitting
        df_forecast = df_forecast[~df_forecast.index.duplicated(keep='first')].sort_index()

    if df_forecast.empty:
        st.error("❌ Data preparation for forecasting failed. Please check the input historical data.")
        return

    # Split data into train and actual future for evaluation
    # Ensure train_size is at least enough for training and future prediction
    if len(df_forecast) < days_to_forecast + 2: # At least 2 points for train + forecast days
        st.error(f"❌ Not enough historical data ({len(df_forecast)} days) to forecast {days_to_forecast} days. Please select a longer historical range (min {days_to_forecast + 2} days needed).")
        return

    train_data_df = df_forecast.iloc[:-days_to_forecast]
    actual_future_data_df = df_forecast.iloc[-days_to_forecast:]

    if train_data_df.empty:
        st.error("❌ Training data is empty after splitting. Please ensure you have sufficient historical data.")
        return

    forecast_results = None
    model_performance = {}
    current_model = None
    # For XGBoost SHAP:
    XGB_X_test_for_shap = None # Will store the X_test for SHAP
    XGB_feature_names = None

    with st.spinner(f"Running {selected_model} Model... 🚀"):
        try:
            if selected_model == "ARIMA":
                # ARIMA expects a Series, not a DataFrame
                model_arima = train_arima_model(train_data_df['y'])
                if model_arima:
                    # Forecast dates should start immediately after the training data ends
                    forecast_period_start = train_data_df.index[-1] + pd.Timedelta(days=1)
                    # Generate a date range for the forecast period (calendar days)
                    forecast_period_index = pd.date_range(start=forecast_period_start,
                                                          periods=days_to_forecast, freq='D')

                    arima_forecast_array = model_arima.forecast(steps=days_to_forecast)
                    # Convert to Series with the correct future index
                    forecast_results = pd.Series(arima_forecast_array, index=forecast_period_index, name='yhat')
                    current_model = model_arima
                else:
                    st.warning("ARIMA model could not be trained.")

            elif selected_model == "Prophet":
                m_prophet = train_prophet_model(train_data_df)
                if m_prophet:
                    # Prophet's make_future_dataframe accounts for daily frequency by default
                    future = m_prophet.make_future_dataframe(periods=days_to_forecast, include_history=False)
                    if future.empty:
                        st.warning("Prophet could not generate future dataframe. Check data frequency.")
                    else:
                        forecast_prophet_df = m_prophet.predict(future)
                        # Align the forecast with the actual future data's index
                        # Use merge_asof or reindex for robust date alignment if dates might differ (e.g., weekends vs trading days)
                        # A simple join/reindex might be enough if dates perfectly align, but .loc caused issues.
                        # Let's align based on 'ds' (date)
                        forecast_results_temp = forecast_prophet_df[['ds', 'yhat']].set_index('ds')

                        # Filter and reindex to match actual_future_data's dates
                        # This will handle cases where Prophet forecasts non-trading days that are not in actual_future_data
                        forecast_results = forecast_results_temp.reindex(actual_future_data_df.index)['yhat'].dropna()
                        current_model = m_prophet
                else:
                    st.warning("Prophet model could not be trained.")

            elif selected_model == "XGBoost (Regression)":
                xgb_model, features, X_test_xgb, y_test_xgb = train_xgboost_model(df_forecast, days_to_forecast)
                if xgb_model and features and not X_test_xgb.empty:
                    # Make predictions on the test set for evaluation
                    xgb_predictions = xgb_model.predict(X_test_xgb)
                    forecast_results = pd.Series(xgb_predictions, index=y_test_xgb.index, name='yhat')
                    current_model = xgb_model
                    XGB_X_test_for_shap = X_test_xgb # This is the correct data for SHAP
                    XGB_feature_names = features
                else:
                    st.warning("XGBoost model could not be trained or insufficient data for testing.")

            elif selected_model == "LSTM (Regression)":
                sequence_length = 20 # You can make this configurable
                X_lstm_full, y_lstm_full, scaler_lstm = prepare_lstm_data_regression(df_forecast, sequence_length)

                if X_lstm_full.shape[0] == 0:
                    st.warning("Not enough data for LSTM model after preparing sequences. Skipping LSTM training.")
                    lstm_model = None
                else:
                    # Split for training and testing LSTM
                    # The test set for LSTM needs to correspond to the actual future data's length
                    # X_lstm_full is already aligned such that y_lstm_full[i] is the next value after X_lstm_full[i]
                    # So, the last `days_to_forecast` in `y_lstm_full` are our future values.
                    # The corresponding `X_lstm_full` are the sequences *leading up to* those future values.

                    if len(X_lstm_full) < days_to_forecast + 1: # Need at least one seq for training
                        st.warning(f"Not enough sequential data for LSTM training/forecasting ({len(X_lstm_full)} sequences, need at least {days_to_forecast + 1}). Skipping LSTM.")
                        lstm_model = None
                    else:
                        train_size_lstm = len(X_lstm_full) - days_to_forecast
                        X_train_lstm = X_lstm_full[:train_size_lstm]
                        y_train_lstm = y_lstm_full[:train_size_lstm]
                        X_test_lstm = X_lstm_full[train_size_lstm:]
                        # y_test_lstm is implicit for evaluation later (actual_future_data_df['y'])

                        if X_train_lstm.shape[0] == 0 or X_test_lstm.shape[0] == 0:
                            st.warning("Not enough data to split into train/test sets for LSTM. Skipping LSTM training.")
                            lstm_model = None
                        else:
                            # input_size for LSTM is 1 as it's a univariate series (one feature per time step)
                            lstm_model = train_lstm_model_regression(X_train_lstm, y_train_lstm, sequence_length, input_size=1)

                if lstm_model:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # Make predictions on the test set
                    X_test_lstm_tensor = torch.tensor(X_test_lstm, dtype=torch.float32).unsqueeze(-1).to(device)
                    lstm_model.eval()
                    with torch.no_grad():
                        lstm_predictions_scaled = lstm_model(X_test_lstm_tensor).cpu().numpy()

                    # Inverse transform to get actual prices
                    lstm_predictions = scaler_lstm.inverse_transform(lstm_predictions_scaled)
                    # The index for LSTM predictions must align with the actual_future_data_df
                    forecast_results = pd.Series(lstm_predictions.flatten(), index=actual_future_data_df.index, name='yhat')
                    current_model = lstm_model
                else:
                    st.warning("LSTM model could not be trained.")

        except Exception as e:
            st.error(f"Error during {selected_model} model execution: {e}")
            import traceback
            st.code(traceback.format_exc()) # Display full traceback for debugging

    if forecast_results is not None and not forecast_results.empty:
        # Align forecast_results and actual_future_data_df['y'] indexes before metric calculation
        # This handles cases where Prophet or ARIMA might forecast non-trading days
        common_index = forecast_results.index.intersection(actual_future_data_df.index)
        if common_index.empty:
            st.warning("No common dates between forecast and actual data for evaluation. Cannot calculate metrics.")
            st.plotly_chart(plot_forecast(actual_future_data_df['y'], forecast_results, selected_model, hist_data), use_container_width=True)
        else:
            aligned_forecast = forecast_results.loc[common_index]
            aligned_actual = actual_future_data_df['y'].loc[common_index]

            st.success(f"✅ {selected_model} Model ran successfully!")
            rmse, mae, mape = calculate_metrics(aligned_actual, aligned_forecast)
            model_performance = {"RMSE": rmse, "MAE": mae, "MAPE": mape}

            st.markdown("<h4>📊 Model Performance Metrics:</h4>", unsafe_allow_html=True)
            col_rmse, col_mae, col_mape = st.columns(3)
            with col_rmse:
                st.metric("Root Mean Squared Error (RMSE)", f"{model_performance['RMSE']:.2f}")
            with col_mae:
                st.metric("Mean Absolute Error (MAE)", f"{model_performance['MAE']:.2f}")
            with col_mape:
                st.metric("Mean Absolute Percentage Error (MAPE)", f"{model_performance['MAPE']:.2f}%")

            st.markdown("<h4>📈 Forecast Visualization:</h4>", unsafe_allow_html=True)
            st.plotly_chart(plot_forecast(aligned_actual, aligned_forecast, selected_model, hist_data), use_container_width=True)

            st.markdown("<h4>🔮 Future Price Forecast:</h4>", unsafe_allow_html=True)
            # Display the raw forecast results, they are already indexed
            st.dataframe(pd.DataFrame(forecast_results).rename(columns={'yhat': 'Predicted Close Price'}).style.format({"Predicted Close Price": "{:.2f}"}))

    else:
        st.warning(f"No forecast generated for {selected_model}. Please check configuration or data.")


    # --- Model Explainability (SHAP) ---
    st.markdown("---")
    st.markdown("<h4 class='section-subtitle'>🔍 Model Explainability (SHAP)</h4>", unsafe_allow_html=True)
    st.info("SHAP values indicate how much each feature contributes to the model's output (positive or negative).")

    shap_success = False
    if selected_model == "XGBoost (Regression)" and current_model and XGB_X_test_for_shap is not None and not XGB_X_test_for_shap.empty:
        # Use a small sample of the test data for SHAP explanation for performance
        # X_test_xgb already contains the correct features
        shap_X_explain = XGB_X_test_for_shap.tail(min(len(XGB_X_test_for_shap), 200)).copy()
        shap_success = plot_shap_summary(current_model, shap_X_explain, "XGBoost (Regression)")
    elif selected_model == "Prophet" and current_model:
        st.info("SHAP explanations are not directly applicable to Prophet as it's a statistical model focused on trend, seasonality, and holidays, not feature importance in the same way as ML models. You can explore Prophet's components using `m.plot_components(forecast)` if needed.")
        shap_success = True
    elif selected_model == "ARIMA" and current_model:
        st.info("ARIMA models are statistical and do not have 'features' in the same sense as machine learning models, so SHAP explanations are not directly applicable.")
        shap_success = True
    elif selected_model == "LSTM (Regression)" and current_model:
        st.info("SHAP explanations for LSTM models are generally more complex and often require specialized libraries (e.g., DeepLift, Integrated Gradients) which are outside the scope of this simplified SHAP implementation for general purpose feature importance.")
        shap_success = True

    if not shap_success and selected_model not in ["Prophet", "ARIMA", "LSTM (Regression)"]:
        st.warning("SHAP plot could not be generated for this model or data.")

    # --- Disclaimer ---
    st.markdown("---")
    st.warning("""
    **🚨 Disclaimer for Forecasting:**
    Stock price forecasting is inherently challenging and subject to numerous unpredictable factors.
    - **Model Limitations:** While advanced models like Prophet, LSTM, ARIMA, and XGBoost account for trends and seasonality, they do not fully incorporate real-world events, breaking news, macroeconomic shifts, company-specific announcements, or significant shifts in investor sentiment. The future is uncertain.
    - **Accuracy:** Forecasts are based on historical patterns and computational algorithms; they **do not guarantee future performance.** Past performance is not indicative of future results.
    - **Risk:** These forecasts are for informational and educational purposes only and should **NOT be used for actual investment decisions.** Always conduct your own thorough research, consider multiple sources of information, and consult with a qualified financial advisor before making any investment choices.
    """)
    st.markdown("💡 **Tip:** Experiment with different models and 'Days to Forecast' to see how they impact the predictions and performance metrics!")
