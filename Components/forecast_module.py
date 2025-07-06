import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from prophet.plot import plot_plotly
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shap
import matplotlib.pyplot as plt
import warnings
import yfinance as yf # Keep yfinance for internal use if needed for some data points, though primary data comes as hist_data

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
    # Avoid division by zero in MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
    return rmse, mae, mape

@st.cache_data(ttl=3600, show_spinner=False)
def prepare_data_for_forecasting(hist_data):
    """Prepares data for various forecasting models."""
    df = hist_data.copy()
    df.index = pd.to_datetime(df.index) # Ensure index is datetime
    df = df[['Close']].rename(columns={'Close': 'y'}) # Prophet needs 'y'
    df['ds'] = df.index # Prophet needs 'ds' for datetime
    return df


# --- ARIMA Model ---
@st.cache_resource(show_spinner=False) # Cache the trained model
def train_arima_model(train_data):
    """Trains an ARIMA model."""
    try:
        # ARIMA order (p,d,q) - typical starting point, can be optimized
        # Using a simple order for speed; hyperparameter tuning could be added.
        model = ARIMA(train_data['y'], order=(5, 1, 0)) # ARIMA(p,d,q)
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        st.error(f"ARIMA model training failed: {e}")
        return None

# --- Prophet Model ---
@st.cache_resource(show_spinner=False) # Cache the trained model
def train_prophet_model(train_data):
    """Trains a Prophet model."""
    # Prophet expects 'ds' and 'y' columns
    m = Prophet(seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05, # Can be tuned
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False) # Daily seasonality often not needed for daily stock data
    m.fit(train_data)
    return m


# --- XGBoost Model (Regression) ---
@st.cache_resource(show_spinner=False) # Cache the trained model
def train_xgboost_model(df, days_to_forecast):
    """Prepares data and trains an XGBoost Regressor model."""
    df_xgb = df.copy()
    # Create lag features
    for i in range(1, 11): # Lag features for past 10 days
        df_xgb[f'lag_{i}'] = df_xgb['y'].shift(i)
    # Moving averages
    df_xgb['rolling_mean_5'] = df_xgb['y'].rolling(window=5).mean()
    df_xgb['rolling_std_5'] = df_xgb['y'].rolling(window=5).std()

    df_xgb.dropna(inplace=True)

    # Use 'y' column as the target for XGBoost
    features = [col for col in df_xgb.columns if col not in ['ds', 'y']]
    X = df_xgb[features]
    y = df_xgb['y']

    if X.empty or len(X) < 2:
        return None, None, None, None # Return model, features used, X_test, y_test

    # Split data: Use the last `days_to_forecast` as test data, rest as train
    # Ensure there's enough data for both train and test
    if len(X) <= days_to_forecast:
        st.warning(f"Not enough data for XGBoost (data points: {len(X)}, forecast days: {days_to_forecast}). Skipping XGBoost.")
        return None, None, None, None

    train_size = len(X) - days_to_forecast
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    model = xgb.XGBRegressor(objective='reg:squarederror',
                             n_estimators=100,
                             learning_rate=0.1,
                             random_state=42,
                             n_jobs=-1,
                             enable_categorical=True if pd.api.types.is_categorical_dtype(X_train) else False) # For potential categorical features

    model.fit(X_train, y_train)

    # Store feature names used for SHAP later
    model.feature_names_in_ = features # Explicitly set feature names

    return model, features, X_test, y_test


# --- LSTM Model (Regression) ---
class LSTMRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegressionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Take the output from the last time step
        return out

@st.cache_resource(show_spinner=False)
def prepare_lstm_data_regression(df, sequence_length=20):
    """Prepares data for LSTM regression."""
    data = df['y'].values.reshape(-1, 1) # Reshape for scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i : (i + sequence_length), 0])
        y.append(scaled_data[i + sequence_length, 0])
    return np.array(X), np.array(y), scaler

@st.cache_resource(show_spinner=False)
def train_lstm_model_regression(X_train, y_train, input_size, hidden_size=50, num_layers=2, epochs=50, batch_size=32):
    """Trains an LSTM regression model."""
    if X_train.shape[0] == 0:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device) # Add feature dimension
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LSTMRegressionModel(input_size, hidden_size, num_layers, 1).to(device)
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

    if historical_data_plot is not None:
        fig.add_trace(go.Scatter(x=historical_data_plot.index, y=historical_data_plot['Close'],
                                 mode='lines', name='Historical Close',
                                 line=dict(color='lightgray')))

    fig.add_trace(go.Scatter(x=actual.index, y=actual, mode='lines', name='Actual Future Prices',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name=f'{model_name} Forecast',
                             line=dict(color='red', dash='dash')))

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
        return None

    try:
        # Determine explainer based on model type
        if isinstance(model, (xgb.XGBRegressor, xgb.XGBClassifier)): # More specific check
            explainer = shap.TreeExplainer(model)
        else: # Fallback to KernelExplainer for other models
            # KernelExplainer needs a background dataset; use a sample of X_data_for_shap
            background_data = X_data_for_shap.sample(min(100, len(X_data_for_shap)), random_state=42)
            explainer = shap.KernelExplainer(model.predict, background_data) # Use .predict for regression

        shap_values = explainer.shap_values(X_data_for_shap)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_data_for_shap, plot_type="bar", show=False, ax=ax,
                          color_bar=True, cmap='coolwarm', max_display=10) # Added cmap and max_display
        ax.set_title(f'SHAP Feature Importance - {model_name}', fontsize=14, color='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig) # Close the figure to free memory
        return True
    except Exception as e:
        st.warning(f"Could not generate SHAP plot for {model_name}: {e}")
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

    if df_forecast.empty:
        st.error("❌ Data preparation for forecasting failed. Please check the input historical data.")
        return

    # Split data into train and actual future for evaluation
    # Ensure train_size is at least 1 for valid split
    if len(df_forecast) <= days_to_forecast:
        st.error(f"❌ Not enough historical data ({len(df_forecast)} days) to forecast {days_to_forecast} days. Please select a longer historical range.")
        return

    train_data = df_forecast.iloc[:-days_to_forecast]
    actual_future_data = df_forecast.iloc[-days_to_forecast:]

    if train_data.empty:
        st.error("❌ Training data is empty after splitting. Please ensure you have sufficient historical data.")
        return

    forecast_results = None
    model_performance = {}
    current_model = None
    XGB_features = None # To store features used by XGBoost for SHAP
    XGB_X_test = None
    XGB_y_test = None


    with st.spinner(f"Running {selected_model} Model... 🚀"):
        try:
            if selected_model == "ARIMA":
                model_arima = train_arima_model(train_data)
                if model_arima:
                    # Forecast and index
                    forecast_period_index = pd.date_range(start=train_data.index[-1] + pd.Timedelta(days=1),
                                                         periods=days_to_forecast, freq='D')
                    arima_forecast_series = model_arima.forecast(steps=days_to_forecast)
                    arima_forecast_series.index = forecast_period_index
                    forecast_results = arima_forecast_series
                    current_model = model_arima
                else:
                    st.warning("ARIMA model could not be trained.")

            elif selected_model == "Prophet":
                m_prophet = train_prophet_model(train_data)
                if m_prophet:
                    future = m_prophet.make_future_dataframe(periods=days_to_forecast)
                    forecast_prophet = m_prophet.predict(future)
                    # Filter forecast to only include the future period
                    forecast_results = forecast_prophet[['ds', 'yhat']].set_index('ds').loc[actual_future_data.index]['yhat']
                    current_model = m_prophet
                else:
                    st.warning("Prophet model could not be trained.")


            elif selected_model == "XGBoost (Regression)":
                # XGBoost requires features
                xgb_model, features, X_test_xgb, y_test_xgb = train_xgboost_model(df_forecast, days_to_forecast)
                if xgb_model and features and not X_test_xgb.empty:
                    # Make predictions on the test set for evaluation
                    xgb_predictions = xgb_model.predict(X_test_xgb)
                    forecast_results = pd.Series(xgb_predictions, index=y_test_xgb.index, name='yhat')
                    current_model = xgb_model
                    XGB_features = features
                    XGB_X_test = X_test_xgb
                    XGB_y_test = y_test_xgb # Keep for SHAP if needed on a different data slice
                else:
                    st.warning("XGBoost model could not be trained or insufficient data for testing.")

            elif selected_model == "LSTM (Regression)":
                sequence_length = 20 # You can make this configurable
                X_lstm, y_lstm, scaler_lstm = prepare_lstm_data_regression(df_forecast)

                if X_lstm.shape[0] == 0:
                    st.warning("Not enough data for LSTM model after preparing sequences. Skipping LSTM training.")
                    lstm_model = None
                else:
                    # Split for training and testing LSTM
                    train_size_lstm = len(X_lstm) - days_to_forecast
                    if train_size_lstm <= 0: # Ensure at least one training point
                        st.warning("Not enough data for LSTM training after sequence preparation. Skipping LSTM.")
                        lstm_model = None
                    else:
                        X_train_lstm, X_test_lstm = X_lstm[:train_size_lstm], X_lstm[train_size_lstm:]
                        y_train_lstm, y_test_lstm = y_lstm[:train_size_lstm], y_lstm[train_size_lstm:]

                        if X_train_lstm.shape[0] == 0 or X_test_lstm.shape[0] == 0:
                             st.warning("Not enough data to split into train/test sets for LSTM. Skipping LSTM training.")
                             lstm_model = None
                        else:
                            lstm_model = train_lstm_model_regression(X_train_lstm, y_train_lstm, X_train_lstm.shape[1])

                if lstm_model:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # Make predictions on the test set
                    X_test_lstm_tensor = torch.tensor(X_test_lstm, dtype=torch.float32).unsqueeze(-1).to(device)
                    lstm_model.eval()
                    with torch.no_grad():
                        lstm_predictions_scaled = lstm_model(X_test_lstm_tensor).cpu().numpy()

                    # Inverse transform to get actual prices
                    lstm_predictions = scaler_lstm.inverse_transform(lstm_predictions_scaled)
                    forecast_results = pd.Series(lstm_predictions.flatten(), index=actual_future_data.index, name='yhat')
                    current_model = lstm_model
                else:
                    st.warning("LSTM model could not be trained.")

        except Exception as e:
            st.error(f"Error during {selected_model} model execution: {e}")

    if forecast_results is not None and not forecast_results.empty:
        st.success(f"✅ {selected_model} Model ran successfully!")
        rmse, mae, mape = calculate_metrics(actual_future_data['y'], forecast_results)
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
        st.plotly_chart(plot_forecast(actual_future_data['y'], forecast_results, selected_model, hist_data), use_container_width=True)

        st.markdown("<h4>🔮 Future Price Forecast:</h4>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(forecast_results).rename(columns={'yhat': 'Predicted Close Price'}).style.format({"Predicted Close Price": "{:.2f}"}))

    else:
        st.warning(f"No forecast generated for {selected_model}. Please check configuration or data.")


    # --- Model Explainability (SHAP) ---
    st.markdown("---")
    st.markdown("<h4 class='section-subtitle'>🔍 Model Explainability (SHAP)</h4>", unsafe_allow_html=True)
    st.info("SHAP values indicate how much each feature contributes to the model's output (positive or negative).")

    shap_success = False
    if selected_model == "XGBoost (Regression)" and current_model and XGB_X_test is not None:
        # Use a small sample of the test data for SHAP explanation for performance
        shap_X_explain = XGB_X_test.tail(min(len(XGB_X_test), 200)) # Use features model was trained on
        shap_success = plot_shap_summary(current_model, shap_X_explain, "XGBoost (Regression)")
    elif selected_model == "Prophet" and current_model:
        st.info("SHAP explanations are not directly applicable to Prophet as it's a statistical model focused on trend, seasonality, and holidays, not feature importance in the same way as ML models.")
        shap_success = True # Consider it 'successful' in that it provided information
    elif selected_model == "ARIMA" and current_model:
        st.info("ARIMA models are statistical and do not have 'features' in the same sense as machine learning models, so SHAP explanations are not applicable.")
        shap_success = True
    elif selected_model == "LSTM (Regression)" and current_model:
        st.info("SHAP explanations for LSTM models are generally more complex and often require specialized libraries (e.g., DeepLift, Integrated Gradients) which are outside the scope of this simplified SHAP implementation.")
        shap_success = True # Indicate as 'successful' in that it provided information

    if not shap_success and selected_model not in ["Prophet", "ARIMA", "LSTM (Regression)"]: # Only show if it was expected to work and failed
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
