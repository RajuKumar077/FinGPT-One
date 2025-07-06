import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set Matplotlib style for consistency with Streamlit dark theme
plt.style.use('dark_background')
plt.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white', 'xtick.color': 'white', 'ytick.color': 'white'})


# --- Utility Functions (Cached) ---
@st.cache_data(ttl=3600, show_spinner=False)
def calculate_metrics(y_true, y_pred):
    """Calculates RMSE, MAE, MAPE. Ensures inputs are finite."""
    # Ensure no NaNs or Infs before calculation for sklearn compatibility
    # Drop NaNs from both Series based on their combined index
    combined_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).dropna()
    y_true_clean = combined_df['y_true']
    y_pred_clean = combined_df['y_pred']

    if y_true_clean.empty:
        return float('inf'), float('inf'), float('inf') # No valid data to calculate metrics

    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)

    # Avoid division by zero in MAPE for 0 or negative actual values, handle NaN/Inf
    mape_denominator = y_true_clean[y_true_clean != 0]
    mape_numerator = np.abs(y_true_clean[y_true_clean != 0] - y_pred_clean[y_true_clean != 0])

    if not mape_denominator.empty:
        mape = np.mean(mape_numerator / mape_denominator) * 100
    else:
        mape = float('inf') # If all true values are 0, MAPE is undefined / infinite

    return rmse, mae, mape

@st.cache_data(ttl=3600, show_spinner=False)
def prepare_data_for_forecasting(hist_data):
    """Prepares data for various forecasting models."""
    df = hist_data.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)

    df_prophet_format = df[['Close']].rename(columns={'Close': 'y'})
    df_prophet_format['ds'] = df_prophet_format.index
    return df_prophet_format


# --- ARIMA Model ---
@st.cache_resource(show_spinner=False)
def train_and_forecast_arima(train_data_series, days_to_forecast, order=(5, 1, 0)):
    """Trains ARIMA on full data and forecasts future."""
    try:
        model = ARIMA(train_data_series, order=order)
        model_fit = model.fit()

        last_date = train_data_series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=days_to_forecast, freq='D')
        forecast_array = model_fit.forecast(steps=days_to_forecast)
        forecast_series = pd.Series(forecast_array, index=future_dates, name='yhat')
        return model_fit, forecast_series
    except Exception as e:
        st.error(f"ARIMA model training or forecasting failed: {e}")
        return None, None

# --- Prophet Model ---
@st.cache_resource(show_spinner=False)
def train_and_forecast_prophet(train_data_df, days_to_forecast):
    """Trains Prophet on full data and forecasts future."""
    try:
        m = Prophet(seasonality_mode='multiplicative',
                    changepoint_prior_scale=0.05,
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False)
        m.fit(train_data_df)

        future = m.make_future_dataframe(periods=days_to_forecast, include_history=False)
        if future.empty:
            return None, None # No future dates generated
        forecast_prophet_df = m.predict(future)
        # Only take the future predictions
        forecast_results = forecast_prophet_df[['ds', 'yhat']].set_index('ds')['yhat']
        return m, forecast_results
    except Exception as e:
        st.error(f"Prophet model training or forecasting failed: {e}")
        return None, None


# --- XGBoost Model (Regression) ---
@st.cache_resource(show_spinner=False)
def train_xgboost_for_future(df_full):
    """Trains XGBoost model on full historical data for future prediction."""
    df_xgb = df_full.copy()
    
    # Create lag features for 'y' (Close price)
    for i in range(1, 11): # Lag features for past 10 days
        df_xgb[f'lag_{i}'] = df_xgb['y'].shift(i)
    # Moving averages
    df_xgb['rolling_mean_5'] = df_xgb['y'].rolling(window=5).mean()
    df_xgb['rolling_std_5'] = df_xgb['y'].rolling(window=5).std()

    # Drop rows with NaN created by lag/rolling features
    df_xgb.dropna(inplace=True)

    features = [col for col in df_xgb.columns if col not in ['ds', 'y']]
    X = df_xgb[features]
    y = df_xgb['y']

    if X.empty or len(X) < 10: # Ensure enough data for lags + training
        return None, None # Not enough data for feature engineering

    model = xgb.XGBRegressor(objective='reg:squarederror',
                             n_estimators=200,
                             learning_rate=0.05,
                             random_state=42,
                             n_jobs=-1)
    model.fit(X, y) # Train on ALL historical data

    return model, features

def forecast_xgboost_future(model, features, df_historical, days_to_forecast):
    """
    Iteratively forecasts future values using a trained XGBoost model.
    Requires 'y' column (Close price) in df_historical.
    """
    if model is None or features is None or df_historical.empty:
        return None

    # Start with a copy of the last part of historical data needed for features
    # This must include enough data to create the lags for the first future prediction
    forecast_df = df_historical.copy()
    
    future_predictions = []
    last_date = forecast_df.index[-1]

    for i in range(days_to_forecast):
        next_date = last_date + pd.Timedelta(days=1)
        
        # Create a temporary row for the next prediction with placeholder for 'y'
        new_row = pd.DataFrame({'y': [np.nan]}, index=[next_date])
        
        # Concatenate to generate features
        forecast_df = pd.concat([forecast_df, new_row])

        # Recalculate features for the *entire* extended dataframe
        # This is inefficient but simple for demonstration. For production, optimize.
        temp_df_for_features = forecast_df.copy()
        for j in range(1, 11):
            temp_df_for_features[f'lag_{j}'] = temp_df_for_features['y'].shift(j)
        temp_df_for_features['rolling_mean_5'] = temp_df_for_features['y'].rolling(window=5).mean()
        temp_df_for_features['rolling_std_5'] = temp_df_for_features['y'].rolling(window=5).std()

        # Get features for the last (newest) row (the one we want to predict)
        # Ensure it has finite values for features
        X_predict = temp_df_for_features[features].iloc[[-1]].dropna()

        if X_predict.empty:
            st.warning(f"Could not generate features for {next_date}. Stopping XGBoost future forecast early.")
            break
        
        # Predict the next value
        predicted_value = model.predict(X_predict)[0]
        future_predictions.append(predicted_value)
        
        # Update the 'y' column in the forecast_df with the predicted value
        forecast_df.loc[next_date, 'y'] = predicted_value
        last_date = next_date

    return pd.Series(future_predictions, index=pd.date_range(start=df_historical.index[-1] + pd.Timedelta(days=1), periods=len(future_predictions), freq='D'), name='yhat')


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
        out = self.fc(out[:, -1, :])
        return out

@st.cache_resource(show_spinner=False)
def prepare_lstm_data_for_training(df_full, sequence_length=20):
    """
    Prepares data for LSTM regression training (on full historical data).
    Returns X (sequences), y (next value), and the scaler.
    """
    data = df_full['y'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i : (i + sequence_length), 0])
        y.append(scaled_data[i + sequence_length, 0])

    return np.array(X), np.array(y), scaler

@st.cache_resource(show_spinner=False)
def train_lstm_model(X_train, y_train, sequence_length, hidden_size=50, num_layers=2, epochs=50, batch_size=32):
    """Trains an LSTM regression model."""
    if X_train.shape[0] == 0:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

def forecast_lstm_future(model, scaler, df_historical, days_to_forecast, sequence_length=20):
    """
    Iteratively forecasts future values using a trained LSTM model.
    """
    if model is None or scaler is None or df_historical.empty:
        return None

    data = df_historical['y'].values.reshape(-1, 1)
    scaled_data = scaler.transform(data) # Use transform, not fit_transform

    # Get the last sequence from the historical data
    current_sequence = list(scaled_data[-sequence_length:].flatten())

    future_predictions_scaled = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval() # Set model to evaluation mode

    with torch.no_grad():
        for _ in range(days_to_forecast):
            # Convert the current sequence to a tensor
            input_tensor = torch.tensor(current_sequence[-sequence_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            
            predicted_value_scaled = model(input_tensor).cpu().numpy()[0][0]
            future_predictions_scaled.append(predicted_value_scaled)
            
            # Update the sequence: remove the oldest value, add the new prediction
            current_sequence.append(predicted_value_scaled)
            current_sequence = current_sequence[1:] # Keep only the last 'sequence_length' elements

    # Generate future dates
    last_hist_date = df_historical.index[-1]
    future_dates = pd.date_range(start=last_hist_date + pd.Timedelta(days=1), periods=days_to_forecast, freq='D')

    # Inverse transform the scaled predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    return pd.Series(future_predictions.flatten(), index=future_dates, name='yhat')


# --- Plotting Functions ---
def plot_forecast(historical_close, future_forecast, model_name):
    """Plots historical close and future forecast using Plotly."""
    fig = go.Figure()

    # Plot historical data
    fig.add_trace(go.Scatter(x=historical_close.index, y=historical_close.values,
                             mode='lines', name='Historical Close',
                             line=dict(color='lightgray')))

    # Plot future forecast
    if future_forecast is not None and not future_forecast.empty:
        fig.add_trace(go.Scatter(x=future_forecast.index, y=future_forecast.values,
                                 mode='lines', name=f'{model_name} Future Forecast',
                                 line=dict(color='red', dash='dash', width=2)))

    fig.update_layout(
        title=f'Stock Price Future Forecast ({model_name})',
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
        if isinstance(model, (xgb.XGBRegressor, xgb.XGBClassifier)):
            explainer = shap.TreeExplainer(model)
        else:
            background_data = X_data_for_shap.sample(min(50, len(X_data_for_shap)), random_state=42)
            explainer = shap.KernelExplainer(model.predict, background_data)

        shap_values = explainer.shap_values(X_data_for_shap.sample(min(200, len(X_data_for_shap)), random_state=42))

        fig, ax = plt.subplots(figsize=(10, 6))
        if isinstance(shap_values, list):
             shap.summary_plot(shap_values[0], X_data_for_shap.columns, plot_type="bar", show=False, ax=ax,
                              color_bar=True, cmap='coolwarm', max_display=10)
        else:
             shap.summary_plot(shap_values, X_data_for_shap.columns, plot_type="bar", show=False, ax=ax,
                              color_bar=True, cmap='coolwarm', max_display=10)

        ax.set_title(f'SHAP Feature Importance - {model_name}', fontsize=14, color='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
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
        <p>The models are trained on all available historical data to make predictions for the next N days into the future.</p>
        """, unsafe_allow_html=True)

    if hist_data is None or hist_data.empty:
        st.info("📊 Please select a stock and date range in the sidebar to load historical data for forecasting.")
        return

    # User Configuration
    st.subheader("⚙️ Forecasting Configuration")
    days_to_forecast = st.slider("Days to Forecast (into the future)", min_value=5, max_value=60, value=30)
    selected_model = st.selectbox(
        "Choose Forecasting Model",
        ["Prophet", "XGBoost (Regression)", "LSTM (Regression)", "ARIMA"]
    )
    st.markdown("---")

    # Prepare data (cached for efficiency)
    with st.spinner("Preparing data for forecasting... ⏳"):
        df_forecast = prepare_data_for_forecasting(hist_data)
        df_forecast = df_forecast[~df_forecast.index.duplicated(keep='first')].sort_index()

    if df_forecast.empty:
        st.error("❌ Data preparation for forecasting failed. Please check the input historical data.")
        return

    # Ensure enough data for training, especially for models with lags/sequences
    min_data_points = max(10, days_to_forecast + 1) # Arbitrary min for training and lags
    if len(df_forecast) < min_data_points:
        st.error(f"❌ Not enough historical data ({len(df_forecast)} days) for robust model training and forecasting. Please select a longer historical range (min {min_data_points} days needed).")
        return

    # Train on ALL historical data for future prediction
    train_data_for_future_pred = df_forecast.copy() # All available data
    
    future_forecast_results = None
    current_model = None
    XGB_X_for_shap = None # Data to be used for SHAP explanation

    with st.spinner(f"Running {selected_model} Model for future prediction... 🚀"):
        try:
            if selected_model == "ARIMA":
                model_arima, future_forecast_results = train_and_forecast_arima(train_data_for_future_pred['y'], days_to_forecast)
                current_model = model_arima

            elif selected_model == "Prophet":
                m_prophet, future_forecast_results = train_and_forecast_prophet(train_data_for_future_pred, days_to_forecast)
                current_model = m_prophet

            elif selected_model == "XGBoost (Regression)":
                xgb_model, features = train_xgboost_for_future(train_data_for_future_pred)
                if xgb_model and features:
                    future_forecast_results = forecast_xgboost_future(xgb_model, features, train_data_for_future_pred, days_to_forecast)
                    current_model = xgb_model
                    # For SHAP, use a sample of the data used for training
                    # Need to regenerate features for the full historical df for SHAP
                    df_xgb_for_shap = train_data_for_future_pred.copy()
                    for i in range(1, 11):
                        df_xgb_for_shap[f'lag_{i}'] = df_xgb_for_shap['y'].shift(i)
                    df_xgb_for_shap['rolling_mean_5'] = df_xgb_for_shap['y'].rolling(window=5).mean()
                    df_xgb_for_shap['rolling_std_5'] = df_xgb_for_shap['y'].rolling(window=5).std()
                    XGB_X_for_shap = df_xgb_for_shap[features].dropna() # Features for SHAP

                else:
                    st.warning("XGBoost model could not be trained or insufficient data.")

            elif selected_model == "LSTM (Regression)":
                sequence_length = 20 # Can make this configurable
                X_lstm_train, y_lstm_train, scaler_lstm = prepare_lstm_data_for_training(train_data_for_future_pred, sequence_length)

                if X_lstm_train.shape[0] == 0:
                    st.warning("Not enough data for LSTM model after preparing sequences. Skipping LSTM training.")
                    lstm_model = None
                else:
                    lstm_model = train_lstm_model(X_lstm_train, y_lstm_train, sequence_length)

                if lstm_model:
                    future_forecast_results = forecast_lstm_future(lstm_model, scaler_lstm, train_data_for_future_pred, days_to_forecast, sequence_length)
                    current_model = lstm_model
                else:
                    st.warning("LSTM model could not be trained.")

        except Exception as e:
            st.error(f"Error during {selected_model} model execution: {e}")
            import traceback
            st.code(traceback.format_exc()) # Display full traceback for debugging

    if future_forecast_results is not None and not future_forecast_results.empty:
        st.success(f"✅ {selected_model} Model successfully generated future forecast!")

        # Calculate metrics (on a held-out set if you still want to show performance)
        # For true future forecasting, we don't have 'actual_future_data' to compare with.
        # So we omit metrics for the true future forecast, or calculate on a historical test set if desired.
        # For simplicity here, we'll only show the future forecast.
        st.markdown("<h4>📈 Future Price Forecast Visualization:</h4>", unsafe_allow_html=True)
        st.plotly_chart(plot_forecast(hist_data['Close'], future_forecast_results, selected_model), use_container_width=True)

        st.markdown("<h4>🔮 Predicted Future Prices:</h4>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(future_forecast_results).rename(columns={'yhat': 'Predicted Close Price'}).style.format({"Predicted Close Price": "{:.2f}"}))

    else:
        st.warning(f"No future forecast generated for {selected_model}. Please check configuration or data.")


    # --- Model Explainability (SHAP) ---
    st.markdown("---")
    st.markdown("<h4 class='section-subtitle'>🔍 Model Explainability (SHAP)</h4>", unsafe_allow_html=True)
    st.info("SHAP values indicate how much each feature contributes to the model's output (positive or negative).")

    shap_success = False
    if selected_model == "XGBoost (Regression)" and current_model and XGB_X_for_shap is not None and not XGB_X_for_shap.empty:
        shap_X_explain = XGB_X_for_shap.tail(min(len(XGB_X_for_shap), 200)).copy()
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
