import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

@st.cache_data(ttl=3600, show_spinner=False)
def prepare_forecast_data(hist_data, ticker):
    """Prepares historical data for forecasting by computing features."""
    if hist_data.empty:
        return pd.DataFrame(), "No historical data provided."

    df = hist_data.copy()
    required_columns = ['Close']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return pd.DataFrame(), f"Missing required columns: {', '.join(missing_cols)}."

    # Ensure numeric 'Close'
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    if df['Close'].isnull().all():
        return pd.DataFrame(), "All 'Close' values are invalid or missing."

    df['Date_Ordinal'] = df.index.map(lambda x: x.toordinal())
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df.dropna(inplace=True)

    if df.empty or len(df) < 2:
        return pd.DataFrame(), "Not enough data after feature engineering (MA20 calculation)."

    return df, None

@st.cache_data(ttl=3600, show_spinner=False)
def train_forecast_model(X_train, y_train):
    """Trains a Linear Regression model for forecasting."""
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    model_pipeline.fit(X_train, y_train)
    return model_pipeline

def display_forecasting(hist_data, ticker):
    """Main function to display stock price forecasting using Linear Regression."""
    if not ticker or not isinstance(ticker, str):
        st.error("❌ Invalid ticker symbol. Please enter a valid ticker (e.g., 'AAPL').")
        return

    ticker = ticker.strip().upper()
    st.markdown(f"<h3 class='section-title'>Stock Price Forecasting for {ticker}</h3>", unsafe_allow_html=True)

    if hist_data.empty:
        st.warning("⚠️ No historical data provided. Please select a stock and load historical data.")
        return

    st.markdown("<p>This section uses Linear Regression to forecast stock prices based on historical trends. It is a basic model and not suitable for investment decisions.</p>", unsafe_allow_html=True)

    # Prepare data
    with st.spinner("Preparing data for forecasting..."):
        df, error = prepare_forecast_data(hist_data, ticker)
        if error:
            st.warning(f"⚠️ {error} Cannot perform forecasting.")
            return

    # Check data sufficiency
    min_data_points = 50
    if len(df) < min_data_points:
        st.warning(f"⚠️ Not enough valid data points ({len(df)} < {min_data_points}) for forecasting.")
        return

    # Define features and target
    X = df[['Date_Ordinal', 'MA20']]
    y = df['Close']

    # Check for identical values (scaler would fail)
    if y.nunique() <= 1:
        st.warning("⚠️ All closing prices are identical. Cannot train a meaningful model.")
        return

    # Split data chronologically
    train_size = int(len(df) * 0.8)
    if train_size < 1:
        st.warning("⚠️ Not enough data for training set. Cannot train model.")
        return

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train model
    with st.spinner("Training forecasting model..."):
        model = train_forecast_model(X_train.values, y_train.values)

    # Evaluate model
    if not X_test.empty and not y_test.empty:
        y_pred_test = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        st.write(f"**Root Mean Squared Error (RMSE) on Test Set:** `{rmse:.2f}`")
    else:
        st.info("⚠️ Insufficient data for test set. Skipping performance metrics.")
        y_pred_test = np.array([])

    # Forecast future prices
    st.markdown("<h5>Future Price Forecast:</h5>", unsafe_allow_html=True)
    forecast_days = st.slider("Number of days to forecast:", min_value=7, max_value=60, value=30, step=7, key=f"forecast_days_slider_{ticker}")

    last_date_ordinal = df['Date_Ordinal'].iloc[-1]
    last_ma20 = df['MA20'].iloc[-1]

    # Extrapolate MA20 using a simple linear regression on recent MA20 values
    ma20_recent = df['MA20'].tail(20)
    if len(ma20_recent) >= 2:
        ma20_X = np.arange(len(ma20_recent)).reshape(-1, 1)
        ma20_y = ma20_recent.values
        ma20_model = LinearRegression().fit(ma20_X, ma20_y)
        future_steps = np.arange(1, forecast_days + 1).reshape(-1, 1)
        future_ma20 = ma20_model.predict(future_steps)
        future_ma20 = np.maximum(future_ma20, 0)  # Ensure non-negative
    else:
        future_ma20 = [last_ma20] * forecast_days  # Fallback to last value

    future_dates_ordinal = [last_date_ordinal + i for i in range(1, forecast_days + 1)]
    X_future = pd.DataFrame({
        'Date_Ordinal': future_dates_ordinal,
        'MA20': future_ma20
    })

    future_predictions = model.predict(X_future)
    future_dates = [pd.Timestamp.fromordinal(int(date)).date() for date in future_dates_ordinal]

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close': future_predictions.round(2)
    })

    st.dataframe(forecast_df.head(), use_container_width=True)

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Historical Close Price',
        line=dict(color='cyan')
    ))

    if not X_test.empty:
        fig.add_trace(go.Scatter(
            x=df.index[train_size:],
            y=y_pred_test,
            mode='lines',
            name='Predicted (Test Set)',
            line=dict(color='orange', dash='dot')
        ))

    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Predicted Close'],
        mode='lines',
        name=f'Forecasted ({forecast_days} Days)',
        line=dict(color='lime', dash='dash')
    ))

    fig.update_layout(
        title=f'📈 Stock Price Forecast for {ticker}',
        xaxis_title='Date',
        yaxis_title='Close Price',
        template='plotly_dark',
        height=550,
        showlegend=True,
        font=dict(family="Inter", size=12, color="#E0E0E0")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.warning("""
    **Disclaimer for Forecasting:**
    This Linear Regression model is a simplified demonstration.
    - **Accuracy:** Stock prices are influenced by complex factors not captured here.
    - **Limitations:** Assumes linear trends, ignoring non-linear dynamics or external events.
    - **Risk:** **Do NOT use for investment decisions.** Verify with official sources.
    """)
