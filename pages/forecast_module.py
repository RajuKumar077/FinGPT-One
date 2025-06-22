import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# This module uses the historical data passed to it directly,
# so no API key is needed here.

def display_forecasting(hist_data):
    st.markdown("<h3 class='section-title'>Stock Price Forecasting (Linear Regression)</h3>", unsafe_allow_html=True)

    if hist_data.empty:
        st.warning("Please select a stock and load historical data to perform forecasting.")
        return

    st.markdown("<p>This section uses a simple Linear Regression model to forecast future stock prices based on historical trends. This is a basic model and should not be used for actual investment decisions.</p>", unsafe_allow_html=True)

    df = hist_data.copy()
    
    # Ensure 'Close' is numeric and handle potential missing values from previous steps
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True) # Drop rows where 'Close' is NaN

    if df.empty or len(df) < 50: # Need sufficient data for forecasting
        st.warning("Not enough valid historical data points for forecasting. At least 50 data points with valid closing prices are recommended.")
        return

    # Create numerical representation of Date (e.g., days since first date)
    df['Date_Ordinal'] = pd.to_datetime(df['Date']).apply(lambda date: date.toordinal())
    
    # Use a rolling mean as a simple feature, which often helps smooth out noise
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df.dropna(inplace=True) # Drop NaNs created by rolling mean

    if df.empty or len(df) < 2:
        st.warning("Not enough data after feature engineering for forecasting. Ensure sufficient historical data and check for missing values.")
        return

    # Define features (X) and target (y)
    # Using 'Date_Ordinal' and 'MA20' as features
    X = df[['Date_Ordinal', 'MA20']]
    y = df['Close']

    # Split data into training and testing sets
    # Using 80% for training, 20% for testing
    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create a pipeline with scaling and linear regression
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_test = model_pipeline.predict(X_test)
    
    # Calculate RMSE on the test set
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    st.write(f"**Root Mean Squared Error (RMSE) on Test Set:** `{rmse:.2f}`")

    # --- Forecasting Future Dates ---
    st.markdown("<h5>Future Price Forecast:</h5>", unsafe_allow_html=True)
    forecast_days = st.slider("Number of days to forecast:", min_value=7, max_value=60, value=30, step=7)

    last_date_ordinal = df['Date_Ordinal'].iloc[-1]
    last_ma20 = df['MA20'].iloc[-1]

    future_dates_ordinal = [last_date_ordinal + i for i in range(1, forecast_days + 1)]
    
    # A very simple extrapolation for MA20 for future dates
    # In a real scenario, this would be more complex or use a more advanced forecasting model.
    future_ma20 = [last_ma20 + (df['MA20'].diff().mean() * i) for i in range(1, forecast_days + 1)]
    
    # Create future feature set
    X_future = pd.DataFrame({
        'Date_Ordinal': future_dates_ordinal,
        'MA20': future_ma20
    })

    # Predict future prices
    future_predictions = model_pipeline.predict(X_future)
    future_dates = [pd.to_datetime(date, unit='D', origin='julian').date() for date in future_dates_ordinal]

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Close': future_predictions
    })

    st.dataframe(forecast_df.head()) # Display first few forecasted days

    # --- Plotting Historical Data and Forecast ---
    fig = go.Figure()

    # Historical Close Price
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df['Date']),
        y=df['Close'],
        mode='lines',
        name='Historical Close Price',
        line=dict(color='cyan')
    ))

    # Predicted values on test set
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(df['Date'][train_size:]),
        y=y_pred_test,
        mode='lines',
        name='Predicted (Test Set)',
        line=dict(color='orange', dash='dot')
    ))

    # Future Forecast
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(forecast_df['Date']),
        y=forecast_df['Predicted Close'],
        mode='lines',
        name=f'Forecasted ({forecast_days} Days)',
        line=dict(color='lime', dash='dash')
    ))

    fig.update_layout(
        title=f'📈 Stock Price Forecast for {hist_data.name}',
        xaxis_title='Date',
        yaxis_title='Close Price',
        template='plotly_dark',
        height=550,
        showlegend=True,
        font=dict(family="Inter", size=12, color="#E0E0E0")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.warning("Forecasting models like this are simplifications. Real-world stock price prediction is highly complex and influenced by numerous factors not included here. Do not use this for actual financial decisions.")
