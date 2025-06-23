import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

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

    if df.empty or len(df) < 50: # Need sufficient data for forecasting, e.g., for MA20
        st.warning("Not enough valid historical data points (at least 50 with valid 'Close' prices) for forecasting. Please try a ticker with a longer history.")
        return

    # Create numerical representation of Date (e.g., days since first date)
    df['Date_Ordinal'] = pd.to_datetime(df['Date']).apply(lambda date: date.toordinal())
    
    # Use a rolling mean as a simple feature, which often helps smooth out noise
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df.dropna(inplace=True) # Drop NaNs created by rolling mean (first 19 rows for MA20)

    if df.empty or len(df) < 2: # After dropping NaNs from MA20 calculation, still need enough data
        st.warning("Not enough data after feature engineering (MA20 calculation) for forecasting. Ensure sufficient historical data.")
        return

    # Define features (X) and target (y)
    X = df[['Date_Ordinal', 'MA20']]
    y = df['Close']

    # Split data into training and testing sets (chronologically)
    train_size = int(len(df) * 0.8) # 80% for training
    
    # Ensure train and test sets have at least 1 sample
    if train_size < 1:
        st.warning("Not enough data to create a training set. Cannot train the forecasting model.")
        return
    if len(df) - train_size < 1:
        st.warning("Not enough data to create a test set. Forecasting will proceed, but accuracy metrics will not be displayed.")
        
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create a pipeline with scaling and linear regression
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # Train the model
    with st.spinner("Training forecasting model..."):
        model_pipeline.fit(X_train, y_train)

    # Make predictions on the test set and calculate RMSE, only if test set exists
    if not X_test.empty and not y_test.empty:
        y_pred_test = model_pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        st.write(f"**Root Mean Squared Error (RMSE) on Test Set:** `{rmse:.2f}`")
    else:
        st.info("Model performance metrics (RMSE) skipped due to insufficient data for a test set.")
        y_pred_test = np.array([]) # Ensure it's an empty array if no test data

    # --- Forecasting Future Dates ---
    st.markdown("<h5>Future Price Forecast:</h5>", unsafe_allow_html=True)
    forecast_days = st.slider("Number of days to forecast:", min_value=7, max_value=60, value=30, step=7, key=f"forecast_days_slider_{hist_data.name}")

    last_date_ordinal = df['Date_Ordinal'].iloc[-1]
    last_ma20 = df['MA20'].iloc[-1]

    future_dates_ordinal = [last_date_ordinal + i for i in range(1, forecast_days + 1)]
    
    # Simple linear extrapolation for MA20 for future dates.
    # A more sophisticated model would use a separate time series forecast for MA20.
    ma20_diff_mean = df['MA20'].diff().mean()
    if pd.isna(ma20_diff_mean): # Handle case where MA20 has no diff (e.g., too few points after MA20 calc)
        ma20_diff_mean = 0 

    future_ma20 = [last_ma20 + (ma20_diff_mean * i) for i in range(1, forecast_days + 1)]
    
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

    # Predicted values on test set (only if test set was not empty)
    if not X_test.empty:
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

    st.warning("""
    **Disclaimer for Forecasting:**
    This forecasting model is a simplified demonstration using Linear Regression on historical data.
    - **Accuracy:** Stock price prediction is extremely complex and influenced by innumerable factors (economic, geopolitical, company-specific news, market sentiment, etc.) that are not captured here.
    - **Limitations:** This model does not account for non-linear relationships, sudden market shifts, or external shocks.
    - **Risk:** **Do NOT use this forecast for real investment decisions.** It is for illustrative and educational purposes only.
    """)
