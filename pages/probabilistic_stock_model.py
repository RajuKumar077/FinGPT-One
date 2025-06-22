import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# This module uses the historical data passed to it directly,
# so no API key is needed here.

def display_probabilistic_models(hist_data):
    st.markdown("<h3 class='section-title'>Probabilistic Stock Models (Daily Prediction)</h3>", unsafe_allow_html=True)

    if hist_data.empty:
        st.warning("Please select a stock and load historical data to run probabilistic models.")
        return

    st.markdown("<p>This section uses a Random Forest Classifier to predict whether the stock's closing price will go up (1) or down (0) the next day, based on historical features.</p>", unsafe_allow_html=True)

    # --- Feature Engineering (consistent with app.py's add_common_features) ---
    df = hist_data.copy()
    
    # Ensure numeric columns are actually numeric
    for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Return_1d'] = df['Close'].pct_change()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['RSI'] = compute_rsi(df['Close'], 14) # Using local RSI
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['High_Low_Diff'] = (df['High'] - df['Low']) / df['Close']
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # 1 = up, 0 = down
    df.dropna(inplace=True)

    if df.empty:
        st.warning("Not enough data to compute features for probabilistic models after cleaning. Please check data range and quality.")
        return

    # --- Model Training ---
    st.markdown("<h5>Model Training & Performance:</h5>", unsafe_allow_html=True)

    features = ['Return_1d', 'MA10', 'MA50', 'Volatility', 'RSI', 'Volume_MA5', 'High_Low_Diff']
    
    # Check if all features exist after dropping NaNs
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.error(f"Error: Missing required features for the model: {', '.join(missing_features)}. This might be due to insufficient historical data or data quality issues.")
        return

    X = df[features]
    y = df['Target']

    if len(X) < 2: # Need at least 2 samples for train_test_split
        st.info("Not enough data points to train the model after feature engineering. At least 2 data points with all features are required.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of going up

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Model Accuracy (on test set):** `{accuracy:.2f}`")
    st.markdown("<h6>Classification Report:</h6>", unsafe_allow_html=True)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    st.json(report_dict)

    # --- Prediction for Next Day ---
    st.markdown("<h5>Tomorrow's Prediction:</h5>", unsafe_allow_html=True)
    last_data_point = df.iloc[-1][features].values.reshape(1, -1)
    
    # Handle potential NaN in last_data_point if the last row still has some due to feature calculation
    if np.isnan(last_data_point).any():
        st.warning("Cannot make a prediction for tomorrow: The latest data point has missing or invalid values after feature calculation.")
        return

    next_day_prediction = model.predict(last_data_point)[0]
    next_day_proba_up = model.predict_proba(last_data_point)[0, 1]

    prediction_text = "UP 🟢" if next_day_prediction == 1 else "DOWN 🔴"
    st.markdown(f"**Predicted Movement for Next Trading Day:** `{prediction_text}`")
    st.markdown(f"**Probability of Price Going Up:** `{next_day_proba_up:.2%}`")

    # --- Feature Importance Visualization ---
    st.markdown("<h5>Feature Importances:</h5>", unsafe_allow_html=True)
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    
    fig_importance = go.Figure(go.Bar(
        x=feature_importances.values,
        y=feature_importances.index,
        orientation='h',
        marker_color='skyblue'
    ))
    fig_importance.update_layout(
        title='📊 Feature Importance in Prediction Model',
        xaxis_title='Importance',
        yaxis_title='Feature',
        template='plotly_dark',
        height=400,
        font=dict(family="Inter", size=12, color="#E0E0E0")
    )
    st.plotly_chart(fig_importance, use_container_width=True)

    # --- Probability vs. Actuals Plot (Mini-chart for recent performance) ---
    st.markdown("<h5>Recent Model Performance Visualization:</h5>", unsafe_allow_html=True)

    # Take a smaller subset for better visualization if data is very long
    recent_data = df.tail(100).copy()
    if not recent_data.empty:
        X_recent = recent_data[features]
        y_recent_actual = recent_data['Target']
        y_recent_proba = model.predict_proba(X_recent)[:, 1]

        fig_proba = make_subplots(specs=[[{"secondary_y": True}]])

        fig_proba.add_trace(go.Scatter(x=recent_data['Date'], y=recent_data['Close'], name='Close Price', line=dict(color='cyan')), secondary_y=False)
        fig_proba.add_trace(go.Scatter(x=recent_data['Date'], y=y_recent_proba, name='Predicted Probability (Up)', line=dict(color='orange', dash='dash')), secondary_y=True)
        fig_proba.add_trace(go.Scatter(x=recent_data['Date'], y=y_recent_actual, mode='markers', name='Actual Direction (1=Up, 0=Down)',
                                       marker=dict(symbol='circle', size=8, color=['green' if val == 1 else 'red' for val in y_recent_actual])), secondary_y=True)

        fig_proba.update_layout(
            title='📈 Recent Close Price vs. Predicted Up Probability',
            xaxis_title='Date',
            yaxis_title='Close Price',
            template='plotly_dark',
            height=500,
            font=dict(family="Inter", size=12, color="#E0E0E0")
        )
        fig_proba.update_yaxes(title_text="Probability of Up", secondary_y=True)
        st.plotly_chart(fig_proba, use_container_width=True)
    else:
        st.info("Not enough recent data to visualize model performance.")

def compute_rsi(series, period=14):
    """Computes the Relative Strength Index (RSI). Copied from app.py to make module self-contained."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    return 100 - (100 / (1 + rs))
