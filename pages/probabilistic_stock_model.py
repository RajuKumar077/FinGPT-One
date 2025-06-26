import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

@st.cache_data(ttl=3600, show_spinner=False)
def compute_rsi(series, period=14):
    """Computes the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=3600, show_spinner=False)
def add_common_features(hist_df):
    """Adds various technical indicators as features common to multiple modules."""
    if hist_df.empty:
        return hist_df

    df = hist_df.copy()
    required_columns = ['Close', 'Volume', 'High', 'Low']
    
    # Validate required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}. Cannot compute features.")
        return pd.DataFrame()

    # Ensure numeric columns
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().all():
            st.error(f"❌ Column '{col}' contains only invalid or missing values. Cannot compute features.")
            return pd.DataFrame()

    df['Return_1d'] = df['Close'].pct_change()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['High_Low_Diff'] = (df['High'] - df['Low']) / df['Close']
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # 1 = up, 0 = down
    df.dropna(inplace=True)
    return df

def display_probabilistic_models(hist_data):
    st.markdown("<h3 class='section-title'>Probabilistic Stock Models (Daily Prediction)</h3>", unsafe_allow_html=True)

    if hist_data.empty:
        st.warning("⚠️ No historical data provided. Please select a stock and load historical data to run probabilistic models.")
        return

    # Validate input data
    required_columns = ['Close', 'Volume', 'High', 'Low']
    missing_cols = [col for col in required_columns if col not in hist_data.columns]
    if missing_cols:
        st.error(f"❌ Historical data missing required columns: {', '.join(missing_cols)}. Cannot run probabilistic models.")
        return

    with st.spinner("Preparing data and training model..."):
        df = add_common_features(hist_data)

    if df.empty:
        st.error("❌ Failed to compute features due to insufficient or invalid data. Check historical data source.")
        return

    # Ensure enough data points
    min_data_points_required = 60
    if len(df) < min_data_points_required:
        st.warning(f"⚠️ Not enough data points ({len(df)} < {min_data_points_required}) after feature engineering. Try a ticker with more history.")
        return

    st.markdown("<p>This section uses a Random Forest Classifier to predict whether the stock's closing price will go up (1) or down (0) the next day, based on technical indicators.</p>", unsafe_allow_html=True)

    # Model Training
    st.markdown("<h5>Model Training & Performance:</h5>", unsafe_allow_html=True)

    features = ['Return_1d', 'MA10', 'MA50', 'Volatility', 'RSI', 'Volume_MA5', 'High_Low_Diff']
    
    # Verify features exist
    missing_features = [f for f in features if f not in df.columns or df[f].isnull().all()]
    if missing_features:
        st.error(f"❌ Missing or invalid features: {', '.join(missing_features)}. Cannot train model.")
        return

    X = df[features]
    y = df['Target']

    # Check for sufficient class diversity
    if y.nunique() < 2:
        st.warning("⚠️ Insufficient class diversity (all price movements are up or down). Cannot train a meaningful model.")
        return

    # Train-test split
    test_size_ratio = 0.2
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42, stratify=y)
        display_metrics = True
    except ValueError as e:
        st.warning(f"⚠️ Failed to split data: {e}. Training on full dataset for prediction, but metrics will not be displayed.")
        X_train, y_train = X, y
        display_metrics = False

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    if display_metrics:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Model Accuracy (on test set):** `{accuracy:.2f}`")
        st.markdown("<h6>Classification Report:</h6>", unsafe_allow_html=True)
        try:
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.json(report_dict)
        except ValueError as e:
            st.warning(f"⚠️ Failed to generate classification report: {e}. Predictions may be biased toward one class.")

    # Prediction for Next Day
    st.markdown("<h5>Tomorrow's Prediction:</h5>", unsafe_allow_html=True)
    
    last_data_point = df.iloc[-1]
    if any(pd.isna(last_data_point[f]) for f in features):
        st.warning("⚠️ Cannot predict: Latest data point has missing or invalid feature values.")
        return

    last_data_point_features = last_data_point[features].values.reshape(1, -1)
    next_day_prediction = model.predict(last_data_point_features)[0]
    next_day_proba_up = model.predict_proba(last_data_point_features)[0, 1]

    prediction_text = "UP 🟢" if next_day_prediction == 1 else "DOWN 🔴"
    st.markdown(f"**Predicted Movement for Next Trading Day:** `{prediction_text}`")
    st.markdown(f"**Probability of Price Going Up:** `{next_day_proba_up:.2%}`")

    # Feature Importance Visualization
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

    # Recent Performance Visualization
    st.markdown("<h5>Recent Model Performance:</h5>", unsafe_allow_html=True)
    
    max_plot_points = 60
    recent_data = df.tail(max_plot_points).copy()
    
    if not recent_data.empty:
        X_recent = recent_data[features]
        y_recent_proba = model.predict_proba(X_recent)[:, 1]
        y_recent_actual = recent_data['Target']

        fig_proba = make_subplots(specs=[[{"secondary_y": True}]])
        fig_proba.add_trace(
            go.Scatter(x=recent_data.index, y=recent_data['Close'], name='Close Price', line=dict(color='cyan')),
            secondary_y=False
        )
        fig_proba.add_trace(
            go.Scatter(x=recent_data.index, y=y_recent_proba, name='Predicted Probability (Up)', line=dict(color='orange', dash='dash')),
            secondary_y=True
        )
        fig_proba.add_trace(
            go.Scatter(
                x=recent_data.index, 
                y=y_recent_actual, 
                mode='markers', 
                name='Actual Direction (1=Up, 0=Down)',
                marker=dict(symbol='circle', size=8, color=['green' if val == 1 else 'red' for val in y_recent_actual])
            ),
            secondary_y=True
        )

        fig_proba.update_layout(
            title='📈 Recent Close Price vs. Predicted Up Probability',
            xaxis_title='Date',
            yaxis_title='Close Price',
            template='plotly_dark',
            height=500,
            showlegend=True,
            font=dict(family="Inter", size=12, color="#E0E0E0")
        )
        fig_proba.update_yaxes(title_text="Probability/Direction", secondary_y=True, range=[0, 1])
        st.plotly_chart(fig_proba, use_container_width=True)
    else:
        st.info("⚠️ Not enough recent data to visualize model performance.")

    st.warning("""
    **Disclaimer for Probabilistic Models:**
    This model is for informational purposes only and is a simplified representation of complex market dynamics.
    - **Accuracy:** Model accuracy is based on historical data and does not guarantee future performance.
    - **Limitations:** It does not account for external events, news, macroeconomic factors, or human behavior.
    - **Risk:** Stock markets are inherently volatile. **Do not use this prediction for actual investment decisions.**
    """)
