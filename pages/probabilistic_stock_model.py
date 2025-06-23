import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def compute_rsi(series, period=14):
    """Computes the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Handle division by zero: if avg_loss is 0, rs becomes infinity, RSI becomes 100
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    
    return 100 - (100 / (1 + rs))

def add_common_features(hist_df):
    """Adds various technical indicators as features common to multiple modules."""
    df = hist_df.copy()
    # Ensure 'Close' column is numeric for calculations
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')

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
        st.warning("Please select a stock and load historical data to run probabilistic models.")
        return

    st.markdown("<p>This section uses a Random Forest Classifier to predict whether the stock's closing price will go up (1) or down (0) the next day, based on historical features.</p>", unsafe_allow_html=True)

    with st.spinner("Preparing data and training model..."):
        df = add_common_features(hist_data.copy())

    # --- Added Robustness Check: Ensure enough data points after feature engineering ---
    min_data_points_required = 60 # For MA50 and sufficient data for train/test split
    if len(df) < min_data_points_required:
        st.warning(f"Not enough clean historical data (need at least {min_data_points_required} data points after feature engineering) for probabilistic models. Please try a ticker with a longer history or wait for more data to accumulate.")
        return

    # --- Model Training ---
    st.markdown("<h5>Model Training & Performance:</h5>", unsafe_allow_html=True)

    features = ['Return_1d', 'MA10', 'MA50', 'Volatility', 'RSI', 'Volume_MA5', 'High_Low_Diff']
    
    # Ensure all features exist in the DataFrame
    missing_features = [f for f in features if f not in df.columns or df[f].isnull().all()]
    if missing_features:
        st.error(f"Error: Missing or all-NaN required features for the model: {', '.join(missing_features)}. This might be due to a problem with data processing or very limited data.")
        return

    X = df[features]
    y = df['Target']

    # Ensure there are at least two classes (up and down) and enough samples for stratification
    if y.nunique() < 2:
        st.warning("Not enough diverse data points to properly train the probabilistic model (e.g., stock only moved in one direction). Skipping model training and prediction.")
        return

    # Using a fixed test size, but ensure train and test sets won't be empty
    test_size_ratio = 0.2
    if len(X) * test_size_ratio < 1 or len(X) * (1 - test_size_ratio) < 1:
        st.warning("Dataset is too small to split into training and testing sets. Cannot evaluate model performance reliably. Training on full dataset for prediction.")
        X_train, y_train = X, y # Train on full data for prediction, but skip test metrics
        display_metrics = False
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42, stratify=y)
        display_metrics = True

        # Final check for empty splits after stratification (can happen with very skewed small datasets)
        if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
            st.warning("Train/test split resulted in empty sets. Not enough data points to properly train and evaluate the probabilistic model.")
            display_metrics = False
            X_train, y_train = X, y # Train on full data for prediction


    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # Use all cores
    model.fit(X_train, y_train)

    if display_metrics:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] # Probability of going up

        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Model Accuracy (on test set):** `{accuracy:.2f}`")
        st.markdown("<h6>Classification Report:</h6>", unsafe_allow_html=True)
        try: # Wrap classification_report in try-except for robustness
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0) # zero_division=0 to handle cases where a class has no predicted samples
            st.json(report_dict)
        except ValueError as e:
            st.warning(f"Could not generate full classification report due to data issues: {e}. Report might be incomplete if all predictions are the same.")


    # --- Prediction for Next Day ---
    st.markdown("<h5>Tomorrow's Prediction:</h5>", unsafe_allow_html=True)
    
    # Get the last valid data point for prediction
    last_data_point_raw = df.iloc[-1]
    
    # Check if the last data point has all required features
    if any(pd.isna(last_data_point_raw[f]) for f in features):
        st.warning("Cannot make a prediction for tomorrow: The latest data point has missing or invalid values after feature calculation. This might indicate issues with very recent data.")
        return

    last_data_point_features = last_data_point_raw[features].values.reshape(1, -1)
    
    next_day_prediction = model.predict(last_data_point_features)[0]
    next_day_proba_up = model.predict_proba(last_data_point_features)[0, 1]

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
    recent_data_len = min(len(df), 100) # Show max 100 recent points
    recent_data = df.tail(recent_data_len).copy()
    
    if not recent_data.empty:
        # Ensure 'Date' is a proper datetime for plotting
        recent_data['Date'] = pd.to_datetime(recent_data['Date'])

        X_recent = recent_data[features]
        # Re-predict probabilities for the visualization subset to ensure consistency
        y_recent_proba = model.predict_proba(X_recent)[:, 1]
        y_recent_actual = recent_data['Target']

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
            showlegend=True,
            font=dict(family="Inter", size=12, color="#E0E0E0")
        )
        fig_proba.update_yaxes(title_text="Probability of Up", secondary_y=True, range=[0,1]) # Secondary y-axis for probability
        st.plotly_chart(fig_proba, use_container_width=True)
    else:
        st.info("Not enough recent data to visualize model performance.")

    st.warning("""
    **Disclaimer for Probabilistic Models:**
    This model is for informational purposes only and is a simplified representation of complex market dynamics.
    - **Accuracy:** Model accuracy is based on historical data and does not guarantee future performance.
    - **Limitations:** It does not account for external events, news, macroeconomic factors, or human behavior.
    - **Risk:** Stock markets are inherently volatile. **Do not use this prediction for actual investment decisions.**
    """)
