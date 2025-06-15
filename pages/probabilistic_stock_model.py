import streamlit as st
import yfinance as yf  # Keep yfinance here for internal stock info if needed, but primary data comes as hist_data
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')


# --- Helper functions (can be imported from a common utils file if created) ---
def compute_rsi(series, period=14):
    """Computes the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def add_features_prob(hist):
    """Adds various technical indicators as features for probabilistic models."""
    df = hist.copy()
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


# --- Deep Learning Model (LSTM) ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


@st.cache_resource(show_spinner=False)  # Cache model training
def prepare_lstm_data_prob(df, features, sequence_length):
    """Prepares data for LSTM model with sequence input."""
    data = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    if len(scaled_data) < sequence_length:
        return np.array([]), np.array([]), scaler

    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(df['Target'].iloc[i + sequence_length])
    return np.array(X), np.array(y), scaler


@st.cache_resource(show_spinner=False)  # Cache model training
def train_lstm_model_prob(X_train, y_train, X_test, y_test, input_size, hidden_size=50, num_layers=2,
                          sequence_length=20,
                          epochs=20, batch_size=32):
    """Trains an LSTM model."""
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        return None, np.array([]), np.array([])

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel(input_size, hidden_size, num_layers, 1)
    criterion = nn.BCELoss()
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
        test_probs = model(X_test_tensor).numpy()
        test_preds = (test_probs > 0.5).astype(int)

    return model, test_preds, test_probs.flatten()


# --- Traditional ML Models ---
@st.cache_resource(show_spinner=False)  # Cache model training
def train_probabilistic_models(df):
    """Trains LightGBM and (Calibrated) Random Forest models."""
    features = ['Return_1d', 'MA10', 'MA50', 'Volatility', 'RSI', 'Volume_MA5', 'High_Low_Diff']

    available_features = [f for f in features if f in df.columns]

    X = df[available_features]
    y = df['Target']

    if X.empty or len(X) < 2:
        return None, None, pd.DataFrame(), pd.Series(), None  # Added rf_raw return

    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    if X_train.empty or X_test.empty:
        return None, None, pd.DataFrame(), pd.Series(), None  # Added rf_raw return

    lgbm = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', n_estimators=100, learning_rate=0.05,
                              num_leaves=31, verbose=-1, random_state=42)
    lgbm.fit(X_train, y_train)

    rf_raw = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_calibrated = CalibratedClassifierCV(rf_raw, method='isotonic', cv=3)
    rf_calibrated.fit(X_train, y_train)

    return lgbm, rf_calibrated, X_test, y_test, rf_raw  # Return rf_raw here


# --- Visualizations ---
def visualize_probabilities(model_probs, dates, model_name):
    """Visualizes predicted probabilities over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=model_probs, mode='lines', name='Probability Up',
                             line=dict(color='deepskyblue', width=2)))
    fig.update_layout(
        title=f"Probability of Price Increase ({model_name})",
        xaxis_title="Date",
        yaxis_title="Probability",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Inter", size=12, color="#E0E0E0")
    )
    return fig


def plot_confusion(y_true, y_pred, model_name):
    """Plots a confusion matrix using Seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', cbar=False,
                xticklabels=['Predict Down', 'Predict Up'],
                yticklabels=['Actual Down', 'Actual Up'], ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_probs, model_name):
    """Plots the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC = {auc_score:.2f})',
                             line=dict(color='firebrick', width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier',
                             line=dict(color='gray', dash='dash')))
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template="plotly_dark",
        showlegend=True,
        margin=dict(l=40, r=40, t=60, b=40),
        font=dict(family="Inter", size=12, color="#E0E0E0")
    )
    return fig


def explain_with_shap(model, X_sample, model_type="tree"):
    """Generates and plots SHAP values for model explainability."""
    try:
        if model_type == "tree":
            # For tree models, including LightGBM and base estimator of RandomForest
            explainer = shap.TreeExplainer(model)
        else:
            # Fallback for other model types or if tree explainer fails
            # Using X_sample.head(100) as background data for KernelExplainer for performance
            # Passing a subset for background data is crucial for KernelExplainer speed
            explainer = shap.KernelExplainer(model.predict_proba, X_sample.head(100))

        shap_values = explainer.shap_values(X_sample)

        # Handle multi-output SHAP values for binary classification (takes positive class)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values

        fig, ax = plt.subplots(figsize=(10, 6))
        # Removed 'ax=ax' from shap.summary_plot as it can cause issues with some versions/plots
        shap.summary_plot(shap_values_to_plot, X_sample, plot_type="bar", show=False)
        ax.set_title(f'SHAP Feature Importance - {model.__class__.__name__}', fontsize=14)
        plt.tight_layout()
        return fig
    except Exception as e:
        # Catch and log specific SHAP errors for better debugging
        st.warning(f"Could not generate SHAP plot for {model.__class__.__name__}: {e}")
        return None


def display_probabilistic_models(hist_data):
    """Main function to display probabilistic stock prediction models and their explanations."""
    st.markdown(f"<h3 class='section-title'>Probabilistic Stock Prediction</h3>", unsafe_allow_html=True)

    if hist_data is None or hist_data.empty:
        st.info("No data available for probabilistic models.")
        return

    # Add common features for probabilistic models
    df_features = add_features_prob(hist_data)
    if df_features.empty:
        st.info("Not enough data after feature engineering for probabilistic models.")
        return

    with st.spinner("Training probabilistic models (LightGBM, Random Forest, LSTM)..."):
        lgbm_model, rf_calibrated_model, X_test_ml, y_test_ml, rf_raw_model = train_probabilistic_models(
            df_features)  # Added rf_raw_model return

        # Prepare and train LSTM
        features_lstm = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return_1d', 'MA10', 'MA50', 'Volatility', 'RSI',
                         'Volume_MA5', 'High_Low_Diff']
        sequence_length = 20
        lstm_features = [f for f in features_lstm if f in df_features.columns]

        X_lstm, y_lstm, scaler_lstm = prepare_lstm_data_prob(df_features, lstm_features, sequence_length)

        if X_lstm.shape[0] == 0:
            st.info("Not enough data for LSTM model after preparing sequences. Skipping LSTM training.")
            lstm_model, lstm_test_preds, lstm_test_probs = None, np.array([]), np.array([])
            y_test_lstm = np.array([])
        else:
            train_size_lstm = int(len(X_lstm) * 0.8)
            X_train_lstm, X_test_lstm = X_lstm[:train_size_lstm], X_lstm[train_size_lstm:]
            y_train_lstm, y_test_lstm = y_lstm[:train_size_lstm], y_lstm[train_size_lstm:]

            if X_train_lstm.shape[0] == 0 or X_test_lstm.shape[0] == 0:
                st.warning("Not enough data to split into train/test sets for LSTM. Skipping LSTM training.")
                lstm_model, lstm_test_preds, lstm_test_probs = None, np.array([]), np.array([])
            else:
                lstm_model, lstm_test_preds, lstm_test_probs = train_lstm_model_prob(
                    X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm,
                    input_size=X_lstm.shape[2], sequence_length=sequence_length
                )

    st.markdown("<h4>Model Performance Metrics</h4>", unsafe_allow_html=True)

    if lgbm_model is None:  # If traditional models couldn't be trained
        st.info("Could not train LightGBM or Random Forest models due to insufficient data.")
        return

    # Prepare test dates for plots (for traditional ML models)
    test_dates_ml = hist_data['Date'].iloc[-len(X_test_ml):].values

    metrics_cols = st.columns(3)

    with metrics_cols[0]:
        st.markdown("<h5>LSTM Model</h5>", unsafe_allow_html=True)
        if lstm_model and len(y_test_lstm) > 0:
            test_dates_lstm = hist_data['Date'].iloc[-len(y_test_lstm):].values
            st.text(f"Classification Report:\n{classification_report(y_test_lstm, lstm_test_preds)}")
            st.pyplot(plot_confusion(y_test_lstm, lstm_test_preds, "LSTM"))
            st.plotly_chart(plot_roc_curve(y_test_lstm, lstm_test_probs, "LSTM"))
            st.plotly_chart(visualize_probabilities(lstm_test_probs, test_dates_lstm, "LSTM"))
        else:
            st.info("LSTM Model data not sufficient for evaluation or training failed.")

    with metrics_cols[1]:
        st.markdown("<h5>LightGBM Model</h5>", unsafe_allow_html=True)
        lgbm_test_preds = lgbm_model.predict(X_test_ml)
        lgbm_test_probs = lgbm_model.predict_proba(X_test_ml)[:, 1]
        st.text(f"Classification Report:\n{classification_report(y_test_ml, lgbm_test_preds)}")
        st.pyplot(plot_confusion(y_test_ml, lgbm_test_preds, "LightGBM"))
        st.plotly_chart(plot_roc_curve(y_test_ml, lgbm_test_probs, "LightGBM"))
        st.plotly_chart(visualize_probabilities(lgbm_test_probs, test_dates_ml, "LightGBM"))

    with metrics_cols[2]:
        st.markdown("<h5>Calibrated Random Forest Model</h5>", unsafe_allow_html=True)
        rf_test_preds = rf_calibrated_model.predict(X_test_ml)
        rf_test_probs = rf_calibrated_model.predict_proba(X_test_ml)[:, 1]
        st.text(f"Classification Report:\n{classification_report(y_test_ml, rf_test_preds)}")
        st.pyplot(plot_confusion(y_test_ml, rf_test_preds, "Calibrated Random Forest"))
        st.plotly_chart(plot_roc_curve(y_test_ml, rf_test_probs, "Calibrated Random Forest"))
        st.plotly_chart(visualize_probabilities(rf_test_probs, test_dates_ml, "Calibrated Random Forest"))

    st.markdown("<h4 class='section-subtitle'>Model Explainability (SHAP)</h4>", unsafe_allow_html=True)
    shap_cols = st.columns(2)
    with shap_cols[0]:
        # Pass lgbm_model directly for TreeExplainer
        lgbm_shap_plot = explain_with_shap(lgbm_model, X_test_ml.head(200), "tree")
        if lgbm_shap_plot:
            st.pyplot(lgbm_shap_plot)
    with shap_cols[1]:
        # Fix: Pass the Calibrated Random Forest model itself to KernelExplainer for SHAP.
        # Use a small subset (e.g., 100 samples) as background data for performance with KernelExplainer.
        if rf_calibrated_model is not None and not X_test_ml.empty:
            rf_shap_plot = explain_with_shap(rf_calibrated_model,
                                             X_test_ml.sample(min(200, len(X_test_ml)), random_state=42), "kernel")
            if rf_shap_plot:
                st.pyplot(rf_shap_plot)
        else:
            st.info("Calibrated Random Forest model or data not available for SHAP explanation.")

# Note: The if __name__ == "__main__": block is removed as this file is now imported as a module.
