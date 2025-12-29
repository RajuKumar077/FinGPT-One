import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import ta
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(30, 58, 138, 0.3);
    }
    .prediction-card {
        background: linear-gradient(145deg, #10b981 0%, #059669 100%);
        border-radius: 20px;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 15px 35px rgba(16, 185, 129, 0.3);
        margin: 2rem 0;
    }
    .prediction-card.bearish {
        background: linear-gradient(145deg, #ef4444 0%, #dc2626 100%);
        box-shadow: 0 15px 35px rgba(239, 68, 68, 0.3);
    }
    .prediction-card.neutral {
        background: linear-gradient(145deg, #f59e0b 0%, #d97706 100%);
        box-shadow: 0 15px 35px rgba(245, 158, 11, 0.3);
    }
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stPlotlyChart {
        border-radius: 12px !important;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def add_advanced_features(hist_df):
    """🚀 Enhanced Feature Engineering - 15+ Robust Indicators (Handles Edge Cases)"""
    if hist_df.empty:
        return pd.DataFrame()
    
    df = hist_df[['Close', 'High', 'Low', 'Open', 'Volume']].copy().tail(500)
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    n_rows = len(df)
    if n_rows < 10:
        return pd.DataFrame()
    
    # Adaptive windows
    short_w = max(3, min(5, n_rows // 10))
    med_w = max(10, min(20, n_rows // 5))
    rsi_w = max(5, min(14, n_rows // 5))
    
    # Core Returns
    df['Return_1d'] = df['Close'].pct_change().fillna(0)
    df['Return_5d'] = df['Close'].pct_change(5).fillna(0)
    
    # Price Action
    df['HL_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Open'].replace(0, 1)
    df['Upper_Shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
    df['Lower_Shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
    
    # Moving Averages
    df['MA_short'] = df['Close'].rolling(window=short_w, min_periods=1).mean()
    df['MA_med'] = df['Close'].rolling(window=med_w, min_periods=1).mean()
    df['MA_ratio_short'] = df['Close'] / df['MA_short']
    df['MA_ratio_med'] = df['Close'] / df['MA_med']
    
    # Volatility
    df['Volatility'] = df['Return_1d'].rolling(window=short_w, min_periods=2).std().fillna(0)
    
    # RSI (Safe)
    try:
        rsi_obj = RSIIndicator(close=df['Close'], window=rsi_w)
        df['RSI'] = rsi_obj.rsi().fillna(50)
        df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
        df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
    except Exception as e:
        df['RSI'] = 50.0
        df['RSI_Oversold'] = 0
        df['RSI_Overbought'] = 0
    
    # Volume
    vol_ma = df['Volume'].rolling(window=short_w, min_periods=1).mean()
    df['Vol_Ratio'] = df['Volume'] / vol_ma.replace(0, 1)
    
    # MACD (Safe)
    try:
        macd_obj = MACD(close=df['Close'])
        df['MACD'] = macd_obj.macd().fillna(0)
        df['MACD_Signal'] = macd_obj.macd_signal().fillna(0)
    except Exception:
        df['MACD'] = 0.0
        df['MACD_Signal'] = 0.0
    
    # Bollinger Bands (Safe)
    try:
        bb_obj = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_Position'] = ((df['Close'] - bb_obj.bollinger_lband()) / (bb_obj.bollinger_hband() - bb_obj.bollinger_lband())).fillna(0.5)
    except Exception:
        df['BB_Position'] = 0.5
    
    # ROC (Rate of Change)
    try:
        roc_obj = ROCIndicator(close=df['Close'], window=10)
        df['ROC_10'] = roc_obj.roc().fillna(0)
    except Exception:
        df['ROC_10'] = 0.0
    
    # Multi-Targets
    df['Target_1d'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['Target_3d'] = (df['Close'].shift(-3) > df['Close']).astype(int)
    df['Target_5d'] = (df['Close'].shift(-5) > df['Close']).astype(int)
    
    # Clean
    df = df.ffill().bfill().iloc[:-1]  # Remove last (no target)
    return df

def train_model(X_train, y_train, X_test, y_test):
    """🎯 Robust Ensemble Training with Class Balancing"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',  # Handle imbalance
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1] if len(model.classes_) == 2 else np.full(len(X_test), 0.5)
    
    return scaler, model, y_pred, y_proba

def display_probabilistic_models(hist_data):
    """🎯 Pro AI Prediction Dashboard - Enhanced with Multi-Horizon & Backtest"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style='margin: 0; font-size: 3rem;'>🎯 AI Stock Prediction Engine Pro</h1>
        <p style='margin: 0; font-size: 1.2rem; opacity: 0.9;'>Powered by 15+ Technical Indicators | Multi-Horizon Forecasting</p>
    </div>
    """, unsafe_allow_html=True)
    
    if hist_data.empty:
        st.error("❌ No historical data available.")
        return
    
    # Progress
    progress_container = st.container()
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    with progress_container:
        status_text.text("🔄 Engineering Advanced Features...")
        pro_df = add_advanced_features(hist_data)
        progress_bar.progress(30)
    
    if pro_df.empty or len(pro_df) < 10:
        st.error(f"❌ Need 10+ days of data. Current: {len(pro_df)}")
        return
    
    status_text.text("⚙️ Configuring Model...")
    
    # Controls
    col1, col2 = st.columns([2, 1])
    with col1:
        test_size = st.slider("🧪 Test Split (%)", 15, 40, 25, 5)
    with col2:
        horizon = st.selectbox("🎯 Prediction Horizon", ["1-Day", "3-Day", "5-Day"], index=0)
        target_col = {'1-Day': 'Target_1d', '3-Day': 'Target_3d', '5-Day': 'Target_5d'}[horizon]
    
    # Features
    features = [
        'Return_1d', 'Return_5d', 'HL_Range', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow',
        'MA_ratio_short', 'MA_ratio_med', 'Volatility', 'RSI', 'RSI_Oversold', 'RSI_Overbought',
        'Vol_Ratio', 'MACD', 'MACD_Signal', 'BB_Position', 'ROC_10'
    ]
    available_features = [f for f in features if f in pro_df.columns]
    
    X = pro_df[available_features]
    y = pro_df[target_col]
    
    # Split
    split_idx = int(len(X) * (1 - test_size / 100))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    status_text.text("🤖 Training Ensemble Model...")
    progress_bar.progress(60)
    
    scaler, model, y_pred, y_proba = train_model(X_train, y_train, X_test, y_test)
    progress_bar.progress(100)
    
    # Metrics Dashboard
    st.markdown("### 📊 Model Performance Metrics")
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Accuracy", f"{acc:.2%}", delta=None)
    col2.metric("📈 AUC-ROC", f"{auc:.3f}", delta=None)
    col3.metric("🏭 Train Size", f"{len(X_train):,}", delta=None)
    col4.metric("🧪 Test Size", f"{len(X_test):,}", delta=None)
    
    # Prediction Card
    last_features = X.tail(1)
    last_scaled = scaler.transform(last_features)
    prob_up = model.predict_proba(last_scaled)[0, 1]
    
    confidence = max(prob_up, 1 - prob_up)
    if prob_up > 0.55:
        signal_class = "bullish"
        signal_text = "🟢 BULLISH"
    elif prob_up < 0.45:
        signal_class = "bearish"
        signal_text = "🔴 BEARISH"
    else:
        signal_class = "neutral"
        signal_text = "🟡 NEUTRAL"
    
    st.markdown(f"""
    <div class="prediction-card {signal_class}">
        <h2 style='margin: 0; font-size: 2.5rem;'>{signal_text}</h2>
        <p style='margin: 0.5rem 0; font-size: 1.5rem;'>P(UP): <strong>{prob_up:.1%}</strong></p>
        <p style='margin: 0; font-size: 1.2rem;'>Confidence: <strong>{confidence:.1%}</strong> | Horizon: {horizon}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Importance
    st.markdown("### 🏆 Feature Importance (Top Contributors)")
    importances = pd.Series(model.feature_importances_, index=available_features).sort_values(ascending=True)
    fig_importance = px.bar(importances.tail(8), orientation='h', title="Key Predictive Features",
                            color=importances.tail(8), color_continuous_scale='viridis')
    fig_importance.update_layout(height=400, template='plotly_white')
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Confusion Matrix
    st.markdown("### 🔍 Confusion Matrix")
    try:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = ff.create_annotated_heatmap(
            z=cm, x=['Actual Down', 'Actual Up'], y=['Pred Down', 'Pred Up'],
            annotation_text=cm, colorscale='Blues', showscale=True
        )
        fig_cm.update_layout(title="Prediction vs Actual", height=350, template='plotly_white')
        st.plotly_chart(fig_cm, use_container_width=True)
    except Exception:
        st.info("⚠️ Skipping confusion matrix due to data issues.")
    
    # Prediction Evolution
    st.markdown("### 📈 Prediction Confidence Over Time")
    recent_n = min(100, len(X))
    recent_proba = model.predict_proba(scaler.transform(X.tail(recent_n)))[:, 1]
    recent_price = pro_df['Close'].tail(recent_n)
    
    fig_evo = go.Figure()
    fig_evo.add_trace(go.Scatter(x=recent_price.index, y=recent_price, name='Price', line=dict(color='#1f77b4', width=2)))
    fig_evo.add_trace(go.Scatter(x=recent_price.index, y=recent_proba, name='P(UP)', line=dict(color='#ff7f0e', width=2, dash='dot'), yaxis='y2'))
    fig_evo.update_layout(
        height=450, template='plotly_white', title="Price Trend vs Prediction Confidence",
        yaxis=dict(title="Price ($)"), yaxis2=dict(title="P(UP)", overlaying='y', side='right')
    )
    st.plotly_chart(fig_evo, use_container_width=True)
    
    # Simple Backtest
    st.markdown("### 💹 Quick Backtest (Strategy vs Buy & Hold)")
    try:
        historical_pred = model.predict(scaler.transform(X))
        strategy_returns = np.where(historical_pred == 1, pro_df['Return_1d'].shift(-1), -pro_df['Return_1d'].shift(-1))
        strategy_returns = pd.Series(strategy_returns, index=pro_df.index).fillna(0)
        cum_strategy = (1 + strategy_returns).cumprod().iloc[-1]
        cum_buyhold = (1 + pro_df['Return_1d']).cumprod().iloc[-1]
        
        col1, col2 = st.columns(2)
        col1.metric("🤖 AI Strategy Return", f"{cum_strategy:.1%}", delta=f"{cum_strategy - cum_buyhold:+.1%}")
        col2.metric("📈 Buy & Hold Return", f"{cum_buyhold:.1%}")
        
        # Backtest Plot
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=pro_df.index, y=(1 + pro_df['Return_1d']).cumprod(), name='Buy & Hold', line=dict(color='#ef4444')))
        fig_bt.add_trace(go.Scatter(x=pro_df.index, y=(1 + strategy_returns).cumprod(), name='AI Strategy', line=dict(color='#10b981')))
        fig_bt.update_layout(height=400, template='plotly_white', title="Cumulative Returns Comparison")
        st.plotly_chart(fig_bt, use_container_width=True)
    except Exception as e:
        st.info(f"⚠️ Backtest skipped: {str(e)[:50]}...")
    
    # Classification Report
    st.markdown("### 📋 Detailed Classification Report")
    try:
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame({
            'Class': ['Down', 'Up', 'Macro Avg'],
            'Precision': [report['0']['precision'], report['1']['precision'], report['macro avg']['precision']],
            'Recall': [report['0']['recall'], report['1']['recall'], report['macro avg']['recall']],
            'F1-Score': [report['0']['f1-score'], report['1']['f1-score'], report['macro avg']['f1-score']]
        }).round(3)
        st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['F1-Score']), use_container_width=True)
    except Exception:
        st.info("⚠️ Report generation skipped due to class imbalance.")
    
    st.markdown("---")
    st.success("✨ **Pro AI Model Deployed** | Ready for Multi-Horizon Predictions")

# Legacy call (for compatibility)
if 'hist_data' in locals():
    display_probabilistic_models(hist_data)