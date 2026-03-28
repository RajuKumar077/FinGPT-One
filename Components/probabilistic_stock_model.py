import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from ta.momentum import ROCIndicator, RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

warnings.filterwarnings("ignore")

CHART_PAPER = "rgba(255,255,255,0.02)"
CHART_PLOT = "rgba(255,255,255,0.01)"
TEXT_COLOR = "#E8EEF9"
GRID_COLOR = "rgba(255,255,255,0.08)"
FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", sans-serif'


def apply_chart_theme(fig, title=None, height=420):
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_PLOT,
        font=dict(color=TEXT_COLOR, family=FONT_FAMILY),
        margin=dict(l=24, r=24, t=60, b=24),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig


def render_metric_card(label, value, detail, accent):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta" style="color:{accent};">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def add_advanced_features(hist_df):
    if hist_df.empty:
        return pd.DataFrame()

    df = hist_df[["Close", "High", "Low", "Open", "Volume"]].copy().tail(600)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_rows = len(df)
    if n_rows < 40:
        return pd.DataFrame()

    short_w = max(5, min(10, n_rows // 20))
    med_w = max(15, min(30, n_rows // 10))

    df["Return_1d"] = df["Close"].pct_change().fillna(0)
    df["Return_5d"] = df["Close"].pct_change(5).fillna(0)
    df["Return_10d"] = df["Close"].pct_change(10).fillna(0)
    df["Gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1).replace(0, np.nan)
    df["HL_Range"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)
    df["Body_Size"] = abs(df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan)
    df["Upper_Shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Close"].replace(0, np.nan)
    df["Lower_Shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Close"].replace(0, np.nan)
    df["MA_short"] = df["Close"].rolling(window=short_w, min_periods=1).mean()
    df["MA_med"] = df["Close"].rolling(window=med_w, min_periods=1).mean()
    df["Trend_Spread"] = (df["MA_short"] - df["MA_med"]) / df["MA_med"].replace(0, np.nan)
    df["Volatility"] = df["Return_1d"].rolling(window=short_w, min_periods=3).std().fillna(0)
    df["Volume_Z"] = ((df["Volume"] - df["Volume"].rolling(short_w, min_periods=3).mean()) / df["Volume"].rolling(short_w, min_periods=3).std()).replace([np.inf, -np.inf], 0)
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(window=med_w, min_periods=3).mean().replace(0, np.nan)

    rsi_obj = RSIIndicator(close=df["Close"], window=min(14, max(5, short_w + 2)))
    df["RSI"] = rsi_obj.rsi().fillna(50)
    macd_obj = MACD(close=df["Close"])
    df["MACD"] = macd_obj.macd().fillna(0)
    df["MACD_Signal"] = macd_obj.macd_signal().fillna(0)
    df["MACD_Hist"] = (df["MACD"] - df["MACD_Signal"]).fillna(0)
    bb_obj = BollingerBands(close=df["Close"], window=min(20, max(10, med_w)))
    band_width = (bb_obj.bollinger_hband() - bb_obj.bollinger_lband()).replace(0, np.nan)
    df["BB_Position"] = ((df["Close"] - bb_obj.bollinger_lband()) / band_width).fillna(0.5)
    roc_obj = ROCIndicator(close=df["Close"], window=min(10, max(5, short_w)))
    df["ROC_10"] = roc_obj.roc().fillna(0)

    df["Target_1d"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df["Target_3d"] = (df["Close"].shift(-3) > df["Close"]).astype(int)
    df["Target_5d"] = (df["Close"].shift(-5) > df["Close"]).astype(int)

    return df.replace([np.inf, -np.inf], np.nan).ffill().bfill().iloc[:-5]


def build_model():
    return VotingClassifier(
        estimators=[
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=220,
                    max_depth=9,
                    min_samples_leaf=3,
                    min_samples_split=8,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=220,
                    max_depth=10,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
            (
                "gb",
                GradientBoostingClassifier(
                    learning_rate=0.05,
                    n_estimators=120,
                    max_depth=2,
                    random_state=42,
                ),
            ),
        ],
        voting="soft",
    )


def train_model(X_train, y_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = build_model()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    return scaler, model, y_pred, y_proba


def get_feature_importance(model, feature_names):
    collected = []
    for estimator in model.estimators_:
        if hasattr(estimator, "feature_importances_"):
            collected.append(estimator.feature_importances_)
    if not collected:
        return pd.Series(dtype=float)
    return pd.Series(np.mean(collected, axis=0), index=feature_names).sort_values(ascending=True)


def display_probabilistic_models(hist_data):
    st.markdown(
        """
        <section class="hero-panel">
            <div class="hero-kicker">Probability Engine</div>
            <h1>Directional prediction lab</h1>
            <p>An upgraded ensemble classifier that blends tree models, validates on recent market behavior, and presents cleaner confidence signals.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if hist_data.empty:
        st.error("No historical data available.")
        return

    pro_df = add_advanced_features(hist_data)
    if pro_df.empty or len(pro_df) < 120:
        st.error(f"Need at least 120 rows for stable ensemble training. Current usable rows: {len(pro_df)}")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        test_size = st.slider("Validation split (%)", 15, 35, 20, 5)
    with col2:
        horizon = st.selectbox("Prediction horizon", ["1-Day", "3-Day", "5-Day"], index=1)
        target_col = {"1-Day": "Target_1d", "3-Day": "Target_3d", "5-Day": "Target_5d"}[horizon]

    features = [
        "Return_1d",
        "Return_5d",
        "Return_10d",
        "Gap",
        "HL_Range",
        "Body_Size",
        "Upper_Shadow",
        "Lower_Shadow",
        "Trend_Spread",
        "Volatility",
        "Volume_Z",
        "Volume_Ratio",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "BB_Position",
        "ROC_10",
    ]
    available_features = [feature for feature in features if feature in pro_df.columns]

    X = pro_df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = pro_df[target_col]

    split_idx = int(len(X) * (1 - test_size / 100))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        st.warning("This ticker currently has too little class variation for a reliable directional model.")
        return

    with st.spinner("Training ensemble classifier..."):
        scaler, model, y_pred, y_proba = train_model(X_train, y_train, X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())

    latest_features = X.tail(1)
    latest_scaled = scaler.transform(latest_features)
    prob_up = float(model.predict_proba(latest_scaled)[0, 1])
    confidence = abs(prob_up - 0.5) * 2

    if prob_up >= 0.58:
        signal = "Bullish"
        accent = "#7EE0C3"
    elif prob_up <= 0.42:
        signal = "Bearish"
        accent = "#F3A6B3"
    else:
        signal = "Neutral"
        accent = "#F8C471"

    st.markdown("### Ensemble Snapshot")
    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Signal", signal, f"P(up) {prob_up:.1%}", accent)
    with cols[1]:
        render_metric_card("Accuracy", f"{acc:.1%}", f"Baseline {baseline_acc:.1%}", "#9DC6FF")
    with cols[2]:
        render_metric_card("AUC", f"{auc:.3f}", "Recent validation", "#7EE0C3")
    with cols[3]:
        render_metric_card("Confidence", f"{confidence:.1%}", horizon, accent)

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="section-kicker">Interpretation</div>
            <h3 style="margin-bottom:0.4rem;">{signal} bias for the next {horizon.lower()}</h3>
            <p style="margin:0;color:var(--text-secondary);">
                This view comes from a soft-voting ensemble of Random Forest, Extra Trees, and Gradient Boosting models trained on technical features.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    importance = get_feature_importance(model, available_features)
    if not importance.empty:
        fig_importance = px.bar(
            importance.tail(10),
            orientation="h",
            title="Most influential features",
            color=importance.tail(10),
            color_continuous_scale=["#9DC6FF", "#7EE0C3"],
        )
        fig_importance.update_coloraxes(showscale=False)
        st.plotly_chart(apply_chart_theme(fig_importance, "Feature importance", 420), use_container_width=True)

    st.markdown("### Validation Diagnostics")
    diag_col1, diag_col2 = st.columns(2)

    with diag_col1:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=["Pred Down", "Pred Up"],
            y=["Actual Down", "Actual Up"],
            annotation_text=cm.astype(str),
            colorscale=[[0, "rgba(157,198,255,0.18)"], [1, "rgba(126,224,195,0.9)"]],
            showscale=False,
        )
        st.plotly_chart(apply_chart_theme(fig_cm, "Confusion matrix", 360), use_container_width=True)

    with diag_col2:
        recent_n = min(120, len(X))
        rolling_proba = model.predict_proba(scaler.transform(X.tail(recent_n)))[:, 1]
        recent_price = pro_df["Close"].tail(recent_n)
        fig_evo = go.Figure()
        fig_evo.add_trace(go.Scatter(x=recent_price.index, y=recent_price.values, name="Price", line=dict(color="#9DC6FF", width=2.2)))
        fig_evo.add_trace(
            go.Scatter(
                x=recent_price.index,
                y=rolling_proba,
                name="P(up)",
                line=dict(color="#7EE0C3", width=2.2, dash="dot"),
                yaxis="y2",
            )
        )
        fig_evo.update_layout(yaxis=dict(title="Price"), yaxis2=dict(title="P(up)", overlaying="y", side="right", range=[0, 1]))
        st.plotly_chart(apply_chart_theme(fig_evo, "Price vs confidence", 360), use_container_width=True)

    st.markdown("### Strategy Backtest")
    shifted_returns = pro_df["Return_1d"].shift(-1).fillna(0)
    historical_pred = model.predict(scaler.transform(X))
    strategy_returns = pd.Series(np.where(historical_pred == 1, shifted_returns, -shifted_returns), index=pro_df.index).fillna(0)
    buy_hold_curve = (1 + pro_df["Return_1d"].fillna(0)).cumprod()
    strategy_curve = (1 + strategy_returns).cumprod()

    backtest_cols = st.columns(2)
    with backtest_cols[0]:
        render_metric_card("AI Strategy", f"{strategy_curve.iloc[-1] - 1:+.1%}", "Cumulative return", "#7EE0C3")
    with backtest_cols[1]:
        render_metric_card("Buy & Hold", f"{buy_hold_curve.iloc[-1] - 1:+.1%}", "Cumulative return", "#9DC6FF")

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=buy_hold_curve.index, y=buy_hold_curve.values, name="Buy & Hold", line=dict(color="#9DC6FF", width=2.2)))
    fig_bt.add_trace(go.Scatter(x=strategy_curve.index, y=strategy_curve.values, name="AI Strategy", line=dict(color="#7EE0C3", width=2.2)))
    st.plotly_chart(apply_chart_theme(fig_bt, "Cumulative returns", 420), use_container_width=True)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(
        {
            "Class": ["Down", "Up", "Macro Avg"],
            "Precision": [report["0"]["precision"], report["1"]["precision"], report["macro avg"]["precision"]],
            "Recall": [report["0"]["recall"], report["1"]["recall"], report["macro avg"]["recall"]],
            "F1-Score": [report["0"]["f1-score"], report["1"]["f1-score"], report["macro avg"]["f1-score"]],
        }
    ).round(3)
    st.markdown("### Classification Report")
    st.dataframe(report_df, use_container_width=True, hide_index=True)
