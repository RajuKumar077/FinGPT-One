import streamlit as st
import yfinance as yf  # Keep yfinance for internal stock info if needed, but primary data comes as hist_data
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import lightgbm as lgb
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


# ========== 📅 Technical Indicators (adjusted to accept hist directly) ==========
@st.cache_data(show_spinner=False)
def add_tech_indicators_forecast(hist):
    """Adds various technical indicators for forecasting."""
    df = hist.copy()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    df['BB_MA'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_MA'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_MA'] - 2 * df['Close'].rolling(window=20).std()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9).mean()

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    df['Momentum'] = df['Close'].diff(4)

    df.fillna(method='bfill', inplace=True)
    return df


# ========== 🔮 Forecasting Models (now directly callable) ==========
@st.cache_resource(show_spinner=False)
def forecast_prophet_model(hist, period):
    df = hist[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}).dropna()
    if len(df) < 2:
        raise ValueError("Not enough data for Prophet model")
    model = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
    model.fit(df)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    forecast_series = forecast.set_index('ds')['yhat'][-period:]
    return forecast_series


@st.cache_resource(show_spinner=False)
def forecast_sarima_model(hist, period):
    if hist['Close'].dropna().shape[0] < 2:
        raise ValueError("Not enough data for SARIMA model")
    model = SARIMAX(hist['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    pred = results.get_forecast(steps=period)
    pred_df = pred.predicted_mean
    last_date = hist['Date'].iloc[-1]
    dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=period)
    pred_series = pd.Series(pred_df.values, index=dates)
    return pred_series


@st.cache_resource(show_spinner=False)
def prepare_lstm_data_forecast(hist, time_steps=60):
    data = hist[['Close']].dropna().values
    if len(data) < time_steps + 1:
        raise ValueError("Not enough data to train LSTM model")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    return X.reshape(X.shape[0], X.shape[1], 1), y, scaler


@st.cache_resource(show_spinner=False)
def build_lstm_model_forecast(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


@st.cache_resource(show_spinner=False)
def forecast_lstm_model(hist, period):
    time_steps = 60
    X, y, scaler = prepare_lstm_data_forecast(hist, time_steps)

    if X.shape[0] == 0:
        raise ValueError("Insufficient data for LSTM forecasting.")

    model = build_lstm_model_forecast((X.shape[1], 1))
    model.fit(X, y, epochs=10, batch_size=16, verbose=0, callbacks=[EarlyStopping(patience=3)])

    last_sequence = scaler.transform(hist['Close'].values[-time_steps:].reshape(-1, 1)).flatten()
    predictions = []
    input_seq = list(last_sequence)

    for _ in range(period):
        input_arr = np.array(input_seq[-time_steps:]).reshape(1, time_steps, 1)
        pred = model.predict(input_arr, verbose=0)[0][0]
        predictions.append(pred)
        input_seq.append(pred)

    forecast_vals = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    dates = pd.date_range(start=hist['Date'].iloc[-1] + pd.Timedelta(days=1), periods=period)
    return pd.Series(forecast_vals, index=dates)


@st.cache_resource(show_spinner=False)
def prepare_lgb_features_forecast(hist):
    df = hist.copy()
    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag2'] = df['Close'].shift(2)
    df['Close_lag3'] = df['Close'].shift(3)
    df.dropna(inplace=True)
    if df.shape[0] < 2:
        raise ValueError("Not enough data for LightGBM model")
    features = ['Close_lag1', 'Close_lag2', 'Close_lag3', 'MA50', 'MA200', 'RSI', 'MACD', 'Signal', 'ATR', 'OBV',
                'Momentum']

    available_features = [f for f in features if f in df.columns]

    return df[available_features], df['Close']


@st.cache_resource(show_spinner=False)
def forecast_lgb_model(hist, period):
    X, y = prepare_lgb_features_forecast(hist)

    if X.empty:
        raise ValueError("Insufficient data for LightGBM forecasting.")

    # Ensure enough data for training after feature engineering
    if len(X) <= period + 1:  # Need at least period + 1 for split
        raise ValueError("Not enough data to train LightGBM for forecasting the requested period.")

    model = lgb.LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)
    model.fit(X.iloc[:-period], y.iloc[:-period])
    pred = model.predict(X.iloc[-period:])
    last_date = hist['Date'].iloc[-1]
    dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=period)
    return pd.Series(pred, index=dates)


@st.cache_data(show_spinner=False)
def run_all_forecasts(hist_data, period, models_to_use):
    hist = add_tech_indicators_forecast(hist_data)
    if hist.empty:
        return {}, {}  # Return empty if no valid data after indicators

    forecasts = {}
    model_performance = {}

    for model_name in models_to_use:
        try:
            if model_name == "Prophet":
                forecasts["Prophet"] = forecast_prophet_model(hist, period)
                model_performance["Prophet"] = "Success"
            elif model_name == "SARIMA":
                forecasts["SARIMA"] = forecast_sarima_model(hist, period)
                model_performance["SARIMA"] = "Success"
            elif model_name == "LSTM":
                forecasts["LSTM"] = forecast_lstm_model(hist, period)
                model_performance["LSTM"] = "Success"
            elif model_name == "LightGBM":
                forecasts["LightGBM"] = forecast_lgb_model(hist, period)
                model_performance["LightGBM"] = "Success"
        except Exception as e:
            forecasts[model_name] = pd.Series(dtype=float)
            model_performance[model_name] = f"Error: {str(e)}"
            st.warning(f"Forecast model {model_name} failed: {e}")

    if forecasts:
        all_dates = pd.Index([])
        for fc in forecasts.values():
            if not fc.empty:
                all_dates = all_dates.union(fc.index)

        for key in forecasts.keys():
            if not forecasts[key].empty:
                forecasts[key] = forecasts[key].reindex(all_dates)
            else:
                forecasts[key] = pd.Series(index=all_dates, dtype=float)  # Fill with NaNs for missing forecasts

    return forecasts, model_performance


# ========== 📈 Visualizations for Forecasting ==========
def create_forecast_comparison_chart(hist, forecasts):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist['Date'][-90:],
        y=hist['Close'][-90:],
        mode='lines',
        name='Historical Price',
        line=dict(color='#00d4ff', width=3)
    ))

    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']

    if forecasts:
        combined_df = pd.DataFrame(index=pd.Index([]))
        for model_name, fc_series in forecasts.items():
            if not fc_series.empty:
                combined_df = combined_df.combine_first(fc_series.to_frame(name=model_name))

        if not combined_df.empty:
            for i, model_name in enumerate(combined_df.columns):
                fig.add_trace(go.Scatter(
                    x=combined_df.index,
                    y=combined_df[model_name],
                    mode='lines+markers',
                    name=f'🤖 {model_name}',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))

            if len(combined_df.columns) > 1:
                combined_df['Average'] = combined_df.mean(axis=1, skipna=True)
                if not combined_df['Average'].dropna().empty:
                    fig.add_trace(go.Scatter(
                        x=combined_df.index,
                        y=combined_df['Average'],
                        mode='lines+markers',
                        name='🎯 Ensemble Average',
                        line=dict(color='gold', width=4, dash='solid')
                    ))

    if not hist['Date'].empty:
        fig.add_vline(x=hist['Date'].iloc[-1], line_dash="dot", line_color="white")

    fig.update_layout(
        template='plotly_dark',
        height=600,
        title='🔮 AI-Powered Stock Price Forecasting',
        xaxis_title='📅 Date',
        yaxis_title='💰 Price (₹)',
        hovermode='x unified',
        font=dict(family="Inter", size=12, color="#E0E0E0")
    )
    return fig


def create_technical_indicators_chart(hist):
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('📈 Price & Moving Averages', '📊 RSI', '⚡ MACD', '📉 Bollinger Bands'),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )

    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], name='Close', line=dict(color='#00d4ff')), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['MA50'], name='MA50', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['MA200'], name='MA200', line=dict(color='red')), row=1, col=1)

    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Signal'], name='Signal', line=dict(color='red')), row=3, col=1)

    fig.add_trace(
        go.Scatter(x=hist['Date'], y=hist['Close'], name='Close (BB)', line=dict(color='#00d4ff'), showlegend=False),
        row=4, col=1)
    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['BB_Upper'], name='BB Upper', line=dict(color='cyan', dash='dash'),
                             showlegend=True), row=4, col=1)
    fig.add_trace(
        go.Scatter(x=hist['Date'], y=hist['BB_Lower'], name='BB Lower', line=dict(color='magenta', dash='dash'),
                   showlegend=True), row=4, col=1)
    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['BB_MA'], name='BB MA', line=dict(color='yellow'), showlegend=True),
                  row=4, col=1)

    fig.update_layout(
        template='plotly_dark',
        height=800,
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="right",
        legend_x=1,
        font=dict(family="Inter", size=12, color="#E0E0E0")
    )
    return fig


def create_risk_analysis_chart(hist):
    returns = hist['Close'].pct_change().dropna()

    if returns.empty:
        st.info("Not enough data to perform risk analysis.")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('📊 Returns Distribution', '📈 Price Volatility', '🎲 Current Volatility', '📉 Drawdown Analysis'),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "indicator"}, {"type": "xy"}]]
    )

    fig.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns', marker_color='skyblue', showlegend=False), row=1,
                  col=1)

    if len(returns) >= 30:
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=hist['Date'][30:], y=rolling_vol, name='30-Day Volatility', line=dict(color='orange'),
                       showlegend=False), row=1, col=2)
    else:
        st.warning("Not enough data for 30-day rolling volatility. Skipping plot.")

    current_vol = rolling_vol.iloc[-1] * 100 if 'rolling_vol' in locals() and not rolling_vol.empty else 0
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=current_vol,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Current Volatility %", 'font': {'size': 16}},
        gauge={'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
               'bar': {'color': "#66CCFF"},
               'steps': [{'range': [0, 25], 'color': "#00A36C"},
                         {'range': [25, 50], 'color': "#FFC300"},
                         {'range': [50, 100], 'color': "#C70039"}],
               'threshold': {'line': {'color': "white", 'width': 3},
                             'thickness': 0.75, 'value': 40}}),
        row=2, col=1)

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    fig.add_trace(go.Scatter(x=hist['Date'][1:], y=drawdown * 100,
                             fill='tonexty', name='Drawdown %', line=dict(color='red'), showlegend=False), row=2, col=2)

    fig.update_layout(height=600, template='plotly_dark', title_text="⚠️ Risk Analysis Dashboard",
                      font=dict(family="Inter", size=12, color="#E0E0E0"))
    return fig


def create_forecast_metrics_cards(hist, forecasts, investment, period):
    col1, col2, col3, col4 = st.columns(4)

    last_close = hist['Close'].iloc[-1]

    combined_df = pd.DataFrame(index=pd.Index([]))
    if forecasts:
        for model_name, fc_series in forecasts.items():
            if not fc_series.empty:
                combined_df = combined_df.combine_first(fc_series.to_frame(name=model_name))

        if not combined_df.empty:
            combined_df['Average'] = combined_df.mean(axis=1, skipna=True)
            proj_price = combined_df['Average'].dropna().iloc[-1] if not combined_df[
                'Average'].dropna().empty else last_close
        else:
            proj_price = last_close
    else:
        proj_price = last_close

    profit = proj_price - last_close
    roi_pct = (profit / last_close) * 100
    volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 class="card-icon">💰</h3>
            <h4 class="card-title">Current Price</h4>
            <h2 class="card-value">₹{last_close:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        color = "#00ff88" if roi_pct > 0 else "#ff4757"
        st.markdown(f"""
        <div class="metric-card">
            <h3 class="card-icon">📈</h3>
            <h4 class="card-title">Projected ROI ({period} Days)</h4>
            <h2 class="card-value" style="color:{color};">{roi_pct:+.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 class="card-icon">📊</h3>
            <h4 class="card-title">Annualized Volatility</h4>
            <h2 class="card-value">{volatility:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        proj_value = investment * (proj_price / last_close) if last_close != 0 else investment
        st.markdown(f"""
        <div class="metric-card">
            <h3 class="card-icon">🎯</h3>
            <h4 class="card-title">Proj. Portfolio Value</h4>
            <h2 class="card-value">₹{proj_value:.0f}</h2>
        </div>
        """, unsafe_allow_html=True)


def create_model_performance_summary(model_performance, forecasts):
    st.markdown("<h4 class='section-subtitle'>Model Performance Summary</h4>", unsafe_allow_html=True)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("<h5>Individual Model Status:</h5>", unsafe_allow_html=True)
        for model, status in model_performance.items():
            if status == "Success":
                st.markdown(f"✅ **{model}**: Successfully generated forecast")
            else:
                st.markdown(f"❌ **{model}**: {status}")

    with cols[1]:
        if forecasts and len(forecasts) > 1:
            st.markdown("<h5>Model Agreement:</h5>", unsafe_allow_html=True)
            combined_df = pd.DataFrame({k: v for k, v in forecasts.items() if not v.empty})
            if not combined_df.empty:
                correlation_matrix = combined_df.corr()
                fig = px.imshow(correlation_matrix.values,
                                labels=dict(x="Model", y="Model", color="Correlation"),
                                x=correlation_matrix.columns,
                                y=correlation_matrix.index,
                                color_continuous_scale='RdYlBu',
                                title="🔗 Model Agreement Matrix")
                fig.update_layout(template='plotly_dark', height=300, font=dict(family="Inter", size=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough successful forecasts to show model agreement.")
        else:
            st.info("Select multiple models to see model agreement matrix.")


def display_forecasting(hist_data):
    """Displays the forecasting section with interactive controls and charts."""
    st.markdown(f"<h3 class='section-title'>Stock Price Forecasting</h3>", unsafe_allow_html=True)

    if hist_data is None or hist_data.empty:
        st.info("No historical data available for forecasting.")
        return

    forecast_period = st.slider("Select Forecast Period (Days)", min_value=7, max_value=90, value=30, step=7,
                                key=f"forecast_period_slider_{hist_data.name}")

    selected_models = st.multiselect(
        "Select AI Models for Forecasting",
        options=["Prophet", "SARIMA", "LSTM", "LightGBM"],
        default=["Prophet", "LightGBM"],
        key=f"selected_forecast_models_{hist_data.name}"
    )

    investment_amount = st.number_input("Enter Investment Amount (₹)", min_value=1000, value=10000, step=1000,
                                        key=f"investment_amount_forecast_{hist_data.name}")

    if st.button("Generate Forecast", key=f"generate_forecast_btn_{hist_data.name}"):
        if not selected_models:
            st.error("Please select at least one forecasting model.")
        else:
            with st.spinner(f"Generating {forecast_period}-day forecast..."):
                forecasts, model_performance = run_all_forecasts(hist_data, forecast_period, selected_models)

                if not forecasts:
                    st.warning(
                        "No forecasts could be generated. Please check data availability or try different models.")
                else:
                    create_forecast_metrics_cards(hist_data, forecasts, investment_amount, forecast_period)
                    st.markdown("---")
                    create_model_performance_summary(model_performance, forecasts)
                    st.markdown("---")

                    st.markdown("<h4 class='section-subtitle'>Forecast Comparison Chart</h4>", unsafe_allow_html=True)
                    forecast_chart = create_forecast_comparison_chart(hist_data, forecasts)
                    st.plotly_chart(forecast_chart, use_container_width=True)

                    st.markdown("<h4 class='section-subtitle'>Technical Indicators Chart</h4>", unsafe_allow_html=True)
                    hist_with_indicators = add_tech_indicators_forecast(hist_data)
                    tech_chart = create_technical_indicators_chart(hist_with_indicators)
                    st.plotly_chart(tech_chart, use_container_width=True)

                    st.markdown("<h4 class='section-subtitle'>Risk Analysis Chart</h4>", unsafe_allow_html=True)
                    risk_chart = create_risk_analysis_chart(hist_data)
                    if risk_chart:
                        st.plotly_chart(risk_chart, use_container_width=True)
                    else:
                        st.info("Risk analysis chart could not be generated due to insufficient data.")

                    st.markdown("<h4 class='section-subtitle'>Raw Forecast Data</h4>", unsafe_allow_html=True)
                    combined_forecast_df = pd.DataFrame({k: v for k, v in forecasts.items() if not v.empty})
                    if not combined_forecast_df.empty:
                        combined_forecast_df.index.name = 'Date'
                        st.dataframe(combined_forecast_df)
                    else:
                        st.info("No valid forecast data to display.")

# Note: The if __name__ == "__main__": main() block is removed as this file is now imported as a module.
