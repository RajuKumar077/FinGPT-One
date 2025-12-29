import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & UTILITIES
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive forecasting metrics"""
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred}).dropna()
    if df.empty or len(df) < 2:
        return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'R2': np.nan}
    
    try:
        rmse = np.sqrt(mean_squared_error(df['y_true'], df['y_pred']))
        mae = mean_absolute_error(df['y_true'], df['y_pred'])
        mape = np.mean(np.abs((df['y_true'] - df['y_pred']) / df['y_true'].replace(0, np.nan))) * 100
        r2 = r2_score(df['y_true'], df['y_pred'])
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }
    except:
        return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan, 'R2': np.nan}

def prepare_data_for_forecast(hist_data, target_col='Close', min_days=252):
    """Prepare and validate data for forecasting"""
    if hist_data.empty:
        return None, None
    
    # Ensure datetime index
    if not isinstance(hist_data.index, pd.DatetimeIndex):
        hist_data.index = pd.to_datetime(hist_data.index)
    
    # Remove any NaN values
    data = hist_data[[target_col]].dropna()
    
    # Check data availability
    available_days = len(data)
    
    if available_days < min_days:
        # Provide helpful error message with suggestions
        if available_days < 60:
            st.error(f"❌ Very limited data for forecasting (only {available_days} trading days available)")
            st.info("💡 Suggestions:")
            st.info("- Try a stock with longer trading history")
            st.info("- Some newer stocks or ETFs may have limited data")
            return None, None
        else:
            # Allow proceeding with warning if enough data for basic models
            st.warning(f"⚠️ Limited data for advanced forecasting ({available_days} days, recommended {min_days}+)")
            st.info("📊 Using simplified models suitable for shorter timeframes")
            # Adjust minimum for this run
            min_days = available_days
    
    # Display data info
    st.sidebar.success(f"📈 {available_days} trading days available")
    st.sidebar.info(f"📅 Data range: {data.index[0]} to {data.index[-1]}")
    
    return data, target_col

# ============================================================================
# FORECASTING MODELS
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def forecast_arima(data, target_col, forecast_days=30, order=None):
    """ARIMA Forecasting Model"""
    try:
        series = data[target_col]
        
        # Adaptive order selection based on data size
        if order is None:
            data_len = len(series)
            if data_len < 100:
                order = (1, 1, 1)  # Simple for small datasets
            elif data_len < 252:
                order = (2, 1, 1)  # Moderate for medium datasets
            else:
                order = (5, 1, 0)  # Full for large datasets
        
        # Fit ARIMA model
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        
        # Generate forecast
        forecast = fitted_model.forecast(steps=forecast_days)
        forecast_index = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        # Get confidence intervals
        forecast_df = fitted_model.get_forecast(steps=forecast_days)
        conf_int = forecast_df.conf_int()
        
        return {
            'forecast': pd.Series(forecast.values, index=forecast_index),
            'lower': pd.Series(conf_int.iloc[:, 0].values, index=forecast_index),
            'upper': pd.Series(conf_int.iloc[:, 1].values, index=forecast_index),
            'model': fitted_model,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'order': order
        }
    except Exception as e:
        st.warning(f"⚠️ ARIMA failed: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def forecast_sarima(data, target_col, forecast_days=30, order=(1,1,1), seasonal_order=(1,1,1,7)):
    """SARIMA Forecasting Model (with seasonality)"""
    try:
        series = data[target_col]
        
        # Fit SARIMA model
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        
        # Generate forecast
        forecast = fitted_model.forecast(steps=forecast_days)
        forecast_index = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        # Get confidence intervals
        forecast_df = fitted_model.get_forecast(steps=forecast_days)
        conf_int = forecast_df.conf_int()
        
        return {
            'forecast': pd.Series(forecast.values, index=forecast_index),
            'lower': pd.Series(conf_int.iloc[:, 0].values, index=forecast_index),
            'upper': pd.Series(conf_int.iloc[:, 1].values, index=forecast_index),
            'model': fitted_model,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic
        }
    except Exception as e:
        st.warning(f"⚠️ SARIMA failed: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def forecast_exponential_smoothing(data, target_col, forecast_days=30):
    """Exponential Smoothing (Holt-Winters) Model"""
    try:
        series = data[target_col]
        
        # Fit model
        model = ExponentialSmoothing(
            series,
            trend='add',
            seasonal=None,
            seasonal_periods=None
        )
        fitted_model = model.fit()
        
        # Generate forecast
        forecast = fitted_model.forecast(steps=forecast_days)
        forecast_index = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        return {
            'forecast': pd.Series(forecast.values, index=forecast_index),
            'model': fitted_model
        }
    except Exception as e:
        st.warning(f"⚠️ Exponential Smoothing failed: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def forecast_prophet(data, target_col, forecast_days=30):
    """Facebook Prophet Forecasting Model"""
    try:
        # Prepare data for Prophet
        df_prophet = pd.DataFrame({
            'ds': data.index,
            'y': data[target_col].values
        })
        
        # Fit model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        model.fit(df_prophet)
        
        # Generate forecast
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
        # Extract forecast for future dates only
        forecast_future = forecast[forecast['ds'] > data.index[-1]]
        
        return {
            'forecast': pd.Series(
                forecast_future['yhat'].values,
                index=pd.to_datetime(forecast_future['ds'])
            ),
            'lower': pd.Series(
                forecast_future['yhat_lower'].values,
                index=pd.to_datetime(forecast_future['ds'])
            ),
            'upper': pd.Series(
                forecast_future['yhat_upper'].values,
                index=pd.to_datetime(forecast_future['ds'])
            ),
            'model': model,
            'full_forecast': forecast
        }
    except Exception as e:
        st.warning(f"⚠️ Prophet failed: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def forecast_moving_average(data, target_col, forecast_days=30, window=20):
    """Moving Average Forecast"""
    try:
        series = data[target_col]
        
        # Calculate moving average
        ma = series.rolling(window=window).mean()
        last_ma = ma.iloc[-1]
        
        # Simple forecast: extend last MA value
        forecast_index = pd.date_range(
            start=series.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        # Add slight trend based on recent data
        recent_trend = (series.iloc[-1] - series.iloc[-window]) / window
        forecast_values = [last_ma + (i * recent_trend) for i in range(1, forecast_days + 1)]
        
        return {
            'forecast': pd.Series(forecast_values, index=forecast_index),
            'ma_window': window
        }
    except Exception as e:
        st.warning(f"⚠️ Moving Average failed: {str(e)}")
        return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_forecast_comparison_chart(data, forecasts, target_col, ticker):
    """Create interactive forecast comparison chart"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[target_col],
        name='Historical',
        line=dict(color='#4ECDC4', width=2),
        mode='lines'
    ))
    
    colors = ['#FF6B6B', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9']
    
    # Add each forecast
    for i, (name, forecast_data) in enumerate(forecasts.items()):
        if forecast_data and 'forecast' in forecast_data:
            color = colors[i % len(colors)]
            
            # Main forecast line
            fig.add_trace(go.Scatter(
                x=forecast_data['forecast'].index,
                y=forecast_data['forecast'].values,
                name=name,
                line=dict(color=color, width=2, dash='dash'),
                mode='lines+markers'
            ))
            
            # Confidence intervals if available
            if 'lower' in forecast_data and 'upper' in forecast_data:
                fig.add_trace(go.Scatter(
                    x=forecast_data['forecast'].index,
                    y=forecast_data['upper'].values,
                    fill=None,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_data['forecast'].index,
                    y=forecast_data['lower'].values,
                    fill='tonexty',
                    mode='lines',
                    line=dict(width=0),
                    fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
                    name=f'{name} CI',
                    hoverinfo='skip'
                ))
    
    fig.update_layout(
        title=f'📈 Multi-Model Forecast Comparison - {ticker}',
        template='plotly_dark',
        hovermode='x unified',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_model_performance_chart(model_metrics):
    """Create model performance comparison chart"""
    if not model_metrics:
        return None
    
    metrics_df = pd.DataFrame(model_metrics).T
    metrics_df = metrics_df.dropna()
    
    if metrics_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RMSE (Lower is Better)', 'MAE (Lower is Better)', 
                       'MAPE % (Lower is Better)', 'R² (Higher is Better)'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # RMSE
    if 'RMSE' in metrics_df.columns:
        fig.add_trace(go.Bar(
            x=metrics_df.index,
            y=metrics_df['RMSE'],
            marker_color='#FF6B6B',
            name='RMSE'
        ), row=1, col=1)
    
    # MAE
    if 'MAE' in metrics_df.columns:
        fig.add_trace(go.Bar(
            x=metrics_df.index,
            y=metrics_df['MAE'],
            marker_color='#45B7D1',
            name='MAE'
        ), row=1, col=2)
    
    # MAPE
    if 'MAPE' in metrics_df.columns:
        fig.add_trace(go.Bar(
            x=metrics_df.index,
            y=metrics_df['MAPE'],
            marker_color='#96CEB4',
            name='MAPE'
        ), row=2, col=1)
    
    # R²
    if 'R2' in metrics_df.columns:
        fig.add_trace(go.Bar(
            x=metrics_df.index,
            y=metrics_df['R2'],
            marker_color='#FFEAA7',
            name='R²'
        ), row=2, col=2)
    
    fig.update_layout(
        title='📊 Model Performance Metrics',
        template='plotly_dark',
        height=600,
        showlegend=False
    )
    
    return fig

def create_residual_analysis(data, forecast_result, target_col, model_name):
    """Create residual analysis plot"""
    if not forecast_result or 'forecast' not in forecast_result:
        return None
    
    try:
        # Get overlapping dates
        forecast_dates = forecast_result['forecast'].index
        actual_data = data[target_col].loc[data.index >= forecast_dates[0]]
        
        if len(actual_data) == 0:
            return None
        
        # Align forecast with actual
        common_dates = actual_data.index.intersection(forecast_dates)
        if len(common_dates) == 0:
            return None
        
        actual = actual_data.loc[common_dates]
        predicted = forecast_result['forecast'].loc[common_dates]
        residuals = actual - predicted
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{model_name} - Actual vs Predicted', 'Residuals'),
            vertical_spacing=0.15
        )
        
        # Actual vs Predicted
        fig.add_trace(go.Scatter(
            x=common_dates, y=actual.values,
            name='Actual', mode='lines+markers',
            line=dict(color='#4ECDC4', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=common_dates, y=predicted.values,
            name='Predicted', mode='lines+markers',
            line=dict(color='#FF6B6B', width=2, dash='dash')
        ), row=1, col=1)
        
        # Residuals
        fig.add_trace(go.Scatter(
            x=common_dates, y=residuals.values,
            name='Residuals', mode='lines+markers',
            line=dict(color='#96CEB4', width=2)
        ), row=2, col=1)
        
        fig.add_hline(y=0, line_dash="dash", line_color="white", 
                     opacity=0.5, row=2, col=1)
        
        fig.update_layout(
            template='plotly_dark',
            height=600,
            showlegend=True
        )
        
        return fig
    except Exception as e:
        return None

# ============================================================================
# MAIN DISPLAY FUNCTION
# ============================================================================

def display_forecasting(hist_data, ticker):
    """Main forecasting dashboard display"""
    
    # Header
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0;'>🔮 Advanced Forecasting Engine</h1>
        <h2 style='color: #e0e0e0; margin-top: 10px;'>{ticker}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare data with configurable minimum days
    min_data_days = st.sidebar.slider("Minimum Historical Data (Days)", 60, 1000, 252, step=30)
    data, target_col = prepare_data_for_forecast(hist_data, min_days=min_data_days)
    if data is None:
        return
    
    # Sidebar controls
    st.sidebar.markdown("### 🎯 Forecast Settings")
    forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 7, 90, 30, step=7)
    
    # Adjust available models based on data size
    available_models = ['Moving Average', 'Exponential Smoothing']
    if len(data) >= 252:  # 1 year+
        available_models.extend(['ARIMA'])
    if len(data) >= 365:  # 1+ year with seasonality
        available_models.extend(['SARIMA'])
    if len(data) >= 365:  # Prophet needs at least 1 year
        available_models.extend(['Prophet'])
    
    # Default models based on data availability
    default_models = ['Moving Average', 'Exponential Smoothing']
    if len(data) >= 252:
        default_models = ['ARIMA', 'Moving Average']
    
    selected_models = st.sidebar.multiselect(
        "Select Models",
        available_models,
        default=default_models
    )
    
    if not selected_models:
        st.warning("⚠️ Please select at least one forecasting model")
        return
    
    # Show data limitations
    if len(data) < 252:
        st.sidebar.warning("⚠️ Limited data detected")
        st.sidebar.info("Advanced models (ARIMA, SARIMA, Prophet) require 1+ years of data")
    
    if len(data) < 365:
        st.sidebar.info("📅 Seasonal models work best with 1+ years of data")
    
    # Generate forecasts
    forecasts = {}
    
    with st.spinner("🔄 Training models and generating forecasts..."):
        if 'ARIMA' in selected_models:
            forecasts['ARIMA'] = forecast_arima(data, target_col, forecast_days)
        
        if 'SARIMA' in selected_models:
            forecasts['SARIMA'] = forecast_sarima(data, target_col, forecast_days)
        
        if 'Prophet' in selected_models:
            forecasts['Prophet'] = forecast_prophet(data, target_col, forecast_days)
        
        if 'Exponential Smoothing' in selected_models:
            forecasts['Exponential Smoothing'] = forecast_exponential_smoothing(data, target_col, forecast_days)
        
        if 'Moving Average' in selected_models:
            forecasts['Moving Average'] = forecast_moving_average(data, target_col, forecast_days)
    
    # Filter out failed forecasts
    forecasts = {k: v for k, v in forecasts.items() if v is not None}
    
    if not forecasts:
        st.error("❌ All forecasting models failed. Please try different settings.")
        return
    
    # === FORECAST SUMMARY ===
    st.markdown("### 📊 Forecast Summary")
    
    cols = st.columns(len(forecasts))
    for i, (model_name, forecast_data) in enumerate(forecasts.items()):
        with cols[i]:
            forecast_val = forecast_data['forecast'].iloc[-1]
            current_val = data[target_col].iloc[-1]
            change = ((forecast_val - current_val) / current_val) * 100
            
            st.markdown(f"""
            <div style='background: #1e1e1e; padding: 20px; border-radius: 10px; text-align: center;'>
                <h4 style='margin: 0; color: #888;'>{model_name}</h4>
                <h2 style='color: #4ECDC4; margin: 10px 0;'>${forecast_val:.2f}</h2>
                <p style='color: {"#4ECDC4" if change > 0 else "#FF6B6B"}; margin: 0;'>
                    {change:+.2f}% from current
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # === MAIN FORECAST CHART ===
    st.markdown("### 📈 Multi-Model Forecast Comparison")
    forecast_chart = create_forecast_comparison_chart(data, forecasts, target_col, ticker)
    st.plotly_chart(forecast_chart, use_container_width=True)
    
    # === TABS FOR DETAILED ANALYSIS ===
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Details", "📉 Residual Analysis", "📋 Forecast Data", "ℹ️ Methodology"
    ])
    
    with tab1:
        st.subheader("Model Performance & Statistics")
        
        for model_name, forecast_data in forecasts.items():
            with st.expander(f"📌 {model_name} Details", expanded=False):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**Model Statistics:**")
                    if 'aic' in forecast_data:
                        st.metric("AIC", f"{forecast_data['aic']:.2f}")
                    if 'bic' in forecast_data:
                        st.metric("BIC", f"{forecast_data['bic']:.2f}")
                    
                    # Forecast range
                    st.markdown("**Forecast Range:**")
                    st.write(f"Min: ${forecast_data['forecast'].min():.2f}")
                    st.write(f"Max: ${forecast_data['forecast'].max():.2f}")
                    st.write(f"Mean: ${forecast_data['forecast'].mean():.2f}")
                
                with col2:
                    # Individual forecast chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data.index[-60:], y=data[target_col].iloc[-60:],
                        name='Historical', line=dict(color='#4ECDC4')
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast_data['forecast'].index,
                        y=forecast_data['forecast'].values,
                        name='Forecast', line=dict(color='#FF6B6B', dash='dash')
                    ))
                    fig.update_layout(
                        title=f'{model_name} Forecast',
                        template='plotly_dark',
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Residual Analysis")
        st.info("📊 Residual analysis helps evaluate model accuracy on historical data")
        
        for model_name, forecast_data in forecasts.items():
            residual_fig = create_residual_analysis(data, forecast_data, target_col, model_name)
            if residual_fig:
                st.plotly_chart(residual_fig, use_container_width=True)
    
    with tab3:
        st.subheader("Detailed Forecast Data")
        
        for model_name, forecast_data in forecasts.items():
            with st.expander(f"📊 {model_name} Forecast Values", expanded=False):
                forecast_df = pd.DataFrame({
                    'Date': forecast_data['forecast'].index,
                    'Forecast': forecast_data['forecast'].values
                })
                
                if 'lower' in forecast_data:
                    forecast_df['Lower Bound'] = forecast_data['lower'].values
                if 'upper' in forecast_data:
                    forecast_df['Upper Bound'] = forecast_data['upper'].values
                
                st.dataframe(
                    forecast_df.style.format({
                        'Forecast': '${:.2f}',
                        'Lower Bound': '${:.2f}',
                        'Upper Bound': '${:.2f}'
                    }),
                    use_container_width=True
                )
                
                # Download button
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download {model_name} Forecast",
                    data=csv,
                    file_name=f"{ticker}_{model_name}_forecast.csv",
                    mime="text/csv"
                )
    
    with tab4:
        st.subheader("📚 Forecasting Methodology")
        
        st.markdown("""
        ### Available Models:
        
        **ARIMA (AutoRegressive Integrated Moving Average)**
        - Best for: Time series with trends and no strong seasonality
        - Uses: Past values and forecast errors to predict future
        - Parameters: p (autoregressive), d (differencing), q (moving average)
        
        **SARIMA (Seasonal ARIMA)**
        - Best for: Time series with seasonal patterns
        - Extension of ARIMA with seasonal components
        - Captures weekly, monthly, or yearly patterns
        
        **Prophet (Facebook Prophet)**
        - Best for: Business time series with strong seasonality
        - Handles missing data and outliers well
        - Automatically detects changepoints
        
        **Exponential Smoothing (Holt-Winters)**
        - Best for: Time series with level and trend
        - Gives more weight to recent observations
        - Simple and computationally efficient
        
        **Moving Average**
        - Best for: Short-term forecasting baseline
        - Uses average of recent values
        - Simple benchmark model
        
        ### Metrics Explained:
        - **RMSE**: Root Mean Square Error - overall forecast accuracy
        - **MAE**: Mean Absolute Error - average forecast error
        - **MAPE**: Mean Absolute Percentage Error - error as percentage
        - **R²**: Coefficient of determination - model fit quality
        
        ### Important Notes:
        ⚠️ **Risk Disclaimer**: These forecasts are statistical projections based on historical data. 
        They do NOT account for future events, market conditions, or fundamental changes.
        
        📊 **Best Practice**: Use multiple models and compare results. The "best" model 
        varies by stock and market conditions.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>🔮 Forecasts generated using {len(forecasts)} model(s) | 
        Horizon: {forecast_days} days | Last Price: ${data[target_col].iloc[-1]:.2f}</p>
        <p style='font-size: 12px;'>⚠️ For informational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)