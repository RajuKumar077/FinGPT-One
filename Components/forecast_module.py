import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from prophet import Prophet
except Exception:
    Prophet = None

warnings.filterwarnings("ignore")

CHART_PAPER = "rgba(255,255,255,0.02)"
CHART_PLOT = "rgba(255,255,255,0.01)"
TEXT_COLOR = "#E8EEF9"
GRID_COLOR = "rgba(255,255,255,0.08)"
FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", sans-serif'


def apply_chart_theme(fig, title=None, height=520):
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor=CHART_PAPER,
        plot_bgcolor=CHART_PLOT,
        font=dict(color=TEXT_COLOR, family=FONT_FAMILY),
        hovermode="x unified",
        margin=dict(l=24, r=24, t=60, b=24),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig


def calculate_metrics(y_true, y_pred):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if df.empty or len(df) < 2:
        return {"RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "R2": np.nan}

    rmse = float(np.sqrt(mean_squared_error(df["y_true"], df["y_pred"])))
    mae = float(mean_absolute_error(df["y_true"], df["y_pred"]))
    mape = float(np.mean(np.abs((df["y_true"] - df["y_pred"]) / df["y_true"].replace(0, np.nan))) * 100)
    r2 = float(r2_score(df["y_true"], df["y_pred"])) if len(df) > 2 else np.nan
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def prepare_data_for_forecast(hist_data, target_col="Close", min_days=252):
    if hist_data.empty:
        return None, None

    if not isinstance(hist_data.index, pd.DatetimeIndex):
        hist_data = hist_data.copy()
        hist_data.index = pd.to_datetime(hist_data.index)

    data = hist_data[[target_col]].dropna().sort_index()
    available_days = len(data)

    if available_days < min_days:
        if available_days < 60:
            st.error(f"Very limited data for forecasting: only {available_days} trading days are available.")
            st.info("Try a stock with a longer price history for stronger forecasts.")
            return None, None
        st.warning(f"Limited data for advanced forecasting: {available_days} days available, {min_days}+ recommended.")

    st.sidebar.success(f"{available_days} trading days available")
    st.sidebar.info(f"Range: {data.index[0].date()} to {data.index[-1].date()}")
    return data, target_col


def build_future_index(last_index, forecast_days):
    start = pd.Timestamp(last_index) + pd.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=forecast_days)


def confidence_band_from_residuals(prediction, residual_std):
    if residual_std is None or np.isnan(residual_std):
        return None, None
    steps = np.arange(1, len(prediction) + 1)
    margin = 1.96 * residual_std * np.sqrt(steps)
    return prediction - margin, prediction + margin


def moving_average_prediction(train_series, forecast_days, window=20):
    effective_window = max(5, min(window, len(train_series)))
    base = train_series.rolling(window=effective_window, min_periods=3).mean().iloc[-1]
    drift_window = max(5, min(20, len(train_series) - 1))
    if drift_window <= 1:
        slope = 0.0
    else:
        slope = (train_series.iloc[-1] - train_series.iloc[-drift_window]) / drift_window
    values = np.array([base + slope * step for step in range(1, forecast_days + 1)])
    return pd.Series(values, index=build_future_index(train_series.index[-1], forecast_days))


def fit_arima(train_series, forecast_days):
    data_len = len(train_series)
    order = (1, 1, 1) if data_len < 120 else (2, 1, 2) if data_len < 300 else (4, 1, 2)
    model = ARIMA(train_series, order=order)
    fitted = model.fit()
    pred = pd.Series(fitted.forecast(steps=forecast_days).values, index=build_future_index(train_series.index[-1], forecast_days))
    return {"forecast": pred, "model": fitted, "aic": getattr(fitted, "aic", np.nan), "bic": getattr(fitted, "bic", np.nan), "order": order}


def fit_sarima(train_series, forecast_days):
    model = SARIMAX(
        train_series,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 5),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    pred = pd.Series(fitted.forecast(steps=forecast_days).values, index=build_future_index(train_series.index[-1], forecast_days))
    return {"forecast": pred, "model": fitted, "aic": getattr(fitted, "aic", np.nan), "bic": getattr(fitted, "bic", np.nan)}


def fit_exponential_smoothing(train_series, forecast_days):
    trend = "add" if len(train_series) >= 40 else None
    seasonal = "add" if len(train_series) >= 180 else None
    seasonal_periods = 5 if seasonal else None
    model = ExponentialSmoothing(train_series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    fitted = model.fit(optimized=True)
    pred = pd.Series(fitted.forecast(steps=forecast_days).values, index=build_future_index(train_series.index[-1], forecast_days))
    return {"forecast": pred, "model": fitted}


def fit_prophet(train_series, forecast_days):
    if Prophet is None:
        raise RuntimeError("Prophet is not installed in this environment.")
    prophet_df = pd.DataFrame({"ds": train_series.index, "y": train_series.values})
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.03,
        seasonality_mode="additive",
    )
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_days, freq="B")
    forecast = model.predict(future).tail(forecast_days)
    pred = pd.Series(forecast["yhat"].values, index=pd.to_datetime(forecast["ds"]))
    lower = pd.Series(forecast["yhat_lower"].values, index=pd.to_datetime(forecast["ds"]))
    upper = pd.Series(forecast["yhat_upper"].values, index=pd.to_datetime(forecast["ds"]))
    return {"forecast": pred, "lower": lower, "upper": upper, "model": model}


MODEL_BUILDERS = {
    "Moving Average": lambda series, days: {"forecast": moving_average_prediction(series, days), "ma_window": 20},
    "Exponential Smoothing": fit_exponential_smoothing,
    "ARIMA": fit_arima,
    "SARIMA": fit_sarima,
}
if Prophet is not None:
    MODEL_BUILDERS["Prophet"] = fit_prophet


def get_validation_window(data_length, forecast_days):
    return max(10, min(forecast_days, max(10, data_length // 8)))


def evaluate_model(train_series, test_series, model_name):
    try:
        model_result = MODEL_BUILDERS[model_name](train_series, len(test_series))
        forecast = model_result["forecast"]
        aligned = pd.DataFrame({"actual": test_series.values, "pred": forecast.values}, index=test_series.index)
        metrics = calculate_metrics(aligned["actual"], aligned["pred"])
        residual_std = float((aligned["actual"] - aligned["pred"]).std(ddof=1)) if len(aligned) > 1 else np.nan
        model_result["validation_metrics"] = metrics
        model_result["validation_actual"] = aligned["actual"]
        model_result["validation_pred"] = aligned["pred"]
        model_result["residual_std"] = residual_std
        return model_result
    except Exception as exc:
        st.warning(f"{model_name} validation failed: {exc}")
        return None


def build_live_model(data, model_name, forecast_days, validation_metrics=None, residual_std=None):
    result = MODEL_BUILDERS[model_name](data, forecast_days)
    result["validation_metrics"] = validation_metrics or {}
    if "lower" not in result or "upper" not in result:
        lower, upper = confidence_band_from_residuals(result["forecast"], residual_std)
        if lower is not None and upper is not None:
            result["lower"] = lower
            result["upper"] = upper
    result["residual_std"] = residual_std
    return result


def create_weighted_ensemble(forecasts, forecast_days):
    eligible = {}
    for name, result in forecasts.items():
        metrics = result.get("validation_metrics", {})
        rmse = metrics.get("RMSE")
        if result.get("forecast") is not None and rmse and not np.isnan(rmse) and rmse > 0:
            eligible[name] = 1 / rmse

    if len(eligible) < 2:
        return None

    total = sum(eligible.values())
    weights = {name: score / total for name, score in eligible.items()}
    forecast_index = next(iter(forecasts.values()))["forecast"].index
    stacked = np.zeros(forecast_days)
    variance = np.zeros(forecast_days)

    for name, weight in weights.items():
        series = forecasts[name]["forecast"].values
        stacked += series * weight
        residual_std = forecasts[name].get("residual_std")
        if residual_std and not np.isnan(residual_std):
            variance += ((residual_std * np.sqrt(np.arange(1, forecast_days + 1))) ** 2) * (weight ** 2)

    ensemble_forecast = pd.Series(stacked, index=forecast_index)
    ensemble_std = np.sqrt(variance) if np.any(variance) else None
    lower = ensemble_forecast - 1.96 * ensemble_std if ensemble_std is not None else None
    upper = ensemble_forecast + 1.96 * ensemble_std if ensemble_std is not None else None

    weighted_rmse = sum(forecasts[name]["validation_metrics"]["RMSE"] * weight for name, weight in weights.items())
    weighted_mae = sum(forecasts[name]["validation_metrics"]["MAE"] * weight for name, weight in weights.items())
    weighted_mape = sum(forecasts[name]["validation_metrics"]["MAPE"] * weight for name, weight in weights.items())

    return {
        "forecast": ensemble_forecast,
        "lower": lower,
        "upper": upper,
        "validation_metrics": {
            "RMSE": weighted_rmse,
            "MAE": weighted_mae,
            "MAPE": weighted_mape,
            "R2": np.nan,
        },
        "weights": weights,
        "residual_std": float(np.mean([forecasts[name]["residual_std"] for name in weights if forecasts[name].get("residual_std") is not None])),
    }


def create_forecast_comparison_chart(data, forecasts, target_col, ticker):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[target_col],
            name="Historical",
            mode="lines",
            line=dict(color="#C4D5F9", width=2.5),
        )
    )

    palette = ["#79B8FF", "#7EE0C3", "#F8C471", "#F6A6B2", "#A6B8FF", "#F4F7FB"]
    for idx, (model_name, forecast_data) in enumerate(forecasts.items()):
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=forecast_data["forecast"].index,
                y=forecast_data["forecast"].values,
                name=model_name,
                mode="lines+markers",
                line=dict(color=color, width=2.5, dash="dash" if model_name != "AI Ensemble" else "solid"),
                marker=dict(size=5),
            )
        )
        if forecast_data.get("lower") is not None and forecast_data.get("upper") is not None:
            fig.add_trace(
                go.Scatter(
                    x=forecast_data["forecast"].index,
                    y=forecast_data["upper"].values,
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_data["forecast"].index,
                    y=forecast_data["lower"].values,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(255,255,255,0.06)" if model_name == "AI Ensemble" else "rgba(121,184,255,0.07)",
                    hoverinfo="skip",
                    name=f"{model_name} range",
                    showlegend=False,
                )
            )

    return apply_chart_theme(fig, f"{ticker} forecast comparison", 580)


def create_model_performance_chart(model_metrics):
    if not model_metrics:
        return None
    metrics_df = pd.DataFrame(model_metrics).T.sort_values("RMSE")
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("RMSE", "MAE", "MAPE"),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]],
    )
    fig.add_trace(go.Bar(x=metrics_df.index, y=metrics_df["RMSE"], marker_color="#79B8FF", name="RMSE"), row=1, col=1)
    fig.add_trace(go.Bar(x=metrics_df.index, y=metrics_df["MAE"], marker_color="#7EE0C3", name="MAE"), row=1, col=2)
    fig.add_trace(go.Bar(x=metrics_df.index, y=metrics_df["MAPE"], marker_color="#F8C471", name="MAPE"), row=1, col=3)
    fig.update_layout(showlegend=False)
    return apply_chart_theme(fig, "Validation scorecard", 420)


def create_validation_chart(validation_actual, validation_pred, model_name):
    if validation_actual is None or validation_pred is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=validation_actual.index, y=validation_actual.values, name="Actual", line=dict(color="#C4D5F9", width=2.4)))
    fig.add_trace(go.Scatter(x=validation_pred.index, y=validation_pred.values, name="Predicted", line=dict(color="#79B8FF", width=2.1, dash="dash")))
    return apply_chart_theme(fig, f"{model_name} validation window", 340)


def render_metric_card(label, value, delta_text, accent="#79B8FF"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta" style="color:{accent};">{delta_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_forecasting(hist_data, ticker):
    st.markdown(
        f"""
        <section class="hero-panel">
            <div class="hero-kicker">Forecast Lab</div>
            <h1>{ticker} predictive outlook</h1>
            <p>Validation-ranked models, weighted ensemble forecasts, and cleaner risk ranges for smarter price projections.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    min_data_days = st.sidebar.slider("Minimum historical data (days)", 60, 1000, 252, step=30)
    data, target_col = prepare_data_for_forecast(hist_data, min_days=min_data_days)
    if data is None:
        return

    forecast_days = st.sidebar.slider("Forecast horizon (days)", 7, 90, 30, step=7)
    validation_days = min(get_validation_window(len(data), forecast_days), max(10, len(data) // 4))

    available_models = ["Moving Average", "Exponential Smoothing"]
    if len(data) >= 220:
        available_models.append("ARIMA")
    if len(data) >= 320:
        available_models.append("SARIMA")
    if len(data) >= 365 and Prophet is not None:
        available_models.append("Prophet")

    default_models = [m for m in ["ARIMA", "Exponential Smoothing", "Moving Average"] if m in available_models]
    selected_models = st.sidebar.multiselect("Models", available_models, default=default_models)
    if not selected_models:
        st.warning("Select at least one forecasting model.")
        return

    if len(data) <= validation_days + 30:
        st.warning("Not enough history to reserve a meaningful validation window.")
        return

    train_data = data.iloc[:-validation_days]
    validation_data = data.iloc[-validation_days:]

    live_forecasts = {}
    model_metrics = {}

    with st.spinner("Scoring models and building the ensemble..."):
        for model_name in selected_models:
            validation_result = evaluate_model(train_data[target_col], validation_data[target_col], model_name)
            if validation_result is None:
                continue

            live_result = build_live_model(
                data[target_col],
                model_name,
                forecast_days,
                validation_metrics=validation_result.get("validation_metrics"),
                residual_std=validation_result.get("residual_std"),
            )
            live_result["validation_actual"] = validation_result.get("validation_actual")
            live_result["validation_pred"] = validation_result.get("validation_pred")
            live_forecasts[model_name] = live_result
            model_metrics[model_name] = validation_result["validation_metrics"]

        ensemble_result = create_weighted_ensemble(live_forecasts, forecast_days)
        if ensemble_result is not None:
            live_forecasts = {"AI Ensemble": ensemble_result, **live_forecasts}
            model_metrics["AI Ensemble"] = ensemble_result["validation_metrics"]

    if not live_forecasts:
        st.error("All forecasting models failed on this ticker. Try another symbol or a shorter horizon.")
        return

    best_model = min(model_metrics.items(), key=lambda item: item[1]["RMSE"] if not np.isnan(item[1]["RMSE"]) else np.inf)[0]

    st.markdown("### Snapshot")
    cols = st.columns(min(4, len(live_forecasts)))
    for idx, (model_name, forecast_data) in enumerate(list(live_forecasts.items())[:4]):
        with cols[idx]:
            forecast_val = forecast_data["forecast"].iloc[-1]
            current_val = data[target_col].iloc[-1]
            change = ((forecast_val - current_val) / current_val) * 100
            accent = "#7EE0C3" if change >= 0 else "#F6A6B2"
            render_metric_card(model_name, f"${forecast_val:,.2f}", f"{change:+.2f}% vs current", accent)

    insight_col, source_col, validation_col = st.columns([1.2, 1, 1])
    with insight_col:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="section-kicker">Best Validation Fit</div>
                <h3 style="margin-bottom:0.35rem;">{best_model}</h3>
                <p style="margin:0;color:var(--text-secondary);">
                    Lowest validation RMSE over the last {validation_days} trading days.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with source_col:
        best_rmse = model_metrics[best_model]["RMSE"]
        render_metric_card("Validation RMSE", f"{best_rmse:,.2f}", "Lower is better", "#79B8FF")
    with validation_col:
        best_mape = model_metrics[best_model]["MAPE"]
        render_metric_card("Validation MAPE", f"{best_mape:,.2f}%", "Percent error", "#F8C471")

    st.markdown("### Forecast Comparison")
    st.plotly_chart(create_forecast_comparison_chart(data, live_forecasts, target_col, ticker), use_container_width=True)

    performance_chart = create_model_performance_chart(model_metrics)
    if performance_chart is not None:
        st.plotly_chart(performance_chart, use_container_width=True)

    tabs = st.tabs(["Leaderboard", "Validation", "Forecast Data", "Methodology"])

    with tabs[0]:
        leaderboard = pd.DataFrame(model_metrics).T.sort_values("RMSE").round(3)
        st.dataframe(leaderboard, use_container_width=True)
        if "AI Ensemble" in live_forecasts and live_forecasts["AI Ensemble"].get("weights"):
            weights = live_forecasts["AI Ensemble"]["weights"]
            weight_df = pd.DataFrame({"Model": list(weights.keys()), "Weight": list(weights.values())}).sort_values("Weight", ascending=False)
            st.markdown("#### Ensemble Weights")
            st.dataframe(weight_df.style.format({"Weight": "{:.1%}"}), use_container_width=True)

    with tabs[1]:
        for model_name, forecast_data in live_forecasts.items():
            validation_chart = create_validation_chart(
                forecast_data.get("validation_actual"),
                forecast_data.get("validation_pred"),
                model_name,
            )
            if validation_chart is not None:
                st.plotly_chart(validation_chart, use_container_width=True)

    with tabs[2]:
        for model_name, forecast_data in live_forecasts.items():
            frame = pd.DataFrame({"Date": forecast_data["forecast"].index, "Forecast": forecast_data["forecast"].values})
            if forecast_data.get("lower") is not None:
                frame["Lower Bound"] = forecast_data["lower"].values
            if forecast_data.get("upper") is not None:
                frame["Upper Bound"] = forecast_data["upper"].values
            with st.expander(f"{model_name} forecast", expanded=(model_name == "AI Ensemble")):
                st.dataframe(frame.style.format({col: "${:,.2f}" for col in frame.columns if col != "Date"}), use_container_width=True)

    with tabs[3]:
        st.markdown(
            """
            - Models are scored on a holdout validation window instead of only showing raw future projections.
            - The app retrains each successful model on the full series after scoring it.
            - `AI Ensemble` blends the better models using inverse-RMSE weights, so stronger recent performers contribute more.
            - Confidence bands for non-probabilistic models are estimated from validation residuals.
            - Forecasts are still statistical estimates, not investment advice.
            """
        )
