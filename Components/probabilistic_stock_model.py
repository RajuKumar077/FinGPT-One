import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import ta  # Library for technical analysis indicators


@st.cache_data(ttl=3600, show_spinner=False)
def compute_rsi(series, period=14):
    """Computes the Relative Strength Index (RSI)."""
    # Using 'ta' library for more robust RSI calculation
    return ta.momentum.RSIIndicator(series, window=period).rsi()


@st.cache_data(ttl=3600, show_spinner=False)
def add_common_features(hist_df):
    """Adds various technical indicators as features common to multiple modules."""
    if hist_df.empty:
        return hist_df

    df = hist_df.copy()
    required_columns = ['Close', 'Volume', 'High', 'Low', 'Open']  # Added 'Open' for more indicators

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

    # --- Basic Features ---
    df['Return_1d'] = df['Close'].pct_change()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()

    # --- Advanced Technical Indicators using 'ta' library ---
    # Momentum Indicators
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
    df['Stoch_K'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    df['Stoch_D'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()

    # Volume Indicators
    df['Volume_MA5'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price() # Using VWAP for Volume MA5, more robust
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

    # Volatility Indicators
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()

    # Price Action Indicators
    df['High_Low_Diff'] = (df['High'] - df['Low']) / df['Close']
    df['Open_Close_Diff'] = (df['Close'] - df['Open']) / df['Open']

    # Target variable: 1 if next day's close > current day's close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df.dropna(inplace=True)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_type='RandomForest'):
    """Trains and evaluates a machine learning model."""
    if model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, None]
        }
    elif model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    else:
        st.error("Unsupported model type.")
        return None, None, None, None

    # Hyperparameter Tuning with GridSearchCV
    # We will control the spinner from the calling function `display_probabilistic_models`
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    st.write(f"**Best {model_type} Parameters:** {grid_search.best_params_}")

    # Cross-validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    st.write(f"**Cross-Validation Accuracy (5-fold):** {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")

    best_model.fit(X_train,
                    y_train)  # Re-fit the best model on the full training set (though GridSearchCV already does this implicitly)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    return best_model, y_pred, y_proba, cv_scores


def display_probabilistic_models(hist_data):
    st.markdown("<h3 class='section-title'>Probabilistic Stock Models (Daily Prediction)</h3>", unsafe_allow_html=True)

    if hist_data.empty:
        st.warning(
            "⚠️ No historical data provided. Please select a stock and load historical data to run probabilistic models.")
        return

    # Validate input data
    required_columns = ['Close', 'Volume', 'High', 'Low', 'Open']
    missing_cols = [col for col in required_columns if col not in hist_data.columns]
    if missing_cols:
        st.error(
            f"❌ Historical data missing required columns: {', '.join(missing_cols)}. Cannot run probabilistic models.")
        return

    # Create a placeholder for status messages and progress
    status_placeholder = st.empty()

    with status_placeholder.container(): # Use a container to hold multiple messages
        with st.spinner("🚀 Preparing data and engineering features..."):
            df = add_common_features(hist_data)

    if df.empty:
        status_placeholder.error("❌ Failed to compute features due to insufficient or invalid data. Check historical data source.")
        return

    # Ensure enough data points
    min_data_points_required = 100  # Increased requirement for more complex features
    if len(df) < min_data_points_required:
        status_placeholder.warning(
            f"⚠️ Not enough data points ({len(df)} < {min_data_points_required}) after feature engineering. Try a ticker with more history or a shorter date range.")
        return

    st.markdown(
        "<p>This section uses machine learning models (Random Forest or Gradient Boosting) to predict whether the stock's closing price will go up (1) or down (0) the next day, based on an extended set of technical indicators.</p>",
        unsafe_allow_html=True)

    # User Configuration for Model
    st.sidebar.subheader("Model Configuration")
    selected_model_type = st.sidebar.selectbox("Choose Model Type:", ["RandomForest", "GradientBoosting"])
    test_size_ratio = st.sidebar.slider("Test Set Size Ratio:", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    # Model Training
    st.markdown("<h5>Model Training & Performance:</h5>", unsafe_allow_html=True)

    features = [
        'Return_1d', 'MA10', 'MA50', 'Volatility', 'RSI', 'MACD', 'MACD_Signal',
        'Stoch_K', 'Stoch_D', 'Volume_MA5', 'OBV', 'ATR', 'High_Low_Diff', 'Open_Close_Diff'
    ]

    # Verify features exist and have no NaNs (after initial dropna, but as a safeguard)
    available_features = [f for f in features if f in df.columns and not df[f].isnull().all()]
    if len(available_features) != len(features):
        st.warning(
            f"⚠️ Some features could not be computed or contain only NaNs and will be excluded: {', '.join(missing)}. Please check data completeness for best results.")
    features = available_features  # Use only computed features

    if not features:
        st.error("❌ No valid features available for model training. Check data and feature engineering steps.")
        return

    X = df[features]
    y = df['Target']

    # Check for sufficient class diversity
    if y.nunique() < 2:
        st.warning(
            "⚠️ Insufficient class diversity (all price movements are up or down) in the target variable. Cannot train a meaningful model.")
        return

    # Train-test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42,
                                                            stratify=y)
        display_metrics = True
    except ValueError as e:
        st.warning(
            f"⚠️ Failed to split data: {e}. Training on full dataset for prediction, but metrics will not be displayed comprehensively.")
        X_train, y_train = X, y
        X_test, y_test = pd.DataFrame(), pd.Series()  # Ensure X_test, y_test are empty if split fails
        display_metrics = False

    # New placeholder for model training status
    model_training_status_placeholder = st.empty()
    with model_training_status_placeholder.container():
        with st.spinner(f"🏋️‍♀️ Training {selected_model_type} model... This includes hyperparameter tuning and cross-validation."):
            best_model, y_pred, y_proba, cv_scores = train_and_evaluate_model(X_train, y_train, X_test, y_test,
                                                                              model_type=selected_model_type)
    
    if best_model is None:
        model_training_status_placeholder.error("Model training failed.")
        return
    else:
        # Replace the spinner with a success message and an expander for details
        model_training_status_placeholder.success(f"✅ {selected_model_type} Model Training Completed!")
        with st.expander("View Training Details"):
            st.write(f"**Best {selected_model_type} Parameters:** {best_model.get_params()}")
            if cv_scores is not None:
                st.write(f"**Cross-Validation Accuracy (5-fold):** {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")


    if display_metrics:
        # Using st.metrics for overall accuracy and AUC score
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Overall Model Accuracy", value=f"{accuracy_score(y_test, y_pred):.2f}")
        try:
            auc_score = roc_auc_score(y_test, y_proba)
            with col2:
                st.metric(label="ROC AUC Score", value=f"{auc_score:.2f}")
        except ValueError:
            st.warning("Could not calculate ROC AUC score (might be due to single class in test set).")

        st.markdown("<h6>Classification Report:</h6>", unsafe_allow_html=True)
        try:
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            # --- Dynamic Classification Report Display (improved) ---
            st.markdown("<p>Detailed performance metrics for each prediction class:</p>", unsafe_allow_html=True)
            
            if '0' in report_dict and '1' in report_dict:
                # Create a DataFrame for better display
                report_data = {
                    'Metric': ['Precision', 'Recall', 'F1-Score', 'Support'],
                    'Class 0 (DOWN) 🔴': [
                        f"{report_dict['0']['precision']:.2f}",
                        f"{report_dict['0']['recall']:.2f}",
                        f"{report_dict['0']['f1-score']:.2f}",
                        int(report_dict['0']['support'])
                    ],
                    'Class 1 (UP) 🟢': [
                        f"{report_dict['1']['precision']:.2f}",
                        f"{report_dict['1']['recall']:.2f}",
                        f"{report_dict['1']['f1-score']:.2f}",
                        int(report_dict['1']['support'])
                    ]
                }
                report_df_display = pd.DataFrame(report_data)
                st.dataframe(report_df_display.set_index('Metric').style.set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])]))


                # --- Dynamic Conclusion (with more emphasis on key metrics) ---
                st.markdown("<h6>Model Performance Summary:</h6>", unsafe_allow_html=True)

                precision_up = report_dict['1']['precision']
                recall_up = report_dict['1']['recall']
                f1_up = report_dict['1']['f1-score']
                support_up = report_dict['1']['support']

                precision_down = report_dict['0']['precision']
                recall_down = report_dict['0']['recall']
                f1_down = report_dict['0']['f1-score']
                support_down = report_dict['0']['support']

                st.markdown(f"""
                <p>The model's overall accuracy on the test set is <b>{accuracy_score(y_test, y_pred):.2f}</b>, meaning it correctly predicts the direction of price movement in approximately {accuracy_score(y_test, y_pred)*100:.0f}% of cases.
                The ROC AUC score of <b>{auc_score:.2f}</b> indicates its ability to distinguish between upward and downward movements, with a score closer to 1 being ideal.</p>
                """, unsafe_allow_html=True)

                st.markdown("<h6>Detailed Class Performance Insights:</h6>", unsafe_allow_html=True)

                col_up, col_down = st.columns(2)

                with col_up:
                    st.markdown(f"<h5 style='color:#32CD32;'>🚀 Upward Movement (Class 1)</h5>", unsafe_allow_html=True)
                    st.metric(label="Precision (Up)", value=f"{precision_up:.2f}", delta_color="off")
                    st.markdown(f"<small><i>When predicting 'Up', the model is correct <b>{precision_up*100:.0f}%</b> of the time.</i></small>", unsafe_allow_html=True)
                    st.metric(label="Recall (Up)", value=f"{recall_up:.2f}", delta_color="off")
                    st.markdown(f"<small><i>It correctly identifies <b>{recall_up*100:.0f}%</b> of all actual 'Up' movements.</i></small>", unsafe_allow_html=True)
                    st.metric(label="F1-Score (Up)", value=f"{f1_up:.2f}", delta_color="off")
                    st.markdown(f"<small><i>Balance between precision and recall for 'Up' predictions.</i></small>", unsafe_allow_html=True)
                    st.info(f"Actual 'Up' Movements: **{int(support_up)}**")

                with col_down:
                    st.markdown(f"<h5 style='color:#FF4500;'>📉 Downward Movement (Class 0)</h5>", unsafe_allow_html=True)
                    st.metric(label="Precision (Down)", value=f"{precision_down:.2f}", delta_color="off")
                    st.markdown(f"<small><i>When predicting 'Down', the model is correct <b>{precision_down*100:.0f}%</b> of the time.</i></small>", unsafe_allow_html=True)
                    st.metric(label="Recall (Down)", value=f"{recall_down:.2f}", delta_color="off")
                    st.markdown(f"<small><i>It correctly identifies <b>{recall_down*100:.0f}%</b> of all actual 'Down' movements.</i></small>", unsafe_allow_html=True)
                    st.metric(label="F1-Score (Down)", value=f"{f1_down:.2f}", delta_color="off")
                    st.markdown(f"<small><i>Balance between precision and recall for 'Down' predictions.</i></small>", unsafe_allow_html=True)
                    st.info(f"Actual 'Down' Movements: **{int(support_down)}**")

                st.markdown("---") # Separator for final conclusion

                # Add a concluding remark based on performance with icons/emojis
                st.markdown("<h6>Overall Model Health:</h6>", unsafe_allow_html=True)
                if f1_up > 0.65 and f1_down > 0.65 and auc_score > 0.75:
                    st.success("✨ **Excellent Performance!** This model demonstrates strong predictive power for both upward and downward movements, and it's highly capable of distinguishing between the two classes. Keep monitoring, but this looks promising.")
                elif f1_up > 0.55 and f1_down > 0.55 and auc_score > 0.65:
                    st.info("👍 **Good Performance.** The model provides reliable predictions, though there might be slight imbalances or areas for fine-tuning to achieve even higher confidence in specific scenarios. A solid foundation!")
                elif (f1_up > 0.5 or f1_down > 0.5) and auc_score > 0.55:
                    st.warning("⚠️ **Moderate Performance.** The model shows potential, but its ability to predict one class might be notably better than the other, or overall distinction needs improvement. Consider further feature engineering or model re-calibration.")
                else:
                    st.error("📉 **Limited Performance.** The model's predictive ability is currently low, struggling to accurately identify price movements. It is **highly recommended** to re-evaluate the features, data quality, or try different modeling approaches. **Do not rely on these predictions for decisions.**")

            else:
                st.warning("Classification report classes (0 or 1) not found. Cannot generate dynamic conclusion or detailed metrics.")


        except ValueError as e:
            st.warning(f"⚠️ Failed to generate classification report: {e}. Predictions may be biased toward one class.")

        # ROC Curve
        if y_test.nunique() > 1:
            st.markdown("<h6>Receiver Operating Characteristic (ROC) Curve:</h6>", unsafe_allow_html=True)
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {auc_score:.2f})'))
            fig_roc.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
            fig_roc.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                template='plotly_dark',
                showlegend=True,
                height=400,
                font=dict(family="Inter", size=12, color="#E0E0E0")
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    # Prediction for Next Day
    st.markdown("<h5>Tomorrow's Prediction:</h5>", unsafe_allow_html=True)

    last_data_point = df.iloc[-1]
    if any(pd.isna(last_data_point[f]) for f in features):
        st.warning(
            "⚠️ Cannot predict: Latest data point has missing or invalid feature values. Ensure your data is up-to-date and complete.")
        return

    last_data_point_features = last_data_point[features].values.reshape(1, -1)
    next_day_prediction = best_model.predict(last_data_point_features)[0]
    next_day_proba_up = best_model.predict_proba(last_data_point_features)[0, 1]

    # Enhanced Tomorrow's Prediction Widget
    col_pred, col_proba = st.columns(2)
    with col_pred:
        prediction_text = "UP" if next_day_prediction == 1 else "DOWN"
        prediction_emoji = "🟢" if next_day_prediction == 1 else "🔴"
        st.metric(label="Predicted Movement for Tomorrow", value=f"{prediction_emoji} {prediction_text}")
    with col_proba:
        # Calculate delta based on deviation from 0.5 (neutral)
        delta_value = (next_day_proba_up - 0.5) * 100
        delta_color = "normal" if abs(delta_value) > 10 else "off" # Only show delta if it's significant

        st.metric(
            label="Probability of Price Going Up",
            value=f"{next_day_proba_up:.2%}",
            delta=f"{delta_value:.2f}% from 50%",
            delta_color=delta_color
        )
    st.markdown(f"<small><i>The model indicates a {next_day_proba_up:.2%} chance of the price closing higher tomorrow.</i></small>", unsafe_allow_html=True)


    # Feature Importance Visualization
    st.markdown("<h5>Feature Importances:</h5>", unsafe_allow_html=True)
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)

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
    else:
        st.info("Feature importance is not available for the selected model type.")

    # Recent Performance Visualization
    st.markdown("<h5>Recent Model Performance (Probability vs. Actual):</h5>", unsafe_allow_html=True)

    max_plot_points = 90  # Extended points for visualization
    recent_data = df.tail(max_plot_points).copy()

    if not recent_data.empty:
        X_recent = recent_data[features]
        # Ensure that X_recent has the same columns as X_train and in the same order
        # This is crucial for consistent prediction
        try:
            y_recent_proba = best_model.predict_proba(X_recent)[:, 1]
            y_recent_actual = recent_data['Target']

            fig_proba = make_subplots(specs=[[{"secondary_y": True}]])
            fig_proba.add_trace(
                go.Scatter(x=recent_data.index, y=recent_data['Close'], name='Close Price', line=dict(color='cyan')),
                secondary_y=False
            )
            fig_proba.add_trace(
                go.Scatter(x=recent_data.index, y=y_recent_proba, name='Predicted Probability (Up)',
                            line=dict(color='orange', dash='dot'), opacity=0.7),
                secondary_y=True
            )
            # Add horizontal lines for common probability thresholds
            fig_proba.add_hline(y=0.5, line_dash="dot", line_color="grey", secondary_y=True,
                                 annotation_text="0.5 Threshold", annotation_position="top left")
            fig_proba.add_hline(y=0.6, line_dash="dot", line_color="green", secondary_y=True,
                                 annotation_text="0.6 Threshold", annotation_position="top right")

            # Plot actual outcomes as markers
            actual_up_dates = recent_data[y_recent_actual == 1].index
            actual_down_dates = recent_data[y_recent_actual == 0].index

            fig_proba.add_trace(
                go.Scatter(
                    x=actual_up_dates,
                    y=[1] * len(actual_up_dates),  # Plot at high end of probability axis
                    mode='markers',
                    name='Actual Up',
                    marker=dict(symbol='triangle-up', size=10, color='lime', line=dict(width=1, color='DarkSlateGrey'))
                ),
                secondary_y=True
            )
            fig_proba.add_trace(
                go.Scatter(
                    x=actual_down_dates,
                    y=[0] * len(actual_down_dates),  # Plot at low end of probability axis
                    mode='markers',
                    name='Actual Down',
                    marker=dict(symbol='triangle-down', size=10, color='red', line=dict(width=1, color='DarkSlateGrey'))
                ),
                secondary_y=True
            )

            fig_proba.update_layout(
                title='📈 Recent Close Price vs. Predicted Up Probability & Actual Outcome',
                xaxis_title='Date',
                yaxis_title='Close Price',
                template='plotly_dark',
                height=600,
                showlegend=True,
                font=dict(family="Inter", size=12, color="#E0E0E0"),
                hovermode="x unified"
            )
            fig_proba.update_yaxes(title_text="Probability/Actual Outcome", secondary_y=True,
                                   range=[-0.1, 1.1])  # Extend range for markers
            st.plotly_chart(fig_proba, use_container_width=True)
        except Exception as e:
            st.error(f"Error during recent performance visualization: {e}. Check data integrity or ensure enough data for prediction.")
    else:
        st.info("⚠️ Not enough recent data to visualize model performance.")

    st.warning("""
    **Disclaimer for Probabilistic Models:**
    This model is for informational purposes only and is a simplified representation of complex market dynamics.
    - **Accuracy:** Model accuracy is based on historical data and does not guarantee future performance. Cross-validation and hyperparameter tuning improve robustness but do not eliminate inherent market unpredictability.
    - **Limitations:** It does not account for external events, news, macroeconomic factors, interest rate changes, company-specific announcements, or significant shifts in human investor behavior. The model assumes stationarity to some extent, which isn't always true in financial markets.
    - **Risk:** Stock markets are inherently volatile and involve substantial risk. **Do not use this prediction for actual investment decisions.** Always conduct your own thorough research and consider consulting with a qualified financial advisor before making any investment choices.
    """)
