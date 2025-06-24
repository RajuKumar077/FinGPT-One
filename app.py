import streamlit as st
import pandas as pd
import numpy as np
import warnings
import time
import requests
import json
import yfinance as yf # Primary for historical data
import os # Imported for potential future path operations if needed, but not for local CSV fallback.

# Import functions from your separate modules
import pages.fmp_autocomplete as fmp_autocomplete
import pages.stock_summary as stock_summary
import pages.financials as financials
import pages.probabilistic_stock_model as probabilistic_stock_model
import pages.forecast_module as forecast_module
import pages.news_sentiment as news_sentiment

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# --- GLOBAL CONFIGURATIONS AND INITIAL STREAMLIT SETUP ---
st.set_page_config(
    page_title="Intelligent Stock Insights",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get help': None,
        'Report a bug': None,
        'About': None
    },
    page_icon="📈"
)

# Custom CSS to hide the Streamlit hamburger menu icon (sidebar toggle) and the header
st.markdown("""
    <style>
        /* Hide the Streamlit hamburger menu icon */
        button[data-testid="stSidebarToggle"] {
            display: none !important;
        }
        /* Hide the entire Streamlit header which often contains the sidebar toggle and other default elements */
        header {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# API Keys (ALL PROVIDED KEYS ARE NOW EMBEDDED)
NEWS_API_KEY = "874ba654bdcd4aa7b68f7367a907cc2f" # Your NewsAPI key
FMP_API_KEY = "5C9DnMCAzYam2ZPjNpOxKLFxUiGhrJDD"     # Your FMP key
GEMINI_API_KEY = "AIzaSyAK8BevJ1wIrwMoYDsnCLQXdZlFglF92WE" # Your Gemini key
ALPHA_VANTAGE_API_KEY = "WLVUE35CQ906QK3K" # Your Alpha Vantage key

# --- Custom CSS and Font Loading ---
def load_css(file_path):
    """Loads custom CSS from a file."""
    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Error: CSS file not found at {file_path}. Please ensure 'assets/style.css' exists.")

st.markdown(
    "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap' rel='stylesheet'>",
    unsafe_allow_html=True)
load_css("assets/style.css")


# --- Historical Data Loading (yfinance, Alpha Vantage, and FMP Fallback - API ONLY) ---
@st.cache_data(ttl=3600, show_spinner=False)  # Cache historical data for 1 hour
def load_historical_data(ticker_symbol, alpha_vantage_api_key, fmp_api_key):
    """
    Loads historical stock data, first attempting yfinance, then Alpha Vantage, then FMP (various endpoints).
    This function relies purely on online APIs as per user's request.
    """
    if not ticker_symbol:
        return pd.DataFrame()

    hist_df = pd.DataFrame()

    # --- Attempt 1: Try yfinance with multiple periods ---
    st.info(f"Attempt 1/4: Trying to load historical data for {ticker_symbol} using yfinance (primary source)...")
    periods_to_try_yf = ["max", "5y", "2y", "1y", "6mo", "3mo", "1mo"] # Ordered from longest to shortest

    for period in periods_to_try_yf:
        try:
            with st.spinner(f"YFinance for {ticker_symbol} (period: {period})..."):
                ticker = yf.Ticker(ticker_symbol)
                hist_df_yf = ticker.history(period=period, auto_adjust=True, timeout=15)

            if not hist_df_yf.empty:
                hist_df_yf.reset_index(inplace=True)
                hist_df_yf.rename(columns={
                    'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low',
                    'Close': 'Close', 'Volume': 'Volume'
                }, inplace=True)
                hist_df_yf['Date'] = pd.to_datetime(hist_df_yf['Date']).dt.date
                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                hist_df = hist_df_yf[required_cols]
                hist_df.sort_values(by='Date', ascending=True, inplace=True)
                hist_df.reset_index(drop=True, inplace=True)
                st.success(f"✅ Successfully loaded historical data for {ticker_symbol} using yfinance (period: '{period}').")
                print(f"DEBUG: YFinance data loaded for {ticker_symbol} with {len(hist_df)} rows.")
                return hist_df
            else:
                print(f"DEBUG: yfinance returned empty data for {ticker_symbol} with period '{period}'. Trying next yfinance period.")
                continue

        except requests.exceptions.RequestException as req_err:
            print(f"DEBUG: yfinance network error for {ticker_symbol} ({period}): {req_err}")
            st.warning(f"⚠️ YFinance network error for {ticker_symbol} (period: {period}). Trying next yfinance period.")
            time.sleep(1)
            continue
        except Exception as e:
            print(f"DEBUG: Generic yfinance error for {ticker_symbol} ({period}): {e}")
            st.warning(f"⚠️ YFinance data issue for {ticker_symbol} (period: {period}): {e}. This often indicates a temporary data source problem. Trying next period.")
            time.sleep(1)
            continue

    # --- Attempt 2: Fallback to Alpha Vantage if yfinance completely failed ---
    st.info(f"Attempt 2/4: YFinance failed for {ticker_symbol}. Falling back to Alpha Vantage...")
    # Alpha Vantage API key is now definitively set at the top of the file
    if not alpha_vantage_api_key or alpha_vantage_api_key == "YOUR_ALPHA_VANTAGE_API_KEY":
        st.error("❌ Alpha Vantage API key is not set. Cannot use Alpha Vantage as a fallback. Please update `app.py`.")
    else:
        alpha_vantage_url = "https://www.alphavantage.co/query"
        params_av = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker_symbol,
            "outputsize": "full",
            "apikey": alpha_vantage_api_key
        }

        try:
            with st.spinner(f"Alpha Vantage for {ticker_symbol}... (Note: Free tier has rate limits.)"):
                time.sleep(15) # Wait 15 seconds to respect the rate limit (5 calls/min)

                response_av = requests.get(alpha_vantage_url, params=params_av, timeout=20)
                response_av.raise_for_status()
                data_av = response_av.json()

                if "Time Series (Daily)" in data_av:
                    raw_data = data_av["Time Series (Daily)"]
                    df_av = pd.DataFrame.from_dict(raw_data, orient="index")
                    df_av.index = pd.to_datetime(df_av.index)
                    df_av.sort_index(inplace=True)

                    column_mapping_av = {
                        '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
                        '5. adjusted close': 'Close', '6. volume': 'Volume'
                    }
                    df_av = df_av[[col for col in column_mapping_av.keys() if col in df_av.columns]]
                    df_av = df_av.rename(columns=column_mapping_av)
                    
                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if col in df_av.columns:
                            df_av[col] = pd.to_numeric(df_av[col], errors='coerce')
                    
                    df_av.reset_index(inplace=True)
                    df_av.rename(columns={'index': 'Date'}, inplace=True)
                    df_av['Date'] = df_av['Date'].dt.date

                    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required_cols:
                        if col not in df_av.columns:
                            df_av[col] = np.nan
                    hist_df = df_av[required_cols].dropna().reset_index(drop=True)

                    if not hist_df.empty:
                        st.success(f"✅ Successfully loaded historical data for {ticker_symbol} using Alpha Vantage.")
                        print(f"DEBUG: Alpha Vantage data loaded for {ticker_symbol} with {len(hist_df)} rows.")
                        return hist_df
                    else:
                        st.warning(f"⚠️ Alpha Vantage returned empty or malformed data for {ticker_symbol} after processing. No historical data available.")
                        print(f"DEBUG: Alpha Vantage empty/malformed data for {ticker_symbol} after processing.")

                elif "Error Message" in data_av:
                    st.error(f"❌ Alpha Vantage API Error for {ticker_symbol}: {data_av['Error Message']}. Please check your API key or usage limits.")
                    print(f"DEBUG: Alpha Vantage API Error: {data_av['Error Message']}")
                else:
                    st.error(f"❌ Alpha Vantage returned unexpected data format for {ticker_symbol}. Raw response keys: {list(data_av.keys()) if isinstance(data_av, dict) else 'Not a dict'}")
                    print(f"DEBUG: Alpha Vantage unexpected data format: {data_av}")

        except requests.exceptions.RequestException as req_err:
            st.error(f"❌ Alpha Vantage API request failed for {ticker_symbol}: {req_err}. Check internet/API status.")
            print(f"DEBUG: Alpha Vantage Request Error: {req_err}")
        except json.JSONDecodeError as json_err:
            st.error(f"❌ Alpha Vantage: Received invalid JSON data. Error: {json_err}")
            print(f"DEBUG: Alpha Vantage JSON Decode Error: {json_err}")
        except KeyError as ke:
            st.error(f"❌ Alpha Vantage: Data parsing error - expected column not found. Error: {ke}. API response format may have changed.")
            print(f"DEBUG: Alpha Vantage KeyError: {ke}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred while fetching from Alpha Vantage: {e}")
            print(f"DEBUG: Alpha Vantage Unexpected Error: {e}")

    # --- Attempt 3: Fallback to Financial Modeling Prep (FMP) historical-chart/daily ---
    st.info(f"Attempt 3/4: YFinance and Alpha Vantage failed for {ticker_symbol}. Falling back to Financial Modeling Prep (FMP) historical chart data (comprehensive)...")
    # FMP_API_KEY is now definitively set at the top of the file
    if not fmp_api_key or fmp_api_key == "YOUR_FMP_KEY":
        st.error("❌ FMP API key is not set. Cannot use FMP as a fallback for historical chart data.")
    else:
        fmp_historical_chart_url = f"https://financialmodelingprep.com/api/v3/historical-chart/daily/{ticker_symbol}"
        params_fmp_chart = {"apikey": fmp_api_key}
        
        try:
            with st.spinner(f"FMP historical chart data for {ticker_symbol}..."):
                response_fmp_chart = requests.get(fmp_historical_chart_url, params=params_fmp_chart, timeout=20)
                response_fmp_chart.raise_for_status()
                data_fmp_chart = response_fmp_chart.json()

                if data_fmp_chart and isinstance(data_fmp_chart, list) and data_fmp_chart:
                    df_fmp_chart = pd.DataFrame(data_fmp_chart)
                    df_fmp_chart['date'] = pd.to_datetime(df_fmp_chart['date'])
                    df_fmp_chart.sort_values('date', ascending=True, inplace=True)
                    
                    df_fmp_chart.rename(columns={
                        'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'
                    }, inplace=True)
                    
                    df_fmp_chart['Date'] = df_fmp_chart['Date'].dt.date
                    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    hist_df = df_fmp_chart[required_cols].dropna().reset_index(drop=True)

                    if not hist_df.empty:
                        st.success(f"✅ Successfully loaded historical data for {ticker_symbol} using FMP (historical chart).")
                        print(f"DEBUG: FMP (historical chart) data loaded for {ticker_symbol} with {len(hist_df)} rows.")
                        return hist_df
                    else:
                        st.warning(f"⚠️ FMP (historical chart) returned empty or malformed data for {ticker_symbol}.")
                        print(f"DEBUG: FMP (historical chart) empty/malformed data for {ticker_symbol}.")

                elif isinstance(data_fmp_chart, dict) and "Error Message" in data_fmp_chart:
                    st.error(f"❌ FMP API Error for {ticker_symbol} historical chart data: {data_fmp_chart['Error Message']}. Check your FMP API key or usage limits. This endpoint may have restrictions.")
                    print(f"DEBUG: FMP (historical chart) API Error: {data_fmp_chart['Error Message']}")
                else:
                    st.error(f"❌ FMP (historical chart) returned unexpected data format for {ticker_symbol}. Raw response: {data_fmp_chart}")
                    print(f"DEBUG: FMP (historical chart) unexpected data format: {data_fmp_chart}")

        except requests.exceptions.RequestException as req_err:
            st.error(f"❌ FMP API request failed for {ticker_symbol} (historical chart): {req_err}. Check internet/API status.")
            print(f"DEBUG: FMP (historical chart) Request Error: {req_err}")
        except json.JSONDecodeError as json_err:
            st.error(f"❌ FMP: Received invalid JSON data for historical chart data. Error: {json_err}")
            print(f"DEBUG: FMP (historical chart) JSON Decode Error: {json_err}")
        except KeyError as ke:
            st.error(f"❌ FMP: Data parsing error - expected column not found for historical chart data. Error: {ke}. API response format may have changed.")
            print(f"DEBUG: FMP (historical chart) KeyError: {ke}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred while fetching historical chart data from FMP: {e}")
            print(f"DEBUG: FMP (historical chart) Unexpected Error: {e}")

    # --- Attempt 4: Fallback to FMP historical-price (simpler endpoint, might be more permissive) ---
    st.info(f"Attempt 4/4: All previous historical data sources failed for {ticker_symbol}. Trying FMP's simpler historical price endpoint...")
    # FMP_API_KEY is now definitively set at the top of the file
    if not fmp_api_key or fmp_api_key == "YOUR_FMP_KEY":
        st.error("❌ FMP API key is not set. Cannot use FMP historical price endpoint as a fallback.")
    else:
        fmp_simple_historical_url = f"https://financialmodelingprep.com/api/v3/historical-price/{ticker_symbol}"
        params_fmp_simple = {"apikey": fmp_api_key}

        try:
            with st.spinner(f"FMP simple historical price for {ticker_symbol}..."):
                response_fmp_simple = requests.get(fmp_simple_historical_url, params=params_fmp_simple, timeout=20)
                response_fmp_simple.raise_for_status()
                data_fmp_simple = response_fmp_simple.json()

                if data_fmp_simple and "historical" in data_fmp_simple and data_fmp_simple["historical"]:
                    df_fmp_simple = pd.DataFrame(data_fmp_simple["historical"])
                    df_fmp_simple['date'] = pd.to_datetime(df_fmp_simple['date'])
                    df_fmp_simple.sort_values('date', ascending=True, inplace=True)
                    
                    df_fmp_simple.rename(columns={
                        'date': 'Date',
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    }, inplace=True)
                    
                    df_fmp_simple['Date'] = df_fmp_simple['Date'].dt.date
                    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    
                    # Fill missing OHLCV columns with Close price to ensure data structure for models
                    for col in ['Open', 'High', 'Low', 'Volume']:
                        if col not in df_fmp_simple.columns or df_fmp_simple[col].isnull().all():
                            df_fmp_simple[col] = df_fmp_simple['Close']
                            st.warning(f"⚠️ Filled missing '{col}' data with 'Close' price from FMP simple endpoint for {ticker_symbol}. Forecasting/models might be less accurate.")

                    hist_df = df_fmp_simple[required_cols].dropna().reset_index(drop=True)

                    if not hist_df.empty:
                        st.success(f"✅ Successfully loaded historical data for {ticker_symbol} using FMP (simple historical price).")
                        print(f"DEBUG: FMP (simple historical price) data loaded for {ticker_symbol} with {len(hist_df)} rows.")
                        return hist_df
                    else:
                        st.warning(f"⚠️ FMP (simple historical price) returned empty or malformed data for {ticker_symbol}.")
                        print(f"DEBUG: FMP (simple historical price) empty/malformed data for {ticker_symbol}.")

                elif isinstance(data_fmp_simple, dict) and "Error Message" in data_fmp_simple:
                    st.error(f"❌ FMP API Error for {ticker_symbol} (simple historical price): {data_fmp_simple['Error Message']}. Check your FMP API key or usage limits.")
                    print(f"DEBUG: FMP (simple historical price) API Error: {data_fmp_simple['Error Message']}")
                else:
                    st.error(f"❌ FMP (simple historical price) returned unexpected data format for {ticker_symbol}. Raw response: {data_fmp_simple}")
                    print(f"DEBUG: FMP (simple historical price) unexpected data format: {data_fmp_simple}")

        except requests.exceptions.RequestException as req_err:
            st.error(f"❌ FMP API request failed for {ticker_symbol} (simple historical price): {req_err}. Check internet/API status.")
            print(f"DEBUG: FMP (simple historical price) Request Error: {req_err}")
        except json.JSONDecodeError as json_err:
            st.error(f"❌ FMP: Received invalid JSON data for simple historical price. Error: {json_err}")
            print(f"DEBUG: FMP (simple historical price) JSON Decode Error: {json_err}")
        except KeyError as ke:
            st.error(f"❌ FMP: Data parsing error - expected column not found for simple historical price. Error: {ke}. API response format may have changed.")
            print(f"DEBUG: FMP (simple historical price) KeyError: {ke}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred while fetching simple historical price from FMP: {e}")
            print(f"DEBUG: FMP (simple historical price) Unexpected Error: {e}")

    # --- FINAL FAILURE MESSAGE (API-ONLY) ---
    st.error(f"""
    ❌ **HISTORICAL DATA UNAVAILABLE FOR {ticker_symbol} FROM ALL FREE ONLINE API SOURCES.**
    
    This is highly likely due to limitations of free API tiers, which often:
    - Have strict rate limits (e.g., 5 calls/minute, 250 calls/day).
    - Provide inconsistent or limited historical data, especially for non-U.S. exchanges (like NSE/BSE).
    - May mark certain data as premium, even if other endpoints work.
    - The specific ticker symbol might not be covered by free access.
    
    **To troubleshoot and verify application functionality:**
    1.  **Verify All API Keys in `app.py`:** Double-check `NEWS_API_KEY`, `FMP_API_KEY`, `GEMINI_API_KEY`, and `ALPHA_VANTAGE_API_KEY`. Ensure they are valid and correctly pasted (no extra spaces, correct characters).
    2.  **Check API Usage Dashboards:** Log in to your NewsAPI, FMP, Alpha Vantage, and Google Cloud Console dashboards to see if you've hit any daily or minute-level rate limits.
    3.  **MOST IMPORTANT: Try a U.S. Ticker:** Please try a well-known U.S. stock like **`AAPL`**, **`MSFT`**, **`GOOGL`**, or **`NVDA`**. Free APIs typically provide much more consistent and comprehensive historical data for these.
        * **If a U.S. ticker works, it *confirms* that your application's logic is perfectly sound and the issue is with data availability for your chosen Indian tickers from free sources.**
    4.  **Wait and Retry:** Sometimes, API issues are temporary.
    
    **Analysis cannot proceed without historical data.**
    """)
    return pd.DataFrame()


# --- Streamlit Application Main Layout ---
def main():
    st.markdown("""
        <div class="main-header">
            <h1>📈 Intelligent Stock Insights</h1>
            <p>Your Comprehensive AI-Powered Stock Analysis Dashboard</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3>Enter a Stock Ticker to Begin Analysis</h3>", unsafe_allow_html=True)

    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = ""
    if 'analyze_triggered' not in st.session_state:
        st.session_state.analyze_triggered = False

    ticker_input_key = "main_ticker_search_input"
    ticker_input_value = st.session_state.current_ticker

    ticker_input = st.text_input(
        "Search Stock Ticker (e.g., AAPL, RELIANCE.NS, TCS.BO)",
        value=ticker_input_value,
        key=ticker_input_key,
        help="Type a few letters to see suggestions. Press Enter to analyze. For Indian stocks, use .NS for NSE (e.g., RELIANCE.NS) and .BO for BSE (e.g., TCS.BO)."
    )

    suggestions = []
    if ticker_input:
        # FMP_API_KEY is now correctly set in the global scope of this file
        if FMP_API_KEY == "YOUR_FMP_KEY": # This check will still trigger if the key is the default placeholder
            st.warning("⚠️ FMP_API_KEY is not set. Autocomplete suggestions may be limited or unavailable. Please update `app.py`.")
        else:
            suggestions = fmp_autocomplete.fetch_fmp_suggestions(ticker_input, api_key=FMP_API_KEY)

    if suggestions:
        st.markdown("<h5>Suggestions:</h5>", unsafe_allow_html=True)
        num_columns_to_create = min(len(suggestions), 5)

        if num_columns_to_create > 0:
            cols = st.columns(num_columns_to_create)
            for i, suggestion in enumerate(suggestions):
                if i < len(cols):
                    with cols[i]:
                        suggested_ticker = suggestion.split(' - ')[0].strip().upper()
                        if st.button(suggestion, key=f"suggestion_{i}"):
                            st.session_state.current_ticker = suggested_ticker
                            st.session_state.analyze_triggered = True
                            st.rerun()

    if st.button("🚀 Analyze Stock", key="analyze_button", type="primary"):
        if ticker_input:
            st.session_state.current_ticker = ticker_input.split(' - ')[0].strip().upper()
            st.session_state.analyze_triggered = True
        else:
            st.warning("Please enter a stock ticker to analyze.")
            st.session_state.analyze_triggered = False
        st.rerun()

    if st.session_state.analyze_triggered and st.session_state.current_ticker:
        ticker_to_analyze = st.session_state.current_ticker

        st.markdown(f"<h2 class='analysis-header'>Comprehensive Analysis for {ticker_to_analyze}</h2>",
                    unsafe_allow_html=True)

        # Load historical data first, as it's a prerequisite for multiple tabs
        if 'historical_data' not in st.session_state or st.session_state.historical_data is None or \
           st.session_state.historical_data.empty or \
           (hasattr(st.session_state.historical_data, 'name') and st.session_state.historical_data.name != ticker_to_analyze):
            st.session_state.historical_data = load_historical_data(ticker_to_analyze, ALPHA_VANTAGE_API_KEY, FMP_API_KEY) 
            if not st.session_state.historical_data.empty:
                st.session_state.historical_data.name = ticker_to_analyze # Store ticker with data

        hist_data_for_tabs = st.session_state.historical_data

        if hist_data_for_tabs.empty:
            st.error(f"❌ Analysis cannot proceed for {ticker_to_analyze}: Historical data could not be retrieved. Please verify the ticker or try again later.")
            st.session_state.analyze_triggered = False # Reset trigger if data is missing
            return

        tab_summary, tab_financials, tab_probabilistic, tab_forecast, tab_news = st.tabs([
            "Company Overview", "Financials", "Probabilistic Models", "Forecasting", "News Sentiment"
        ])

        # --- Pass relevant data and API keys to each module ---
        with tab_summary:
            if FMP_API_KEY == "YOUR_FMP_KEY": 
                st.warning("⚠️ FMP_API_KEY is not set. Company overview might be incomplete (relying solely on yfinance) and financial data/news company name lookup will be unavailable.")
            if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
                st.warning("⚠️ GEMINI_API_KEY is not set. AI-powered company insights will be unavailable. Please update `app.py`.")
            stock_summary.display_stock_summary(ticker_to_analyze, fmp_api_key=FMP_API_KEY, gemini_api_key=GEMINI_API_KEY)

        with tab_financials:
            if FMP_API_KEY == "YOUR_FMP_KEY":
                st.error("❌ FMP_API_KEY is not set. Financial statements cannot be loaded. Please set your FMP_API_KEY in app.py.")
            else:
                financials.display_financials(ticker_to_analyze, fmp_api_key=FMP_API_KEY)

        with tab_probabilistic:
            probabilistic_stock_model.display_probabilistic_models(hist_data_for_tabs)

        with tab_news:
            if NEWS_API_KEY == "YOUR_NEWSAPI_KEY" or FMP_API_KEY == "YOUR_FMP_KEY":
                st.error("❌ NewsAPI_KEY or FMP_API_KEY is not set. News sentiment analysis will not work. Please set your API keys in app.py.")
            else:
                news_sentiment.display_news_sentiment(ticker_to_analyze, news_api_key=NEWS_API_KEY, fmp_api_key=FMP_API_KEY)

        with tab_forecast:
            forecast_module.display_forecasting(hist_data_for_tabs)


if __name__ == "__main__":
    main()
