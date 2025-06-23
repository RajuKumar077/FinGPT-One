import streamlit as st
import pandas as pd
import numpy as np
import warnings
import time
import requests
import json
import yfinance as yf # Primary for historical data
import os # Import os for path checking

# Import functions from your separate modules
# Assuming these modules exist in a 'pages' directory
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

# API Keys (IMPORTANT: REPLACE "YOUR_KEY_HERE" WITH YOUR ACTUAL KEYS)
NEWS_API_KEY = "874ba654bdcd4aa7b68f7367a907cc2f" # Get your free key from newsapi.com
FMP_API_KEY = "5C9DnMCAzYam2ZPjNpOxKLFxUiGhrJDD"     # Your provided FMP key
GEMINI_API_KEY = "AIzaSyAK8BevJ1wIrwMoYDsnCLQXdZlFglF92WE" # Your provided Gemini key
ALPHA_VANTAGE_API_KEY = "WLVUE35CQ906QK3K" # IMPORTANT: Get your free key from www.alphavantage.co

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


# --- Historical Data Loading (yfinance, Alpha Vantage, FMP, and CSV Fallback) ---
@st.cache_data(ttl=3600, show_spinner=False)  # Cache historical data for 1 hour
def load_historical_data(ticker_symbol, alpha_vantage_api_key, fmp_api_key):
    """
    Loads historical stock data, attempting yfinance, then Alpha Vantage, then FMP (historical-chart/daily, then historical-price-full).
    As a guaranteed last resort, it will try to load from a local CSV file.
    """
    if not ticker_symbol:
        return pd.DataFrame()

    hist_df = pd.DataFrame()

    # --- Attempt 1: Try yfinance with multiple periods ---
    st.info(f"Trying to load historical data for {ticker_symbol} using yfinance (primary source)...")
    periods_to_try_yf = ["max", "5y", "2y", "1y", "6mo", "3mo", "1mo"] # Ordered from longest to shortest

    for period in periods_to_try_yf:
        try:
            with st.spinner(f"Attempting yfinance for {ticker_symbol} (period: {period})..."):
                ticker = yf.Ticker(ticker_symbol)
                hist_df_yf = ticker.history(period=period, auto_adjust=True, timeout=15)

            if not hist_df_yf.empty:
                hist_df_yf.reset_index(inplace=True)
                hist_df_yf.rename(columns={
                    'Date': 'Date',
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume'
                }, inplace=True)
                hist_df_yf['Date'] = pd.to_datetime(hist_df_yf['Date']).dt.date
                required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                # Ensure all required columns are present, fill with NaN if not, then drop rows with NaNs in these cols
                for col in required_cols:
                    if col not in hist_df_yf.columns:
                        hist_df_yf[col] = np.nan
                hist_df = hist_df_yf[required_cols].dropna().reset_index(drop=True)
                
                if not hist_df.empty:
                    st.success(f"✅ Successfully loaded historical data for {ticker_symbol} using yfinance (period: '{period}').")
                    print(f"DEBUG: YFinance data loaded for {ticker_symbol} with {len(hist_df)} rows.")
                    return hist_df # Return on first successful yfinance load
                else:
                    print(f"DEBUG: yfinance returned empty data for {ticker_symbol} with period '{period}' after processing. Trying next yfinance period.")
                    continue # Continue to next period if data is empty after processing

        except requests.exceptions.RequestException as req_err:
            print(f"DEBUG: yfinance network error for {ticker_symbol} ({period}): {req_err}")
            st.warning(f"⚠️ YFinance network error for {ticker_symbol} (period: {period}). Trying next yfinance period.")
            continue # Try next period on network errors
        except Exception as e:
            print(f"DEBUG: Generic yfinance error for {ticker_symbol} ({period}): {e}")
            st.warning(f"⚠️ YFinance data issue for {ticker_symbol} (period: {period}): {e}. This often indicates an issue with the data source's response, an invalid ticker, or temporary data unavailability. Trying next period.")
            continue # Try next period on other exceptions

    # --- Attempt 2: Fallback to Alpha Vantage if yfinance completely failed ---
    st.info(f"YFinance failed for {ticker_symbol}. Falling back to Alpha Vantage...")
    if not alpha_vantage_api_key or alpha_vantage_api_key == "YOUR_ALPHA_VANTAGE_API_KEY":
        st.error("❌ Alpha Vantage API key is not set. Cannot use Alpha Vantage as a fallback.")
    else:
        alpha_vantage_url = "https://www.alphavantage.co/query"
        params_av = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker_symbol,
            "outputsize": "full", # or "compact" for last 100 days
            "apikey": alpha_vantage_api_key
        }

        try:
            with st.spinner(f"Attempting to load historical data for {ticker_symbol} from Alpha Vantage... (This may take a moment due to API limits)"):
                time.sleep(15) # Wait 15 seconds to respect the rate limit (5 calls/min means 12s per call average)

                response_av = requests.get(alpha_vantage_url, params=params_av, timeout=20)
                response_av.raise_for_status()
                data_av = response_av.json()

                if not data_av: # Explicitly check for empty dictionary
                    st.warning(f"⚠️ Alpha Vantage returned an empty response for {ticker_symbol}. This might indicate no data or rate limit.")
                    print(f"DEBUG: Alpha Vantage returned empty JSON for {ticker_symbol}.")
                elif "Time Series (Daily)" in data_av:
                    raw_data = data_av["Time Series (Daily)"]
                    df_av = pd.DataFrame.from_dict(raw_data, orient="index")
                    df_av.index = pd.to_datetime(df_av.index)
                    df_av.sort_index(inplace=True)

                    column_mapping_av = {
                        '1. open': 'Open',
                        '2. high': 'High',
                        '3. low': 'Low',
                        '5. adjusted close': 'Close', # This is what we primarily need for 'Close'
                        '6. volume': 'Volume'
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

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                st.error(f"❌ Alpha Vantage API rate limit hit for {ticker_symbol}. Please wait at least 1 minute before trying another stock.")
            elif http_err.response.status_code in [401, 403]:
                st.error("❌ Alpha Vantage API key is invalid or unauthorized. Please check your ALPHA_VANTAGE_API_KEY in app.py.")
            else:
                st.error(f"❌ Alpha Vantage HTTP error occurred: {http_err}. Status: {http_err.response.status_code}")
            print(f"DEBUG: Alpha Vantage HTTP Error: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            st.error(f"❌ Alpha Vantage Connection error: {conn_err}. Please check your internet connection.")
            print(f"DEBUG: Alpha Vantage Connection Error: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            st.error(f"❌ Alpha Vantage Request timed out. The server might be slow or unresponsive. Please try again.")
            print(f"DEBUG: Alpha Vantage Timeout: {timeout_err}")
        except json.JSONDecodeError as json_err:
            st.error(f"❌ Alpha Vantage: Received invalid JSON data from API. Please try again later. Error: {json_err}")
            print(f"DEBUG: Alpha Vantage JSON Decode Error: {json_err}")
        except KeyError as ke:
            st.error(f"❌ Alpha Vantage: Data parsing error - expected column not found. Error: {ke}. This may indicate a change in API response format.")
            print(f"DEBUG: Alpha Vantage KeyError: {ke}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred while fetching from Alpha Vantage: {e}")
            print(f"DEBUG: Alpha Vantage Unexpected Error: {e}")

    # --- Attempt 3: Fallback to Financial Modeling Prep (FMP) historical-chart/daily ---
    st.info(f"YFinance and Alpha Vantage failed for {ticker_symbol}. Falling back to Financial Modeling Prep (FMP) historical chart data...")
    if not fmp_api_key or fmp_api_key == "YOUR_FMP_KEY":
        st.error("❌ FMP API key is not set. Cannot use FMP as a fallback for historical chart data.")
    else:
        fmp_historical_chart_url = f"https://financialmodelingprep.com/api/v3/historical-chart/daily/{ticker_symbol}"
        params_fmp_chart = {"apikey": fmp_api_key}
        
        try:
            with st.spinner(f"Attempting to load historical chart data for {ticker_symbol} from FMP (historical-chart/daily)..."):
                response_fmp_chart = requests.get(fmp_historical_chart_url, params=params_fmp_chart, timeout=20)
                response_fmp_chart.raise_for_status()
                data_fmp_chart = response_fmp_chart.json()

                if data_fmp_chart and isinstance(data_fmp_chart, list) and data_fmp_chart:
                    df_fmp_chart = pd.DataFrame(data_fmp_chart)
                    df_fmp_chart['date'] = pd.to_datetime(df_fmp_chart['date'])
                    df_fmp_chart.sort_values('date', ascending=True, inplace=True)
                    
                    df_fmp_chart.rename(columns={
                        'date': 'Date',
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    }, inplace=True)
                    
                    df_fmp_chart['Date'] = df_fmp_chart['Date'].dt.date
                    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required_cols:
                        if col not in df_fmp_chart.columns:
                            df_fmp_chart[col] = np.nan
                    hist_df = df_fmp_chart[required_cols].dropna().reset_index(drop=True)

                    if not hist_df.empty:
                        st.success(f"✅ Successfully loaded historical data for {ticker_symbol} using FMP (historical-chart/daily).")
                        print(f"DEBUG: FMP (historical-chart/daily) data loaded for {ticker_symbol} with {len(hist_df)} rows.")
                        return hist_df
                    else:
                        st.warning(f"⚠️ FMP (historical-chart/daily) returned empty or malformed data for {ticker_symbol}.")
                        print(f"DEBUG: FMP (historical-chart/daily) empty/malformed data for {ticker_symbol}.")

                elif isinstance(data_fmp_chart, dict) and "Error Message" in data_fmp_chart:
                    st.error(f"❌ FMP API Error for {ticker_symbol} historical chart data: {data_fmp_chart['Error Message']}. The 'historical-chart' endpoint may have restrictions on your plan.")
                    print(f"DEBUG: FMP (historical-chart/daily) API Error: {data_fmp_chart['Error Message']}")
                else:
                    st.error(f"❌ FMP (historical-chart/daily) returned unexpected data format for {ticker_symbol}. Raw response: {data_fmp_chart}")
                    print(f"DEBUG: FMP (historical-chart/daily) unexpected data format: {data_fmp_chart}")

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                st.error(f"❌ FMP API rate limit hit for historical chart data for {ticker_symbol}. Please wait and try again.")
            elif http_err.response.status_code in [401, 403]:
                st.error("❌ FMP API key is invalid or unauthorized for historical chart data. Please check your FMP_API_KEY in app.py.")
            else:
                st.error(f"❌ FMP HTTP error occurred fetching historical chart data: {http_err}. Status: {http_err.response.status_code}")
            print(f"DEBUG: FMP (historical-chart/daily) HTTP Error: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            st.error(f"❌ FMP Connection error fetching historical chart data: {conn_err}. Check your internet connection.")
            print(f"DEBUG: FMP (historical-chart/daily) Connection Error: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            st.error(f"❌ FMP Request timed out fetching historical chart data. Server might be slow or unresponsive. Please try again.")
            print(f"DEBUG: FMP (historical-chart/daily) Timeout: {timeout_err}")
        except json.JSONDecodeError as json_err:
            st.error(f"❌ FMP: Received invalid JSON data for historical chart data. Error: {json_err}")
            print(f"DEBUG: FMP (historical-chart/daily) JSON Decode Error: {json_err}")
        except KeyError as ke:
            st.error(f"❌ FMP: Data parsing error - expected column not found for historical chart data. Error: {ke}. This may indicate a change in API response format.")
            print(f"DEBUG: FMP (historical-chart/daily) KeyError: {ke}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred while fetching historical chart data from FMP: {e}")
            print(f"DEBUG: FMP (historical-chart/daily) Unexpected Error: {e}")

    # --- Attempt 4: FMP Historical Price Full (if historical-chart/daily failed) ---
    if hist_df.empty: # Only try this if the previous FMP attempt didn't succeed
        st.info(f"FMP (historical-chart/daily) failed for {ticker_symbol}. Trying FMP (historical-price-full)...")
        if not fmp_api_key or fmp_api_key == "YOUR_FMP_KEY":
            st.error("❌ FMP API key is not set. Cannot use FMP as a fallback for historical-price-full data.")
        else:
            fmp_historical_price_full_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker_symbol}"
            params_fmp_full = {"apikey": fmp_api_key}

            try:
                with st.spinner(f"Attempting to load historical price data for {ticker_symbol} from FMP (historical-price-full)..."):
                    response_fmp_full = requests.get(fmp_historical_price_full_url, params=params_fmp_full, timeout=20)
                    response_fmp_full.raise_for_status()
                    data_fmp_full = response_fmp_full.json()
                    
                    if data_fmp_full and isinstance(data_fmp_full, dict) and "historical" in data_fmp_full and data_fmp_full["historical"]:
                        df_fmp_full = pd.DataFrame(data_fmp_full["historical"])
                        df_fmp_full['date'] = pd.to_datetime(df_fmp_full['date'])
                        df_fmp_full.sort_values('date', ascending=True, inplace=True)
                        
                        df_fmp_full.rename(columns={
                            'date': 'Date',
                            'open': 'Open',
                            'high': 'High',
                            'low': 'Low',
                            'close': 'Close',
                            'volume': 'Volume'
                        }, inplace=True)
                        
                        df_fmp_full['Date'] = df_fmp_full['Date'].dt.date
                        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                        for col in required_cols:
                            if col not in df_fmp_full.columns:
                                df_fmp_full[col] = np.nan
                        hist_df = df_fmp_full[required_cols].dropna().reset_index(drop=True)

                        if not hist_df.empty:
                            st.success(f"✅ Successfully loaded historical data for {ticker_symbol} using FMP (historical-price-full).")
                            print(f"DEBUG: FMP (historical-price-full) data loaded for {ticker_symbol} with {len(hist_df)} rows.")
                            return hist_df
                        else:
                            st.warning(f"⚠️ FMP (historical-price-full) returned empty or malformed data for {ticker_symbol}.")
                            print(f"DEBUG: FMP (historical-price-full) empty/malformed data for {ticker_symbol}.")

                    elif isinstance(data_fmp_full, dict) and "Error Message" in data_fmp_full:
                        st.error(f"❌ FMP API Error for {ticker_symbol} historical-price-full data: {data_fmp_full['Error Message']}. Check your FMP API key or usage limits.")
                        print(f"DEBUG: FMP (historical-price-full) API Error: {data_fmp_full['Error Message']}")
                    else:
                        st.error(f"❌ FMP (historical-price-full) returned unexpected data format for {ticker_symbol}. Raw response: {data_fmp_full}")
                        print(f"DEBUG: FMP (historical-price-full) unexpected data format: {data_fmp_full}")

            except requests.exceptions.HTTPError as http_err:
                if http_err.response.status_code == 429:
                    st.error(f"❌ FMP API rate limit hit for historical-price-full data for {ticker_symbol}. Please wait and try again.")
                elif http_err.response.status_code in [401, 403]:
                    st.error("❌ FMP API key is invalid or unauthorized for historical-price-full data. Please check your FMP_API_KEY in app.py.")
                else:
                    st.error(f"❌ FMP HTTP error occurred fetching historical-price-full data: {http_err}. Status: {http_err.response.status_code}")
                print(f"DEBUG: FMP (historical-price-full) HTTP Error: {http_err}")
            except requests.exceptions.ConnectionError as conn_err:
                st.error(f"❌ FMP Connection error fetching historical-price-full data: {conn_err}. Check your internet connection.")
                print(f"DEBUG: FMP (historical-price-full) Connection Error: {conn_err}")
            except requests.exceptions.Timeout as timeout_err:
                st.error(f"❌ FMP Request timed out fetching historical-price-full data. Server might be slow or unresponsive. Please try again.")
                print(f"DEBUG: FMP (historical-price-full) Timeout: {timeout_err}")
            except json.JSONDecodeError as json_err:
                st.error(f"❌ FMP: Received invalid JSON data for historical-price-full data. Error: {json_err}")
                print(f"DEBUG: FMP (historical-price-full) JSON Decode Error: {json_err}")
            except KeyError as ke:
                st.error(f"❌ FMP: Data parsing error - expected column not found for historical-price-full data. Error: {ke}. This may indicate a change in API response format.")
                print(f"DEBUG: FMP (historical-price-full) KeyError: {ke}")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred while fetching historical-price-full data from FMP: {e}")
                print(f"DEBUG: FMP (historical-price-full) Unexpected Error: {e}")

    # --- LAST RESORT: Load from a local CSV file if all API calls failed ---
    # This block is ENABLED as a failsafe for demonstration purposes.
    # It ensures the app runs even if live API data is consistently unavailable.
    # ACTION REQUIRED:
    # 1. Create a folder named 'data' in your project's root directory.
    # 2. Download historical data CSVs for the tickers you want to analyze (e.g., from Yahoo Finance).
    #    Ensure the CSV has columns like 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'.
    # 3. Save the CSV file(s) in the 'data/' folder with the ticker symbol as the filename.
    #    Example: For RELIANCE.NS, save it as 'data/RELIANCE.NS.csv'.
    try:
        csv_path = os.path.join("data", f"{ticker_symbol.upper()}.csv")
        if os.path.exists(csv_path):
            st.info(f"All APIs failed for {ticker_symbol}. Attempting to load from local CSV: {csv_path}...")
            hist_df_csv = pd.read_csv(csv_path)
            hist_df_csv['Date'] = pd.to_datetime(hist_df_csv['Date']).dt.date
            hist_df_csv.sort_values(by='Date', ascending=True, inplace=True)
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            # Ensure required columns exist, fill missing with NaN, then drop rows with NaNs in required_cols
            for col in required_cols:
                if col not in hist_df_csv.columns:
                    hist_df_csv[col] = np.nan
            hist_df = hist_df_csv[required_cols].dropna().reset_index(drop=True)

            if not hist_df.empty:
                st.success(f"✅ Successfully loaded historical data for {ticker_symbol} from local CSV (all APIs failed).")
                st.warning(f"⚠️ Data for {ticker_symbol} is loaded from a local CSV. It may not be live or the most recent. Please ensure you have placed the CSV in the 'data/' folder correctly.")
                print(f"DEBUG: Loaded from CSV for {ticker_symbol} with {len(hist_df)} rows.")
                return hist_df
            else:
                st.warning(f"⚠️ Local CSV for {ticker_symbol} was empty or malformed after processing.")
                print(f"DEBUG: Local CSV empty/malformed for {ticker_symbol}.")
        else:
            print(f"DEBUG: Local CSV file not found at {csv_path}")
    except FileNotFoundError:
        print(f"DEBUG: Local CSV file not found (handled by os.path.exists check).")
    except Exception as e:
        st.error(f"❌ Error loading local CSV for {ticker_symbol}: {e}. Please check the CSV file format and contents.")
        print(f"DEBUG: Error loading CSV: {e}")
    # --- END LAST RESORT CSV BLOCK ---


    # If all sources (APIs and CSV) fail
    st.error(f"❌ Historical data for {ticker_symbol} could not be retrieved from any live API or local CSV. Please double-check the ticker symbol, your API keys, and ensure any necessary CSV files are correctly placed in the 'data/' folder. Data for this symbol may be consistently unavailable from free sources.")
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
        if FMP_API_KEY == "YOUR_FMP_KEY": # This check will still pass if the key is default
            st.warning("⚠️ FMP_API_KEY appears to be the default 'YOUR_FMP_KEY'. Autocomplete suggestions may be limited or unavailable. Please update `app.py` with your actual FMP key.")
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
        # This block now uses the enhanced load_historical_data with FMP historical-chart fallback AND CSV fallback
        if 'historical_data' not in st.session_state or st.session_state.historical_data is None or \
           st.session_state.historical_data.empty or \
           (hasattr(st.session_state.historical_data, 'name') and st.session_state.historical_data.name != ticker_to_analyze):
            st.session_state.historical_data = load_historical_data(ticker_to_analyze, ALPHA_VANTAGE_API_KEY, FMP_API_KEY) # Pass FMP key
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
                st.warning("⚠️ FMP_API_KEY appears to be the default 'YOUR_FMP_KEY'. Company overview might be incomplete (relying solely on yfinance) and financial data/news company name lookup will be unavailable.")
            if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
                st.warning("⚠️ GEMINI_API_KEY appears to be the default 'YOUR_GEMINI_API_KEY'. AI-powered company insights will be unavailable.")
            stock_summary.display_stock_summary(ticker_to_analyze, fmp_api_key=FMP_API_KEY, gemini_api_key=GEMINI_API_KEY)

        with tab_financials:
            if FMP_API_KEY == "YOUR_FMP_KEY":
                st.error("❌ FMP_API_KEY appears to be the default 'YOUR_FMP_KEY'. Financial statements cannot be loaded. Please set your FMP_API_KEY in app.py.")
            else:
                financials.display_financials(ticker_to_analyze, fmp_api_key=FMP_API_KEY)

        with tab_probabilistic:
            probabilistic_stock_model.display_probabilistic_models(hist_data_for_tabs)

        with tab_news:
            if NEWS_API_KEY == "YOUR_NEWSAPI_KEY" or FMP_API_KEY == "YOUR_FMP_KEY":
                st.error("❌ NewsAPI_KEY or FMP_API_KEY appears to be the default. News sentiment analysis will not work. Please set your API keys in app.py.")
            else:
                news_sentiment.display_news_sentiment(ticker_to_analyze, news_api_key=NEWS_API_KEY, fmp_api_key=FMP_API_KEY)

        with tab_forecast:
            forecast_module.display_forecasting(hist_data_for_tabs)


if __name__ == "__main__":
    main()
```
