import streamlit as st
import pandas as pd
import numpy as np
import warnings
import time
import yfinance as yf # Import yfinance directly in app.py for load_historical_data

# Import functions from your separate modules
import pages.yahoo_autocomplete as yahoo_autocomplete
import pages.stock_summary as stock_summary
import pages.financials as financials
import pages.probabilistic_stock_model as probabilistic_stock_model
import pages.forecast_module as forecast_module
import pages.news_sentiment as news_sentiment

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# --- GLOBAL CONFIGURATIONS AND INITIAL STREAMLIT SETUP ---
# st.set_page_config() MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Intelligent Stock Insights",
    layout="wide",
    # Hide the Streamlit main menu and sidebar toggle
    initial_sidebar_state="collapsed",  # Ensures sidebar is collapsed
    menu_items={
        'Get help': None,  # Disables 'Get help' in the menu
        'Report a bug': None,  # Disables 'Report a bug' in the menu
        'About': None  # Disables 'About' in the menu
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

# NewsAPI.com API Key - IMPORTANT: Replace with your actual key
NEWS_API_KEY = "874ba654bdcd4aa7b68f7367a907cc2f"  # Replace with your NewsAPI key

# --- Custom CSS and Font Loading ---
def load_css(file_path):
    """Loads custom CSS from a file."""
    try:
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Error: CSS file not found at {file_path}. Please ensure 'assets/style.css' exists.")

# Call CSS loading AFTER set_page_config
st.markdown(
    "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap' rel='stylesheet'>",
    unsafe_allow_html=True)
load_css("assets/style.css")


# --- Common Data Loading & Feature Engineering Functions ---
@st.cache_data(ttl=3600, show_spinner=False)  # Cache historical data for 1 hour
def load_historical_data(stock_name, period="5y", retries=3, initial_delay=0.5):
    """
    Loads historical stock data from Yahoo Finance with retry logic.
    Handles potential API blocking or invalid responses gracefully.
    """
    if not stock_name:
        return pd.DataFrame()

    # Ensure no proxy is set if it was causing issues (can be left out if not relevant)
    yf.set_proxy(None) 

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                print(f"Retrying historical data fetch for {stock_name} (attempt {attempt}/{retries}). Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)

            stock = yf.Ticker(stock_name)
            # Add a timeout to the history call for robustness
            hist = stock.history(period=period, timeout=15) 

            if hist.empty:
                # If history is empty but no error, it might mean no data for ticker/period
                if attempt == retries: # Only show error on last attempt if still empty
                    st.error(f"❌ No historical data found for {stock_name}. This might be due to issues with the Yahoo Finance API, an invalid ticker, or the stock being delisted. Please try another symbol or check again later.")
                continue # Continue to retry if not last attempt
            
            hist.reset_index(inplace=True)
            hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None) # Ensure no timezone for consistency
            return hist # Data loaded successfully, return it

        except Exception as e:
            # Catch all exceptions during yfinance call, including JSONDecodeError from requests indirectly
            print(f"Attempt {attempt}/{retries}: Failed to get historical data for {stock_name}. Error: {e}")
            if "Expecting value: line 1 column 1" in str(e) or "429 Client Error: Too Many Requests" in str(e):
                print(f"Likely Yahoo Finance API blocking/rate limiting for {stock_name}. Retrying...")
            
            if attempt == retries:
                st.error(f"⚠️ Failed to load historical data for {stock_name} after multiple attempts. This often happens with Yahoo Finance's unofficial API due to rate limits or bot detection. Please try again later or with a different symbol.")
                return pd.DataFrame() # Return empty DataFrame on final failure
    return pd.DataFrame() # Fallback return if loop finishes without returning


@st.cache_data(show_spinner=False)
def add_common_features(hist):
    """Adds various technical indicators as features common to multiple modules."""
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


@st.cache_data(show_spinner=False)
def compute_rsi(series, period=14):
    """Computes the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --- Streamlit Application Main Layout ---
def main():
    st.markdown("""
        <div class="main-header">
            <h1>📈 Intelligent Stock Insights</h1>
            <p>Your Comprehensive AI-Powered Stock Analysis Dashboard</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Centralized Stock Ticker Input and Autocomplete ---
    st.markdown("<h3>Enter a Stock Ticker to Begin Analysis</h3>", unsafe_allow_html=True)

    # Initialize session state for ticker if not already present
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = ""
    if 'analyze_triggered' not in st.session_state:
        st.session_state.analyze_triggered = False

    # Text input for ticker search
    ticker_input = st.text_input(
        "Search Stock Ticker (e.g., AAPL, RELIANCE.NS)",
        value=st.session_state.current_ticker,
        key="main_ticker_search",
        help="Type a few letters to see suggestions. Press Enter to analyze."
    )

    # Autocomplete suggestions
    suggestions = []
    if ticker_input:
        # Use the robust fetch_yahoo_suggestions from yahoo_autocomplete module
        suggestions = yahoo_autocomplete.fetch_yahoo_suggestions(ticker_input)

    # --- Only display columns and buttons if suggestions exist ---
    if suggestions:
        st.markdown("<h5>Suggestions:</h5>", unsafe_allow_html=True)
        num_columns_to_create = min(len(suggestions), 5)
        
        if num_columns_to_create > 0: # Ensure we create at least one column if suggestions are present
            cols = st.columns(num_columns_to_create)
            for i, suggestion in enumerate(suggestions):
                if i < len(cols): # Defensive check to prevent index out of range
                    with cols[i]:
                        if st.button(suggestion, key=f"suggestion_{i}"):
                            st.session_state.current_ticker = suggestion.split(' - ')[0]  # Extract just the ticker symbol
                            st.session_state.analyze_triggered = True
                            st.rerun()  # Rerun to update the main input and trigger analysis
    # --- END FIX ---

    # Analyze Button
    if st.button("🚀 Analyze Stock", key="analyze_button", type="primary"):
        if ticker_input:
            st.session_state.current_ticker = ticker_input.split(' - ')[0]  # Ensure just ticker symbol is saved
            st.session_state.analyze_triggered = True
        else:
            st.warning("Please enter a stock ticker to analyze.")
            st.session_state.analyze_triggered = False
        st.rerun()  # Rerun the app to process the analysis

    # --- Display Analysis Results in Tabs ---
    if st.session_state.analyze_triggered and st.session_state.current_ticker:
        ticker_to_analyze = st.session_state.current_ticker

        st.markdown(f"<h2 class='analysis-header'>Comprehensive Analysis for {ticker_to_analyze.upper()}</h2>",
                    unsafe_allow_html=True)

        # Tabs for different analysis sections
        tab_summary, tab_financials, tab_probabilistic, tab_forecast, tab_news = st.tabs([
            "Company Overview", "Financials", "Probabilistic Models", "Forecasting", "News Sentiment"
        ])

        # Use st.session_state to cache historical data
        # Only load data if ticker changes or not loaded yet
        if 'historical_data' not in st.session_state or st.session_state.historical_data is None or st.session_state.historical_data.empty or (hasattr(st.session_state.historical_data, 'name') and st.session_state.historical_data.name != ticker_to_analyze):
            with st.spinner(f"Loading historical data for {ticker_to_analyze.upper()}..."):
                # Call the load_historical_data function with retry logic
                st.session_state.historical_data = load_historical_data(ticker_to_analyze)
                if not st.session_state.historical_data.empty: # Only store name if data was actually loaded
                    st.session_state.historical_data.name = ticker_to_analyze  # Store ticker name with data for caching logic

        hist_data_for_tabs = st.session_state.historical_data

        if hist_data_for_tabs.empty: # Check for empty DataFrame instead of None
            # Error message is already shown by load_historical_data
            st.session_state.analyze_triggered = False  # Reset trigger
            # st.stop() # Uncomment if you want to stop further execution when no data
            return  # Exit function if no data

        with tab_summary:
            stock_summary.display_stock_summary(ticker_to_analyze)  # Call from stock_summary module

        with tab_financials:
            financials.display_financials(ticker_to_analyze)  # Call from financials module

        with tab_probabilistic:
            # Pass the loaded historical data to the probabilistic module
            probabilistic_stock_model.display_probabilistic_models(hist_data_for_tabs)

        with tab_forecast:
            # Pass the loaded historical data to the forecast module
            forecast_module.display_forecasting(hist_data_for_tabs)

        with tab_news:
            # Pass the NEWS_API_KEY to the news_sentiment module's display function
            news_sentiment.display_news_sentiment(ticker_to_analyze, news_api_key=NEWS_API_KEY)


if __name__ == "__main__":
    main()
