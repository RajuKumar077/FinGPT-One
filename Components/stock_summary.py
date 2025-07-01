import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import time
from textblob import TextBlob
from dotenv import load_dotenv
import os

# Load API keys from .env file
load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY", "5C9DnMCAzYam2ZPjNpOxKLFxUiGhrJDD")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "8UU32LX81NSED6CM")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAK8BevJ1wIrwMoYDsnCLQXdZlFglF92WE")

# Debug API key loading
if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_KEY":
    st.error("❌ FMP API key is missing or invalid. Please set it in your .env file.")
if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
    st.warning("⚠️ Alpha Vantage API key is missing or invalid. Some features may not work.")
if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    st.warning("⚠️ Gemini API key is missing or invalid. AI insights will be mocked.")

@st.cache_data(ttl=3600, show_spinner="Fetching financial data...")
def fetch_financial_data(ticker_symbol, statement_type, fmp_api_key, period="annual", retries=3, initial_delay=0.5):
    """
    Fetches financial statement data (Income Statement, Balance Sheet, or Cash Flow) from FMP API.

    Args:
        ticker_symbol (str): Stock ticker.
        statement_type (str): 'income-statement', 'balance-sheet-statement', or 'cash-flow-statement'.
        fmp_api_key (str): FMP API key.
        period (str): 'annual' or 'quarterly'.
        retries (int): Number of retry attempts.
        initial_delay (float): Initial retry delay in seconds.

    Returns:
        pd.DataFrame: Financial statement data or empty DataFrame if fetching fails.
    """
    if not fmp_api_key or fmp_api_key == "YOUR_FMP_KEY":
        st.error(f"❌ FMP API key is missing for {statement_type}. Please set it in your .env file.")
        return pd.DataFrame()

    url = f"https://financialmodelingprep.com/api/v3/{statement_type}/{ticker_symbol}?period={period}&apikey={fmp_api_key}"

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(ascending=False, inplace=True)
                return df
            else:
                st.warning(f"⚠️ No {period} {statement_type} data returned for {ticker_symbol}.")
                return pd.DataFrame()
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 429:
                st.error(f"⚠️ FMP API rate limit reached for {ticker_symbol}. Wait 24 hours or upgrade your plan.")
            elif response.status_code in [401, 403]:
                st.error(f"❌ FMP API key unauthorized for {ticker_symbol}. Verify your API key.")
            else:
                st.error(f"⚠️ FMP HTTP error for {ticker_symbol}: {http_err} (Status: {response.status_code})")
            return pd.DataFrame()
        except requests.exceptions.ConnectionError as conn_err:
            st.error(f"❌ Network error for {ticker_symbol}: {conn_err}. Check your internet connection.")
            return pd.DataFrame()
        except requests.exceptions.Timeout as timeout_err:
            st.error(f"⏳ FMP API request timed out for {ticker_symbol}: {timeout_err}.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"⚠️ Unexpected error for {ticker_symbol}: {e}")
            return pd.DataFrame()
        time.sleep(initial_delay * (2 ** attempt))  # Exponential backoff
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="Fetching company profile...")
def fetch_stock_data(ticker_symbol, alpha_vantage_api_key, fmp_api_key, retries=3, initial_delay=0.5):
    """
    Fetches company profile using FMP API.
    (Unchanged from your provided code, included for completeness)
    """
    if not fmp_api_key or fmp_api_key == "YOUR_FMP_KEY":
        st.error("❌ FMP API key is missing for company profile. Please set it in your .env file.")
        return None

    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker_symbol}"
    params = {"apikey": fmp_api_key}

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0 and data[0]:
                return data[0]
            else:
                st.warning(f"⚠️ No company profile data returned for {ticker_symbol}.")
                return None
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 429:
                st.error(f"⚠️ FMP API rate limit reached for {ticker_symbol}. Wait 24 hours or upgrade your plan.")
            elif response.status_code in [401, 403]:
                st.error(f"❌ FMP API key unauthorized for {ticker_symbol}. Verify your API key.")
            else:
                st.error(f"⚠️ FMP HTTP error for {ticker_symbol}: {http_err} (Status: {response.status_code})")
            return None
        except requests.exceptions.ConnectionError as conn_err:
            st.error(f"❌ Network error for {ticker_symbol}: {conn_err}. Check your internet connection.")
            return None
        except requests.exceptions.Timeout as timeout_err:
            st.error(f"⏳ FMP API request timed out for {ticker_symbol}: {timeout_err}.")
            return None
        except Exception as e:
            st.error(f"⚠️ Unexpected error for {ticker_symbol}: {e}")
            return None
        time.sleep(initial_delay * (2 ** attempt))
    return None

def display_financial_statements(ticker_symbol, fmp_api_key):
    """Displays financial statements (Income Statement, Balance Sheet, Cash Flow) for the given ticker."""
    st.subheader(f"Financial Statements for {ticker_symbol}")
    st.markdown("⚠️ **Disclaimer**: Data sourced from Financial Modeling Prep's free API tier. Limits: 250 requests/day, limited historical data. Accuracy: Data may be incomplete or delayed. **Do NOT rely on this for investment decisions. Verify with official filings (e.g., SEC EDGAR).**")

    # Fetch and display Income Statement
    st.markdown("### Income Statement")
    for period in ["Annual", "Quarterly"]:
        st.markdown(f"#### {period}")
        income_data = fetch_financial_data(ticker_symbol, "income-statement", fmp_api_key, period=period.lower())
        if not income_data.empty:
            # Select key columns for display
            key_columns = ['revenue', 'grossProfit', 'netIncome', 'eps']
            display_data = income_data[key_columns].head(5)  # Show last 5 periods
            display_data = display_data.rename(columns={
                'revenue': 'Revenue ($)',
                'grossProfit': 'Gross Profit ($)',
                'netIncome': 'Net Income ($)',
                'eps': 'EPS ($)'
            })
            # Format numbers
            display_data = display_data.applymap(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
            st.dataframe(display_data, use_container_width=True)
        else:
            st.warning(f"⚠️ No {period.lower()} Income Statement data available.")

    # Fetch and display Balance Sheet
    st.markdown("### Balance Sheet")
    for period in ["Annual", "Quarterly"]:
        st.markdown(f"#### {period}")
        balance_data = fetch_financial_data(ticker_symbol, "balance-sheet-statement", fmp_api_key, period=period.lower())
        if not balance_data.empty:
            key_columns = ['totalAssets', 'totalLiabilities', 'totalEquity']
            display_data = balance_data[key_columns].head(5)
            display_data = display_data.rename(columns={
                'totalAssets': 'Total Assets ($)',
                'totalLiabilities': 'Total Liabilities ($)',
                'totalEquity': 'Total Equity ($)'
            })
            display_data = display_data.applymap(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
            st.dataframe(display_data, use_container_width=True)
        else:
            st.warning(f"⚠️ No {period.lower()} Balance Sheet data available.")

    # Fetch and display Cash Flow
    st.markdown("### Cash Flow")
    for period in ["Annual", "Quarterly"]:
        st.markdown(f"#### {period}")
        cashflow_data = fetch_financial_data(ticker_symbol, "cash-flow-statement", fmp_api_key, period=period.lower())
        if not cashflow_data.empty:
            key_columns = ['netCashProvidedByOperatingActivities', 'netCashUsedForInvestingActivites', 'freeCashFlow']
            display_data = cashflow_data[key_columns].head(5)
            display_data = display_data.rename(columns={
                'netCashProvidedByOperatingActivities': 'Operating Cash Flow ($)',
                'netCashUsedForInvestingActivites': 'Investing Cash Flow ($)',
                'freeCashFlow': 'Free Cash Flow ($)'
            })
            display_data = display_data.applymap(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
            st.dataframe(display_data, use_container_width=True)
        else:
            st.warning(f"⚠️ No {period.lower()} Cash Flow data available.")

def display_stock_summary(ticker_symbol, hist_data, fmp_api_key, alpha_vantage_api_key, gemini_api_key):
    """Displays the stock summary page with price data and company profile."""
    st.subheader(f"Stock Summary for {ticker_symbol}")

    # Fetch company profile
    profile = fetch_stock_data(ticker_symbol, alpha_vantage_api_key, fmp_api_key)

    # Display company profile
    if profile:
        st.markdown("##### Company Profile")
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"**Name**: {profile.get('companyName', 'N/A')}")
            st.markdown(f"**Sector**: {profile.get('sector', 'N/A')}")
            st.markdown(f"**Industry**: {profile.get('industry', 'N/A')}")
        with cols[1]:
            market_cap = profile.get('mktCap', 'N/A')
            try:
                market_cap = f"${float(market_cap):,.0f}" if market_cap != 'N/A' else 'N/A'
            except (ValueError, TypeError):
                market_cap = 'N/A'
            st.markdown(f"**Market Cap**: {market_cap}")
            st.markdown(f"**Exchange**: {profile.get('exchangeShortName', 'N/A')}")
            website = profile.get('website', 'N/A')
            st.markdown(f"**Website**: [{website}]({website if website != 'N/A' else '#'})")
        st.markdown(f"**Description**: {profile.get('description', 'No description available.')}")

        # AI Insights
        st.markdown("##### AI Insights")
        if gemini_api_key and gemini_api_key != "YOUR_GEMINI_API_KEY":
            st.write("AI insights not implemented in this version.")
        else:
            desc = profile.get('description', '')
            st.write(f"Mock analysis: {desc[:100] + '...' if len(desc) > 100 else desc} (AI insights unavailable without Gemini API key).")
    else:
        st.warning(f"⚠️ Could not retrieve company profile for {ticker_symbol}. Check the ticker or API key.")

    # Display price data
    if hist_data is None or hist_data.empty:
        st.error(f"❌ No historical price data available for {ticker_symbol}. Check the ticker or data source.")
        return

    st.markdown("##### Price Data")
    try:
        latest_data = hist_data.iloc[-1]
        prev_data = hist_data.iloc[-2] if len(hist_data) > 1 else latest_data
        price_change = latest_data['Close'] - prev_data['Close']
        price_change_pct = (price_change / prev_data['Close'] * 100) if prev_data['Close'] != 0 else 0

        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"**Current Price**: ${latest_data['Close']:.2f}")
        with cols[1]:
            color = "green" if price_change >= 0 else "red"
            st.markdown(
                f"**Price Change**: <span style='color:{color}'>{price_change:.2f} ({price_change_pct:.2f}%)</span>",
                unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"**52-Week High**: ${hist_data['High'].max():.2f}")
        with cols[3]:
            st.markdown(f"**52-Week Low**: ${hist_data['Low'].min():.2f}")
    except Exception as e:
        st.error(f"❌ Error processing price data for {ticker_symbol}: {e}")
        return

    # Price chart
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#00ACC1')
        ))
        fig.update_layout(
            title=f"Price History for {ticker_symbol}",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error rendering price chart for {ticker_symbol}: {e}")

# Example usage in app.py
def main():
    st.title("Stock Analysis Dashboard")
    ticker_symbol = st.text_input("Enter Stock Ticker", value="AAPL")
    if ticker_symbol:
        # Fetch historical data (example using Alpha Vantage)
        @st.cache_data(ttl=3600, show_spinner="Fetching historical data...")
        def fetch_historical_data(ticker):
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}"
            try:
                response = requests.get(url, timeout=20)
                response.raise_for_status()
                data = response.json()
                if "Time Series (Daily)" not in data:
                    st.error(f"❌ No historical data for {ticker}. Check ticker or API key.")
                    return pd.DataFrame()
                df = pd.DataFrame(data["Time Series (Daily)"]).T
                df = df.rename(columns={
                    "1. open": "Open",
                    "2. high": "High",
                    "3. low": "Low",
                    "4. close": "Close",
                    "5. volume": "Volume"
                })
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                return df
            except Exception as e:
                st.error(f"❌ Error fetching historical data for {ticker}: {e}")
                return pd.DataFrame()

        hist_data = fetch_historical_data(ticker_symbol)
        display_stock_summary(ticker_symbol, hist_data, FMP_API_KEY, ALPHA_VANTAGE_API_KEY, GEMINI_API_KEY)
        display_financial_statements(ticker_symbol, FMP_API_KEY)

if __name__ == "__main__":
    main()