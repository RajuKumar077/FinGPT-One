import streamlit as st
import requests
import json
import time

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "9NBXSBBIYEBJHBIP"

@st.cache_data(ttl=3600, show_spinner=False)
def get_company_overview(ticker_symbol, retries=3, initial_delay=1):
    """Fetches company overview data from Alpha Vantage."""
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": ticker_symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                print(f"Retrying company overview fetch for {ticker_symbol} (attempt {attempt}/{retries}). Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            else:
                time.sleep(initial_delay) # Initial delay before first API call

            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                error_msg = data["Error Message"]
                print(f"Alpha Vantage API Overview Error for {ticker_symbol}: {error_msg}")
                if "daily limit" in error_msg.lower() or "throttle" in error_msg.lower():
                    st.warning(f"Alpha Vantage API daily limit reached for company overview of {ticker_symbol}. Please try again later (max 25 calls/day for free tier).")
                else:
                    st.error(f"Alpha Vantage API overview error for {ticker_symbol}: {error_msg}. Please check the ticker or API key.")
                if attempt == retries:
                    return None
                continue

            if not data: # Empty dictionary usually means no data for the symbol
                print(f"No company overview data found for {ticker_symbol}.")
                if attempt == retries:
                    st.error(f"❌ No company overview data found for {ticker_symbol}. Please check the ticker symbol.")
                return None
            
            return data
        except requests.exceptions.RequestException as req_err:
            print(f"Attempt {attempt}/{retries}: Network or API request error for company overview {ticker_symbol}: {req_err}")
            if attempt == retries:
                st.error(f"⚠️ Network error or API issue for company overview of {ticker_symbol}. Please check your internet connection or try again later.")
                return None
        except json.JSONDecodeError as json_err:
            print(f"Attempt {attempt}/{retries}: JSON Decode Error for company overview {ticker_symbol}: {json_err}. Response content starts with: {response.text[:200]}...")
            if attempt == retries:
                st.error(f"⚠️ Received invalid data from API for company overview of {ticker_symbol}. Please try again later.")
                return None
        except Exception as e:
            print(f"Attempt {attempt}/{retries}: An unexpected error occurred while fetching company overview for {ticker_symbol}: {e}")
            if attempt == retries:
                st.error(f"⚠️ An unexpected error occurred while fetching company overview for {ticker_symbol}. Please try again later.")
                return None
    return None # Fallback if all retries fail

def format_value(value, is_currency=False):
    """Helper function to format large numbers for display."""
    if value is None or value == 'None':
        return "N/A"
    try:
        num = float(value)
        if abs(num) >= 1e12:
            return f"${num/1e12:.2f}T" if is_currency else f"{num/1e12:.2f}T"
        elif abs(num) >= 1e9:
            return f"${num/1e9:.2f}B" if is_currency else f"{num/1e9:.2f}B"
        elif abs(num) >= 1e6:
            return f"${num/1e6:.2f}M" if is_currency else f"{num/1e6:.2f}M"
        elif abs(num) >= 1e3:
            return f"${num/1e3:.2f}K" if is_currency else f"{num/1e3:.2f}K"
        elif num == int(num):
            return f"${int(num):,}" if is_currency else f"{int(num):,}"
        else:
            return f"${num:.2f}" if is_currency else f"{num:.2f}"
    except (ValueError, TypeError):
        return str(value)


def display_stock_summary(ticker_symbol):
    st.subheader(f"Company Overview for {ticker_symbol.upper()}")
    
    overview = get_company_overview(ticker_symbol)

    if overview:
        st.markdown(f"**{overview.get('Name', 'N/A')} ({overview.get('Symbol', 'N/A')})**")
        st.markdown(f"**Exchange:** {overview.get('Exchange', 'N/A')}")
        st.markdown(f"**Sector:** {overview.get('Sector', 'N/A')} | **Industry:** {overview.get('Industry', 'N/A')}")
        st.markdown(f"**Description:** {overview.get('Description', 'N/A')}")

        st.markdown("---")
        st.subheader("Key Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Market Capitalization", format_value(overview.get('MarketCapitalization'), is_currency=True))
            st.metric("PE Ratio", format_value(overview.get('PERatio')))
            st.metric("Beta", format_value(overview.get('Beta')))
            st.metric("52 Week High", format_value(overview.get('52WeekHigh'), is_currency=True))
            st.metric("Analyst Target Price", format_value(overview.get('AnalystTargetPrice'), is_currency=True))
            
        with col2:
            st.metric("Dividend Yield", f"{float(overview.get('DividendYield', 0)) * 100:.2f}%")
            st.metric("EPS", format_value(overview.get('EPS'), is_currency=True))
            st.metric("Book Value", format_value(overview.get('BookValue'), is_currency=True))
            st.metric("52 Week Low", format_value(overview.get('52WeekLow'), is_currency=True))
            st.metric("Forward PE", format_value(overview.get('ForwardPE')))

        with col3:
            st.metric("Profit Margin", f"{float(overview.get('ProfitMargin', 0)) * 100:.2f}%")
            st.metric("Revenue Per Share", format_value(overview.get('RevenuePerShare'), is_currency=True))
            st.metric("EBITDA", format_value(overview.get('EBITDA'), is_currency=True))
            st.metric("Shares Outstanding", format_value(overview.get('SharesOutstanding')))
            st.metric("Price to Book Ratio", format_value(overview.get('PriceToBookRatio')))
        
        st.markdown("---")
        st.subheader("Company Details")
        st.write(f"**Address:** {overview.get('Address', 'N/A')}")
        st.write(f"**Fiscal Year End:** {overview.get('FiscalYearEnd', 'N/A')}")
        st.write(f"**Latest Quarter:** {overview.get('LatestQuarter', 'N/A')}")
        st.write(f"**Currency:** {overview.get('Currency', 'N/A')}")

    else:
        st.info(f"Could not load comprehensive company overview for {ticker_symbol}. Data might be unavailable or API limit reached.")
