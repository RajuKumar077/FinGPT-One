import streamlit as st
import requests
import json
import time

@st.cache_data(ttl=3600, show_spinner=False)
def get_company_overview(ticker_symbol, api_key, retries=3, initial_delay=1):
    """Fetches company overview data from Financial Modeling Prep (FMP)."""
    base_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker_symbol}"
    params = {
        "apikey": api_key
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

            if not data or not isinstance(data, list) or not data[0]: # FMP returns a list
                print(f"No company overview data found for {ticker_symbol}. Response: {data}")
                if "Error Message" in str(data): # Check for FMP-specific error messages
                    st.error(f"FMP API Error for {ticker_symbol}: {data.get('Error Message', 'Unknown API Error')}. Please check the ticker or API key.")
                elif "limit" in str(data).lower(): # Generic check for rate limit messages
                    st.warning(f"⚠️ FMP API rate limit might be hit for company overview. Please wait and try again or check your API key usage.")
                else:
                    st.error(f"❌ No company overview data found for {ticker_symbol}. Please check the ticker symbol.")
                if attempt == retries:
                    return None
                continue
            
            return data[0] # FMP returns a list with the company profile as the first item
        except requests.exceptions.RequestException as req_err:
            print(f"Attempt {attempt}/{retries}: Network or API request error for company overview {ticker_symbol}: {req_err}")
            if attempt == retries:
                st.error(f"⚠️ Network error or API issue for company overview of {ticker_symbol}. Please check your internet connection or try again later.")
                return None
        except json.JSONDecodeError as json_err:
            print(f"Attempt {attempt}/{retries}: JSON Decode Error for company overview {ticker_symbol}: {json_err}. Response content starts with: {response.text[:200]}...")
            if attempt == retries:
                st.error(f"⚠️ Received invalid data from FMP API for company overview of {ticker_symbol}. Please try again later.")
                return None
        except Exception as e:
            print(f"Attempt {attempt}/{retries}: An unexpected error occurred while fetching company overview for {ticker_symbol}: {e}")
            if attempt == retries:
                st.error(f"⚠️ An unexpected error occurred while fetching company overview for {ticker_symbol}. Please try again later.")
                return None
    return None # Fallback if all retries fail

def format_value(value, is_currency=False):
    """Helper function to format large numbers for display."""
    if value is None or value == 'None' or value == '':
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


def display_stock_summary(ticker_symbol, api_key):
    st.subheader(f"Company Overview for {ticker_symbol.upper()}")
    
    overview = get_company_overview(ticker_symbol, api_key)

    if overview:
        st.markdown(f"**{overview.get('companyName', 'N/A')} ({overview.get('symbol', 'N/A')})**")
        st.markdown(f"**Exchange:** {overview.get('exchange', 'N/A')} | **Industry:** {overview.get('industry', 'N/A')}")
        st.markdown(f"**Description:** {overview.get('description', 'N/A')}")

        st.markdown("---")
        st.subheader("Key Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Market Capitalization", format_value(overview.get('mktCap'), is_currency=True))
            st.metric("PE Ratio", format_value(overview.get('peRatio')))
            st.metric("Beta", format_value(overview.get('beta')))
            st.metric("52 Week High", format_value(overview.get('52WeekHigh'), is_currency=True))
            # FMP's profile endpoint doesn't have analyst target price directly, usually in a separate endpoint
            st.metric("Price", format_value(overview.get('price'), is_currency=True)) # Use current price
            
        with col2:
            st.metric("Dividend Yield", f"{float(overview.get('dividendYield', 0)) * 100:.2f}%")
            st.metric("EPS", format_value(overview.get('eps'), is_currency=True))
            st.metric("Book Value", format_value(overview.get('bookValue'), is_currency=True))
            st.metric("52 Week Low", format_value(overview.get('52WeekLow'), is_currency=True))
            st.metric("Price to Book Ratio", format_value(overview.get('priceToBookRatio'))) # Adjusted name
            
        with col3:
            st.metric("Profit Margin", f"{float(overview.get('profitMargin', 0)) * 100:.2f}%")
            st.metric("Revenue", format_value(overview.get('revenue'), is_currency=True)) # Use total revenue
            st.metric("EBITDA", format_value(overview.get('ebitda'), is_currency=True))
            st.metric("Shares Outstanding", format_value(overview.get('sharesOutstanding')))
            st.metric("Volume", format_value(overview.get('volume'))) # Current volume

        st.markdown("---")
        st.subheader("Company Details")
        st.write(f"**CEO:** {overview.get('ceo', 'N/A')}")
        st.write(f"**Website:** [{overview.get('website', 'N/A')}]({overview.get('website', '#')})")
        st.write(f"**Country:** {overview.get('country', 'N/A')}")
        st.write(f"**Currency:** {overview.get('currency', 'N/A')}")
        st.write(f"**Fiscal Year End:** {overview.get('lastDiv', 'N/A')}") # Using lastDiv as a proxy, FMP profile doesn't have direct fiscalYearEnd

    else:
        st.info(f"Could not load comprehensive company overview for {ticker_symbol}. Data might be unavailable, API limit reached, or ticker is incorrect.")

