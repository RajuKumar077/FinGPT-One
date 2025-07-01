import requests
import streamlit as st
import time
import urllib.parse

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fmp_suggestions(query, api_key, retries=3, initial_delay=0.5):
    """
    Fetches stock ticker suggestions from the Financial Modeling Prep (FMP) API.

    Args:
        query (str): The partial ticker symbol or company name to search for.
        api_key (str): Your FMP API key.
        retries (int): Number of retry attempts for API requests.
        initial_delay (float): Initial delay for retries in seconds.

    Returns:
        list: A list of formatted suggestions (e.g., "AAPL - Apple Inc. (NASDAQ)"),
              or an empty list if no suggestions are found or an error occurs.
    """
    if not query or not isinstance(query, str):
        st.error("❌ Invalid search query. Please enter a valid ticker or company name.")
        return []

    if not api_key or api_key == "YOUR_FMP_KEY":
        st.error("❌ FMP API key is missing or invalid in `app.py`. Autocomplete disabled.")
        return []

    # Sanitize query to prevent URL encoding issues
    query = urllib.parse.quote(query.strip())

    # Use major exchanges to cover most stocks
    url = f"https://financialmodelingprep.com/api/v3/search?query={query}&limit=20&exchange=NASDAQ,NYSE,AMEX,NSE,BSE,TSX&apikey={api_key}"

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                time.sleep(sleep_time)

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            suggestions = []
            if data:
                for item in data:
                    symbol = item.get('symbol')
                    name = item.get('name')
                    exchange = item.get('exchangeShortName')
                    if symbol and name:
                        display_name = f"{symbol} - {name}"
                        if exchange:
                            display_name += f" ({exchange})"
                        suggestions.append(display_name)
            return suggestions[:20]  # Ensure max 20 suggestions
        except requests.exceptions.HTTPError as http_err:
            if attempt == retries:
                if http_err.response.status_code == 429:
                    st.error("⚠️ FMP API rate limit reached (250 requests/day). Try again later.")
                elif http_err.response.status_code in [401, 403]:
                    st.error("❌ Invalid FMP API key. Verify `FMP_API_KEY` in `app.py`.")
                else:
                    st.error(f"⚠️ FMP Autocomplete HTTP error: {http_err} (Status: {http_err.response.status_code})")
                return []
        except requests.exceptions.ConnectionError:
            if attempt == retries:
                st.error("⚠️ FMP Autocomplete connection error. Check your internet connection.")
                return []
        except requests.exceptions.Timeout:
            if attempt == retries:
                st.error("⚠️ FMP Autocomplete request timed out. Try again later.")
                return []
        except requests.exceptions.JSONDecodeError:
            if attempt == retries:
                st.error("⚠️ FMP Autocomplete returned invalid data. Check API key or try again later.")
                return []
        except Exception as e:
            if attempt == retries:
                st.error(f"⚠️ FMP Autocomplete unexpected error: {e}")
                return []
    return []
