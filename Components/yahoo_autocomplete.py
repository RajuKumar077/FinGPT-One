import requests
import json
import streamlit as st
import time

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_suggestions(query: str, api_key: str, retries: int = 5, initial_delay: float = 1.0) -> list:
    """
    Fetch stock ticker suggestions using Alpha Vantage's Symbol Search Endpoint.
    
    Args:
        query (str): Search query for stock tickers (e.g., 'Apple' or 'AAPL').
        api_key (str): Alpha Vantage API key.
        retries (int): Number of retry attempts for failed requests.
        initial_delay (float): Initial delay before API calls (in seconds).
    
    Returns:
        list: List of formatted ticker suggestions (e.g., "AAPL - Apple Inc. (United States)").
    """
    if not query or not api_key or api_key == "YOUR_ALPHA_VANTAGE_API_KEY":
        st.error("❌ Missing search query or invalid Alpha Vantage API key in `app.py`.")
        return []

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "SYMBOL_SEARCH",
        "keywords": query.strip(),
        "apikey": api_key
    }

    for attempt in range(retries):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                time.sleep(sleep_time)
            else:
                time.sleep(initial_delay)

            response = requests.get(base_url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                error_msg = data["Error Message"].lower()
                if "daily limit" in error_msg or "throttle" in error_msg:
                    st.warning("⚠️ Alpha Vantage API daily limit reached (25 calls/day for free tier). Try again later or check your API key.")
                elif "invalid api key" in error_msg:
                    st.error("❌ Invalid Alpha Vantage API key. Verify `ALPHA_VANTAGE_API_KEY` in `app.py`.")
                else:
                    st.error(f"⚠️ Alpha Vantage API error: {error_msg}")
                return []
            elif "Information" in data and "premium endpoint" in data["Information"].lower():
                st.error(f"⚠️ Alpha Vantage API error: {data['Information']}. Upgrade to a premium plan or wait for limit reset.")
                return []

            best_matches = data.get("bestMatches", [])
            suggestions = [
                f"{item.get('1. symbol')} - {item.get('2. name', 'Unknown')} ({item.get('4. region', 'Unknown')})"
                for item in best_matches
                if item.get("1. symbol") and item.get("2. name")
            ]
            if not suggestions:
                st.info(f"No ticker suggestions found for '{query}'.")
            return suggestions

        except requests.exceptions.HTTPError as http_err:
            if attempt == retries - 1:
                st.error(f"⚠️ HTTP error during symbol search: {http_err}. Check API key or internet connection.")
                return []
        except requests.exceptions.RequestException as req_err:
            if attempt == retries - 1:
                st.error(f"⚠️ Network error during symbol search: {req_err}. Check your internet connection.")
                return []
        except json.JSONDecodeError:
            if attempt == retries - 1:
                st.error("⚠️ Invalid data received from Alpha Vantage API. Try again later.")
                return []
        except Exception as e:
            if attempt == retries - 1:
                st.error(f"⚠️ Unexpected error during symbol search: {str(e)}")
                return []

    return []
