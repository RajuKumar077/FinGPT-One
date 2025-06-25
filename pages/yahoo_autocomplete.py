import requests
import json
import streamlit as st
import time

def fetch_ticker_suggestions(query: str, api_key: str, retries: int = 3, initial_delay: float = 1.0) -> list:
    """
    Fetch stock ticker suggestions using Alpha Vantage's Symbol Search Endpoint.
    
    Args:
        query (str): Search query for stock tickers.
        api_key (str): Alpha Vantage API key.
        retries (int): Number of retry attempts for failed requests.
        initial_delay (float): Initial delay before API calls (in seconds).
    
    Returns:
        list: List of formatted ticker suggestions (e.g., "AAPL - Apple Inc. (United States)").
    """
    if not query or not api_key:
        st.error("❌ Missing search query or Alpha Vantage API key.")
        return []

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "SYMBOL_SEARCH",
        "keywords": query,
        "apikey": api_key
    }

    for attempt in range(retries):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                time.sleep(sleep_time)
            else:
                time.sleep(initial_delay)  # Initial delay before first API call

            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            # Handle Alpha Vantage-specific errors
            if "Error Message" in data:
                error_msg = data["Error Message"]
                if "daily limit" in error_msg.lower() or "throttle" in error_msg.lower():
                    st.warning("⚠️ Alpha Vantage API daily limit reached (25 calls/day for free tier). Try again later.")
                elif "invalid api key" in error_msg.lower():
                    st.error("❌ Invalid Alpha Vantage API key. Please check your API key in `app.py`.")
                else:
                    st.error(f"⚠️ Alpha Vantage API error: {error_msg}")
                return []
            elif "Information" in data and "premium endpoint" in data["Information"].lower():
                st.error(f"🚨 Alpha Vantage API error: {data['Information']}. Upgrade to a premium plan or wait for daily limit reset.")
                return []

            # Process suggestions
            best_matches = data.get("bestMatches", [])
            suggestions = [
                f"{item.get('1. symbol')} - {item.get('2. name', '')} ({item.get('4. region', '')})"
                for item in best_matches
                if item.get("1. symbol") and item.get("2. name")
            ]
            return suggestions

        except requests.exceptions.RequestException as req_err:
            if attempt == retries - 1:
                st.error("⚠️ Network error during symbol search. Check your internet connection and try again.")
                return []
        except json.JSONDecodeError as json_err:
            if attempt == retries - 1:
                st.error("⚠️ Invalid data received from Alpha Vantage API. Try again later.")
                return []
        except Exception as e:
            if attempt == retries - 1:
                st.error(f"⚠️ Unexpected error during symbol search: {str(e)}")
                return []

    return []  # Fallback if all retries fail
