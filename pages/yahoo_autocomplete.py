import requests
import time
import json
import streamlit as st

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "9NBXSBBIYEBJHBIP"

def fetch_yahoo_suggestions(query, retries=3, initial_delay=0.75):
    """
    Fetch stock ticker suggestions using Alpha Vantage's Search Endpoint.
    This API is more reliable than Yahoo Finance's unofficial one.
    """
    if not query:
        return []

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "SYMBOL_SEARCH",
        "keywords": query,
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                print(f"Retrying Alpha Vantage symbol search (attempt {attempt}/{retries}). Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                time.sleep(initial_delay) # Initial delay before first API call

            response = requests.get(base_url, params=params, timeout=15) # Increased timeout
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            # Check for API call limits or errors in the response
            if "Error Message" in data:
                error_msg = data["Error Message"]
                print(f"Alpha Vantage API Search Error: {error_msg}")
                if "daily limit" in error_msg.lower() or "throttle" in error_msg.lower():
                    st.warning(f"Alpha Vantage API daily limit reached for symbol search. Please try again later (max 25 calls/day for free tier).")
                else:
                    st.error(f"Alpha Vantage API search error: {error_msg}. Please check query or API key.")
                if attempt == retries:
                    return []
                continue # Retry if not last attempt

            best_matches = data.get("bestMatches", [])
            suggestions = []
            for item in best_matches:
                symbol = item.get("1. symbol")
                name = item.get("2. name", "")
                region = item.get("4. region", "")
                if symbol and name:
                    suggestions.append(f"{symbol} - {name} ({region})")
            return suggestions

        except requests.exceptions.RequestException as req_err:
            print(f"Attempt {attempt}/{retries}: Network or API request error for symbol search: {req_err}")
            if attempt == retries:
                st.error(f"⚠️ Network error or API issue during symbol search. Please check your internet connection or try again later.")
                return []
        except json.JSONDecodeError as json_err:
            print(f"Attempt {attempt}/{retries}: JSON Decode Error for symbol search: {json_err}. Response content starts with: {response.text[:200]}...")
            if attempt == retries:
                st.error(f"⚠️ Received invalid data from API during symbol search. Please try again later.")
                return []
        except Exception as e:
            print(f"Attempt {attempt}/{retries}: An unexpected error occurred during symbol search: {e}")
            if attempt == retries:
                st.error(f"⚠️ An unexpected error occurred during symbol search. Please try again later.")
                return []
    return [] # Fallback if all retries fail
