import requests
import time
import json
import streamlit as st

@st.cache_data(ttl=3600, show_spinner=False) # Cache suggestions for an hour
def fetch_fmp_suggestions(query, api_key, retries=3, initial_delay=0.5):
    """
    Fetch stock ticker suggestions using Financial Modeling Prep (FMP) Search Endpoint.
    """
    if not query:
        return []
    if not api_key or api_key == "YOUR_FMP_KEY": # Check for unset key
        return []

    base_url = "https://financialmodelingprep.com/api/v3/search"
    params = {
        "query": query,
        "limit": 10, # Limit to 10 suggestions for quick display
        "exchange": "NASDAQ,NYSE,AMEX,TSX,LON,NSE,BSE", # Include major global exchanges and Indian ones
        "apikey": api_key
    }

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                print(f"Retrying FMP symbol search (attempt {attempt}/{retries}). Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)

            response = requests.get(base_url, params=params, timeout=15) # Increased timeout
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if not data: # Empty list usually means no matches
                print(f"No FMP symbol search results found for '{query}'.")
                return []
            
            suggestions = []
            for item in data:
                symbol = item.get("symbol")
                name = item.get("name", "")
                exchange = item.get("exchange", "")
                if symbol and name:
                    suggestions.append(f"{symbol} - {name} ({exchange})")
            return suggestions

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429: # Rate limit
                st.warning(f"⚠️ FMP API rate limit hit for symbol search. Please wait a moment and try again.")
            elif http_err.response.status_code in [401, 403]: # Unauthorized/Forbidden
                 st.error("❌ FMP API key is invalid or unauthorized for symbol search. Please check your FMP_API_KEY in app.py.")
            else:
                st.error(f"❌ FMP Symbol Search: HTTP error occurred: {http_err}. Status: {http_err.response.status_code}")
            if attempt == retries: return []

        except requests.exceptions.ConnectionError as conn_err:
            st.error(f"❌ FMP Symbol Search: Connection error occurred: {conn_err}. Please check your internet connection.")
            if attempt == retries: return []
        except requests.exceptions.Timeout as timeout_err:
            st.error(f"❌ FMP Symbol Search: Request timed out. The server might be slow or unresponsive. Please try again.")
            if attempt == retries: return []
        except json.JSONDecodeError as json_err:
            st.error(f"❌ FMP Symbol Search: Received invalid data from API. Please try again later. Error: {json_err}")
            if attempt == retries: return []
        except Exception as e:
            st.error(f"❌ FMP Symbol Search: An unexpected error occurred: {e}")
            if attempt == retries: return []
    return [] # Fallback if all retries fail
