import requests
import time
import json
import streamlit as st

def fetch_fmp_suggestions(query, api_key, retries=3, initial_delay=0.75):
    """
    Fetch stock ticker suggestions using Financial Modeling Prep (FMP) Search Endpoint.
    """
    if not query:
        return []

    base_url = "https://financialmodelingprep.com/api/v3/search"
    params = {
        "query": query,
        "limit": 10, # Limit to 10 suggestions
        "exchange": "NASDAQ,NYSE,AMEX,NSE,BSE", # Include major global exchanges and Indian ones
        "apikey": api_key
    }

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                print(f"Retrying FMP symbol search (attempt {attempt}/{retries}). Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                time.sleep(initial_delay) # Initial delay before first API call

            response = requests.get(base_url, params=params, timeout=15) # Increased timeout
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if not data: # Empty list usually means no matches
                print(f"No FMP symbol search results found for '{query}'.")
                if "Error Message" in str(data): # Check for FMP-specific error messages
                    st.error(f"FMP API Search Error: {data.get('Error Message', 'Unknown API Error')}. Please check query or API key.")
                elif "limit" in str(data).lower(): # Generic check for rate limit messages
                    st.warning(f"⚠️ FMP API rate limit might be hit for symbol search. Please wait and try again or check your API key usage.")
                return []
            
            suggestions = []
            for item in data:
                symbol = item.get("symbol")
                name = item.get("name", "")
                exchange = item.get("exchange", "")
                if symbol and name:
                    suggestions.append(f"{symbol} - {name} ({exchange})")
            return suggestions

        except requests.exceptions.RequestException as req_err:
            print(f"Attempt {attempt}/{retries}: Network or API request error for FMP symbol search: {req_err}")
            if attempt == retries:
                st.error(f"⚠️ Network error or API issue during FMP symbol search. Please check your internet connection or try again later.")
                return []
        except json.JSONDecodeError as json_err:
            print(f"Attempt {attempt}/{retries}: JSON Decode Error for FMP symbol search: {json_err}. Response content starts with: {response.text[:200]}...")
            if attempt == retries:
                st.error(f"⚠️ Received invalid data from FMP API during symbol search. Please try again later.")
                return []
        except Exception as e:
            print(f"Attempt {attempt}/{retries}: An unexpected error occurred during FMP symbol search: {e}")
            if attempt == retries:
                st.error(f"⚠️ An unexpected error occurred during FMP symbol search. Please try again later.")
                return []
    return [] # Fallback if all retries fail
