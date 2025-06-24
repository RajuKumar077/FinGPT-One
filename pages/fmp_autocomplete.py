import requests
import streamlit as st
import time # Ensure time is imported for delays if needed

def fetch_fmp_suggestions(query, api_key):
    """
    Fetches stock ticker suggestions from the Financial Modeling Prep (FMP) API.

    Args:
        query (str): The partial ticker symbol or company name to search for.
        api_key (str): Your FMP API key.

    Returns:
        list: A list of formatted suggestions (e.g., "AAPL - Apple Inc."),
              or an empty list if no suggestions are found or an error occurs.
    """
    if not query or not api_key or api_key == "YOUR_FMP_KEY":
        print("DEBUG: FMP API key is not set or query is empty for autocomplete.")
        return []

    # FMP has a search endpoint for symbols
    # Increased limit to 20 for more suggestions
    # CORRECTED URL SYNTAX: Removed markdown link formatting
    url = f"https://financialmodelingprep.com/api/v3/search?query={query}&limit=20&exchange=NASDAQ,NYSE,AMEX,NSE,BSE&apikey={api_key}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        suggestions = []
        if data:
            for item in data:
                symbol = item.get('symbol')
                name = item.get('name')
                exchange = item.get('exchangeShortName')

                # Ensure we only add valid stock symbols with names
                if symbol and name:
                    display_name = f"{symbol} - {name}"
                    if exchange:
                        display_name += f" ({exchange})"
                    suggestions.append(display_name)
        return suggestions
    except requests.exceptions.HTTPError as http_err:
        print(f"DEBUG: FMP Autocomplete HTTP error: {http_err} for query '{query}'. Status: {http_err.response.status_code}. Response: {http_err.response.text}")
        st.error(f"❌ FMP Autocomplete Error: {http_err.response.status_code}. Please check your FMP API key and usage limits for the /search endpoint.")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"DEBUG: FMP Autocomplete Connection error for query '{query}': {conn_err}")
        st.error(f"❌ FMP Autocomplete Connection Error: Please check your internet connection.")
    except requests.exceptions.Timeout as timeout_err:
        print(f"DEBUG: FMP Autocomplete Timeout for query '{query}': {timeout_err}")
        st.error(f"❌ FMP Autocomplete Timeout: The request took too long. Please try again.")
    except json.JSONDecodeError as json_err:
        print(f"DEBUG: FMP Autocomplete JSON decode error for query '{query}': {json_err}. Raw response: {response.text}")
        st.error(f"❌ FMP Autocomplete: Received invalid data from API. Please try again later. This can happen if the API returns an empty or non-JSON response (e.g., due to a rate limit or invalid key).")
    except Exception as e:
        print(f"DEBUG: Unexpected error in FMP Autocomplete for query '{query}': {e}")
        st.error(f"❌ An unexpected error occurred during autocomplete: {e}")
    return []
