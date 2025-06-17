import requests
import time
import yfinance as yf # Add yfinance import here as it's used in app.py's load_historical_data
import json # Import json for explicit parsing

def fetch_yahoo_suggestions(query):
    """
    Fetch stock ticker suggestions from Yahoo Finance's unofficial API.
    Supports Indian stocks (e.g., NALCO.NS) and global stocks.
    """
    if not query:
        return []

    # Respectful delay to avoid rate limiting (429)
    # Increased sleep time slightly for better rate limit avoidance
    time.sleep(0.2) 

    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"  # More complete User-Agent
    }

    try:
        response = requests.get(url, headers=headers, timeout=10) # Added timeout
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        # Explicitly attempt to parse JSON, as the error "Expecting value: line 1 column 1 (char 0)"
        # suggests the response content might not be valid JSON.
        try:
            data = response.json()
        except json.JSONDecodeError as json_err:
            print(f"JSON Decode Error: {json_err} - Response content: {response.text[:200]}...") # Log partial content
            return []

        results = data.get("quotes", [])
        suggestions = []
        for item in results:
            symbol = item.get("symbol")
            shortname = item.get("shortname", "")
            exch_disp = item.get("exchDisp", "")
            if symbol and shortname:
                suggestions.append(f"{symbol} - {shortname} ({exch_disp})")
        return suggestions

    except requests.exceptions.Timeout:
        print("Request timed out when fetching Yahoo suggestions.")
        return []
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection Error when fetching Yahoo suggestions: {conn_err}")
        return []
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Error {response.status_code} when fetching Yahoo suggestions: {http_err}")
        if response.status_code == 429:
            print("Rate limited by Yahoo Finance. Please try again after some time.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while fetching Yahoo suggestions: {e}")
        return []

# Note: The if __name__ == "__main__": block is removed as this file is now imported as a module.
