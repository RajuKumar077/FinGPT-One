import requests
import time
import yfinance as yf # Add yfinance import here as it's used in app.py's load_historical_data
import json # Import json for explicit parsing

def fetch_yahoo_suggestions(query, retries=3, initial_delay=0.5):
    """
    Fetch stock ticker suggestions from Yahoo Finance's unofficial API.
    Supports Indian stocks (e.g., NALCO.NS) and global stocks.
    Includes retry logic for robustness.
    """
    if not query:
        return []

    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for attempt in range(retries + 1): # Try original attempt + 'retries' number of extra attempts
        try:
            # Respectful delay before each attempt to avoid rate limiting
            if attempt > 0:
                # Exponential backoff: delay doubles with each retry
                sleep_time = initial_delay * (2 ** (attempt - 1))
                print(f"Retrying Yahoo suggestion fetch (attempt {attempt}/{retries}). Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            else:
                # Initial delay for the first attempt
                time.sleep(initial_delay)

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            data = response.json() # Attempt JSON parsing
            
            # If we reach here, the request was successful and JSON was parsed.
            results = data.get("quotes", [])
            suggestions = []
            for item in results:
                symbol = item.get("symbol")
                shortname = item.get("shortname", "")
                exch_disp = item.get("exchDisp", "")
                if symbol and shortname:
                    suggestions.append(f"{symbol} - {shortname} ({exch_disp})")
            return suggestions

        except json.JSONDecodeError as json_err:
            print(f"Attempt {attempt}/{retries}: JSON Decode Error: {json_err} - Response content starts with: {response.text[:200]}...")
            if attempt == retries: # If this was the last attempt
                print("Max retries reached for JSON decode error.")
                return []
        except requests.exceptions.Timeout:
            print(f"Attempt {attempt}/{retries}: Request timed out when fetching Yahoo suggestions.")
            if attempt == retries:
                return []
        except requests.exceptions.ConnectionError as conn_err:
            print(f"Attempt {attempt}/{retries}: Connection Error when fetching Yahoo suggestions: {conn_err}")
            if attempt == retries:
                return []
        except requests.exceptions.HTTPError as http_err:
            print(f"Attempt {attempt}/{retries}: HTTP Error {response.status_code} when fetching Yahoo suggestions: {http_err}")
            if response.status_code == 429:
                print("Rate limited by Yahoo Finance. Automatically retrying...")
            if attempt == retries:
                return []
        except Exception as e:
            print(f"Attempt {attempt}/{retries}: An unexpected error occurred while fetching Yahoo suggestions: {e}")
            if attempt == retries:
                return []
    return [] # Should not be reached if retries are exhausted and error occurs, but as a fallback

# Note: The if __name__ == "__main__": block is removed as this file is now imported as a module.
