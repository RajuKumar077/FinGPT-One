import requests
import time
import yfinance as yf # Add yfinance import here as it's used in app.py's load_historical_data

def fetch_yahoo_suggestions(query):
    """
    Fetch stock ticker suggestions from Yahoo Finance's unofficial API.
    Supports Indian stocks (e.g., NALCO.NS) and global stocks.
    """
    if not query:
        return []

    # Respectful delay to avoid rate limiting (429)
    time.sleep(0.1) # Reduced sleep for better responsiveness, but still good practice

    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"

    headers = {
        "User-Agent": "Mozilla/5.0"  # Helps mimic browser and reduces blocking
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        results = data.get("quotes", [])
        suggestions = []
        for item in results:
            symbol = item.get("symbol")
            shortname = item.get("shortname", "")
            exch_disp = item.get("exchDisp", "")
            if symbol and shortname:
                suggestions.append(f"{symbol} - {shortname} ({exch_disp})")
        return suggestions

    except requests.exceptions.HTTPError as http_err:
        # In a Streamlit app, this warning would be handled by st.warning in app.py
        # For a standalone module, a print is fine for debugging.
        # print(f"HTTP Error: {http_err}")
        # if response.status_code == 429:
        #     print("Rate limited. Please wait a bit before trying again.")
        return []
    except Exception as e:
        # print(f"Error fetching Yahoo suggestions: {e}")
        return []

# Note: The if __name__ == "__main__": block is removed as this file is now imported as a module.
