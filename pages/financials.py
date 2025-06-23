import streamlit as st
import pandas as pd
import requests
import json
import time

@st.cache_data(ttl=3600, show_spinner=False) # Cache financial statement data for 1 hour
def get_financial_statements_data(ticker_symbol, statement_type, api_key, retries=3, initial_delay=1):
    """
    Fetches financial statements (Income Statement, Balance Sheet, Cash Flow)
    from Financial Modeling Prep (FMP) for a given ticker and statement type.
    """
    if not api_key or api_key == "YOUR_FMP_KEY":
        st.error("❌ FMP_API_KEY is not set. Financial statements cannot be loaded from FMP.")
        return None, None

    function_map = {
        "Income Statement": "income-statement",
        "Balance Sheet": "balance-sheet-statement",
        "Cash Flow": "cash-flow-statement"
    }
    
    fmp_function_path = function_map.get(statement_type)
    if not fmp_function_path:
        st.error(f"Invalid statement type: {statement_type}")
        return None, None

    base_url = f"https://financialmodelingprep.com/api/v3/{fmp_function_path}/{ticker_symbol}"
    params = {"apikey": api_key}
    
    annual_reports = []
    quarterly_reports = []

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                print(f"Retrying {statement_type} fetch for {ticker_symbol} (attempt {attempt}/{retries}). Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)

            # Fetch annual reports
            response_annual = requests.get(base_url, params={"period": "annual", **params}, timeout=15)
            response_annual.raise_for_status()
            data_annual = response_annual.json()
            if data_annual and isinstance(data_annual, list):
                annual_reports = data_annual
            
            # Fetch quarterly reports
            response_quarterly = requests.get(base_url, params={"period": "quarter", **params}, timeout=15)
            response_quarterly.raise_for_status()
            data_quarterly = response_quarterly.json()
            if data_quarterly and isinstance(data_quarterly, list):
                quarterly_reports = data_quarterly

            # Check for FMP specific error messages in response content
            if isinstance(data_annual, dict) and "Error Message" in data_annual:
                st.warning(f"⚠️ FMP API Error for {ticker_symbol} {statement_type} (Annual): {data_annual['Error Message']}")
            if isinstance(data_quarterly, dict) and "Error Message" in data_quarterly:
                st.warning(f"⚠️ FMP API Error for {ticker_symbol} {statement_type} (Quarterly): {data_quarterly['Error Message']}")

            if not annual_reports and not quarterly_reports:
                print(f"No {statement_type} data found for {ticker_symbol} on FMP.")
                if attempt == retries:
                    st.info(f"No {statement_type} data found for {ticker_symbol} on FMP's free tier.")
                    return None, None
                continue # Retry if both are empty or error was received

            return annual_reports, quarterly_reports

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429: # Rate limit
                st.warning(f"⚠️ FMP API rate limit hit for {statement_type} of {ticker_symbol}. Please wait and try again or check your API key usage.")
            elif http_err.response.status_code in [401, 403]: # Unauthorized/Forbidden
                st.error("❌ FMP API key is invalid or unauthorized. Please check your FMP_API_KEY in app.py.")
            else:
                st.error(f"❌ FMP {statement_type}: HTTP error occurred: {http_err}. Status: {http_err.response.status_code}")
            if attempt == retries: return None, None

        except requests.exceptions.ConnectionError as conn_err:
            st.error(f"❌ FMP {statement_type}: Connection error occurred: {conn_err}. Please check your internet connection.")
            if attempt == retries: return None, None
        except requests.exceptions.Timeout as timeout_err:
            st.error(f"❌ FMP {statement_type}: Request timed out. The server might be slow or unresponsive. Please try again.")
            if attempt == retries: return None, None
        except json.JSONDecodeError as json_err:
            st.error(f"❌ FMP {statement_type}: Received invalid data from API. Please try again later. Error: {json_err}")
            if attempt == retries: return None, None
        except Exception as e:
            st.error(f"❌ FMP {statement_type}: An unexpected error occurred: {e}")
            if attempt == retries: return None, None
    return None, None # Fallback if all retries fail


def display_financials(ticker_symbol, fmp_api_key):
    st.subheader(f"Financial Statements for {ticker_symbol.upper()}")
    st.warning("""
    ⚠️ **Disclaimer for Financial Statements:**
    This data is sourced exclusively from Financial Modeling Prep's (FMP) free API tier.
    - **Data Availability:** Comprehensive historical data may be limited or unavailable for certain tickers (especially less liquid or international ones) on the free tier.
    - **Rate Limits:** FMP's free tier has daily API request limits (e.g., 250 requests/day). Frequent requests for different tickers/statements might hit these limits, causing temporary data unavailability.
    - **Accuracy & Delays:** Data may be incomplete, delayed, or subject to FMP's own data collection and processing.

    **DO NOT rely on this data for critical financial analysis or investment decisions.** Always cross-verify with official company filings (e.g., SEC EDGAR, BSE/NSE filings) and reputable paid financial data providers.
    """)

    st.markdown("---")
    
    # Define a common function to display DataFrame or a message
    def display_statement_df(df, period_type):
        if df is not None and not df.empty:
            # Drop unnecessary columns that might appear from FMP but are not financial data
            cols_to_drop = ['cik', 'finalLink', 'fillingDate', 'acceptedDate', 'period', 'link', 'reportedCurrency', 'symbol']
            df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
            
            # Set 'date' as index and transpose for better readability
            if 'date' in df.columns:
                df = df.set_index('date')
            else: # If 'date' is not a column, create a dummy index for transpose if possible
                df.index = [f"Report {i+1}" for i in range(len(df))]

            # Convert all data to numeric where possible, coercing errors to NaN
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # Fill NaN values with a readable indicator for display
            df = df.fillna('N/A')

            st.dataframe(df.T, use_container_width=True) # Transpose for dates as columns
        else:
            st.info(f"No {period_type} data available for this statement.")


    st.markdown("#### Income Statement")
    with st.spinner("Loading Income Statement..."):
        income_annual, income_quarterly = get_financial_statements_data(ticker_symbol, "Income Statement", fmp_api_key)
        st.markdown("##### Annual")
        display_statement_df(pd.DataFrame(income_annual) if income_annual else None, "annual")
        st.markdown("##### Quarterly")
        display_statement_df(pd.DataFrame(income_quarterly) if income_quarterly else None, "quarterly")

    st.markdown("---")
    st.markdown("#### Balance Sheet")
    with st.spinner("Loading Balance Sheet..."):
        balance_annual, balance_quarterly = get_financial_statements_data(ticker_symbol, "Balance Sheet", fmp_api_key)
        st.markdown("##### Annual")
        display_statement_df(pd.DataFrame(balance_annual) if balance_annual else None, "annual")
        st.markdown("##### Quarterly")
        display_statement_df(pd.DataFrame(balance_quarterly) if balance_quarterly else None, "quarterly")

    st.markdown("---")
    st.markdown("#### Cash Flow Statement")
    with st.spinner("Loading Cash Flow Statement..."):
        cashflow_annual, cashflow_quarterly = get_financial_statements_data(ticker_symbol, "Cash Flow", fmp_api_key)
        st.markdown("##### Annual")
        display_statement_df(pd.DataFrame(cashflow_annual) if cashflow_annual else None, "annual")
        st.markdown("##### Quarterly")
        display_statement_df(pd.DataFrame(cashflow_quarterly) if cashflow_quarterly else None, "quarterly")
