import streamlit as st
import pandas as pd
import requests
import json
import time

@st.cache_data(ttl=3600, show_spinner=False)
def get_financial_statements_data(ticker_symbol, statement_type, api_key, retries=3, initial_delay=1):
    """
    Fetches financial statements (Income Statement, Balance Sheet, Cash Flow)
    from Financial Modeling Prep (FMP) for a given ticker and statement type.
    """
    function_map = {
        "Income Statement": "income-statement",
        "Balance Sheet": "balance-sheet-statement",
        "Cash Flow": "cash-flow-statement"
    }
    
    fmp_function_path = function_map.get(statement_type)
    if not fmp_function_path:
        st.error(f"Invalid statement type: {statement_type}")
        return None, None

    # FMP provides both annual and quarterly reports under the same endpoint
    base_url_annual = f"https://financialmodelingprep.com/api/v3/{fmp_function_path}/{ticker_symbol}"
    base_url_quarterly = f"https://financialmodelingprep.com/api/v3/{fmp_function_path}/{ticker_symbol}"

    params = {
        "apikey": api_key
    }
    
    annual_reports = []
    quarterly_reports = []

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                print(f"Retrying {statement_type} fetch for {ticker_symbol} (attempt {attempt}/{retries}). Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            else:
                time.sleep(initial_delay) # Initial delay before first API call

            # Fetch annual reports
            response_annual = requests.get(base_url_annual, params={"period": "annual", **params}, timeout=15)
            response_annual.raise_for_status()
            data_annual = response_annual.json()
            if data_annual and isinstance(data_annual, list):
                annual_reports = data_annual
            
            # Fetch quarterly reports
            response_quarterly = requests.get(base_url_quarterly, params={"period": "quarter", **params}, timeout=15)
            response_quarterly.raise_for_status()
            data_quarterly = response_quarterly.json()
            if data_quarterly and isinstance(data_quarterly, list):
                quarterly_reports = data_quarterly

            if not annual_reports and not quarterly_reports:
                print(f"No {statement_type} data found for {ticker_symbol}.")
                if "Error Message" in str(data_annual) or "Error Message" in str(data_quarterly):
                     st.error(f"FMP API Error for {ticker_symbol} {statement_type}: {data_annual.get('Error Message', data_quarterly.get('Error Message', 'Unknown API Error'))}. Please check the ticker or API key.")
                elif "limit" in str(data_annual).lower() or "limit" in str(data_quarterly).lower():
                    st.warning(f"⚠️ FMP API rate limit might be hit for {statement_type}. Please wait and try again or check your API key usage.")
                else:
                    st.info(f"No {statement_type} data found for {ticker_symbol}. Data might be unavailable or API limit reached.")
                if attempt == retries:
                    return None, None
                continue
            
            return annual_reports, quarterly_reports

        except requests.exceptions.RequestException as req_err:
            print(f"Attempt {attempt}/{retries}: Network or API request error for {statement_type} {ticker_symbol}: {req_err}")
            if attempt == retries:
                st.error(f"⚠️ Network error or API issue for {statement_type} of {ticker_symbol}. Please check your internet connection or try again later.")
                return None, None
        except json.JSONDecodeError as json_err:
            print(f"Attempt {attempt}/{retries}: JSON Decode Error for {statement_type} {ticker_symbol}: {json_err}. Response content starts with: Annual: {response_annual.text[:200]}..., Quarterly: {response_quarterly.text[:200]}...")
            if attempt == retries:
                st.error(f"⚠️ Received invalid data from FMP API for {statement_type} of {ticker_symbol}. Please try again later.")
                return None, None
        except Exception as e:
            print(f"Attempt {attempt}/{retries}: An unexpected error occurred while fetching {statement_type} for {ticker_symbol}: {e}")
            if attempt == retries:
                st.error(f"⚠️ An unexpected error occurred while fetching {statement_type} for {ticker_symbol}. Please try again later.")
                return None, None
    return None, None # Fallback if all retries fail


def display_financials(ticker_symbol, api_key):
    st.subheader(f"Financial Statements for {ticker_symbol.upper()}")

    st.markdown("---")
    st.markdown("#### Income Statement")
    income_annual, income_quarterly = get_financial_statements_data(ticker_symbol, "Income Statement", api_key)
    if income_annual:
        st.markdown("##### Annual")
        # Ensure 'date' is used as index and all values are handled for potential non-numeric types
        df_annual_income = pd.DataFrame(income_annual).set_index('date')
        df_annual_income = df_annual_income.apply(pd.to_numeric, errors='coerce') # Convert all to numeric where possible
        st.dataframe(df_annual_income.T)
    else:
        st.info("No annual income statement data available.")

    if income_quarterly:
        st.markdown("##### Quarterly")
        df_quarterly_income = pd.DataFrame(income_quarterly).set_index('date')
        df_quarterly_income = df_quarterly_income.apply(pd.to_numeric, errors='coerce')
        st.dataframe(df_quarterly_income.T)
    else:
        st.info("No quarterly income statement data available.")

    st.markdown("---")
    st.markdown("#### Balance Sheet")
    balance_annual, balance_quarterly = get_financial_statements_data(ticker_symbol, "Balance Sheet", api_key)
    if balance_annual:
        st.markdown("##### Annual")
        df_annual_balance = pd.DataFrame(balance_annual).set_index('date')
        df_annual_balance = df_annual_balance.apply(pd.to_numeric, errors='coerce')
        st.dataframe(df_annual_balance.T)
    else:
        st.info("No annual balance sheet data available.")

    if balance_quarterly:
        st.markdown("##### Quarterly")
        df_quarterly_balance = pd.DataFrame(balance_quarterly).set_index('date')
        df_quarterly_balance = df_quarterly_balance.apply(pd.to_numeric, errors='coerce')
        st.dataframe(df_quarterly_balance.T)
    else:
        st.info("No quarterly balance sheet data available.")

    st.markdown("---")
    st.markdown("#### Cash Flow Statement")
    cashflow_annual, cashflow_quarterly = get_financial_statements_data(ticker_symbol, "Cash Flow", api_key)
    if cashflow_annual:
        st.markdown("##### Annual")
        df_annual_cashflow = pd.DataFrame(cashflow_annual).set_index('date')
        df_annual_cashflow = df_annual_cashflow.apply(pd.to_numeric, errors='coerce')
        st.dataframe(df_annual_cashflow.T)
    else:
        st.info("No annual cash flow statement data available.")

    if cashflow_quarterly:
        st.markdown("##### Quarterly")
        df_quarterly_cashflow = pd.DataFrame(cashflow_quarterly).set_index('date')
        df_quarterly_cashflow = df_quarterly_cashflow.apply(pd.to_numeric, errors='coerce')
        st.dataframe(df_quarterly_cashflow.T)
    else:
        st.info("No quarterly cash flow statement data available.")
