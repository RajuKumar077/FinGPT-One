import streamlit as st
import pandas as pd
import requests
import json
import time

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = "9NBXSBBIYEBJHBIP"

@st.cache_data(ttl=3600, show_spinner=False)
def get_financial_statements_data(ticker_symbol, statement_type, retries=3, initial_delay=1):
    """
    Fetches financial statements (Income Statement, Balance Sheet, Cash Flow)
    from Alpha Vantage for a given ticker and statement type.
    """
    function_map = {
        "Income Statement": "INCOME_STATEMENT",
        "Balance Sheet": "BALANCE_SHEET",
        "Cash Flow": "CASH_FLOW"
    }
    
    alpha_vantage_function = function_map.get(statement_type)
    if not alpha_vantage_function:
        st.error(f"Invalid statement type: {statement_type}")
        return None

    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": alpha_vantage_function,
        "symbol": ticker_symbol,
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                print(f"Retrying {statement_type} fetch for {ticker_symbol} (attempt {attempt}/{retries}). Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            else:
                time.sleep(initial_delay) # Initial delay before first API call

            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                error_msg = data["Error Message"]
                print(f"Alpha Vantage API {statement_type} Error for {ticker_symbol}: {error_msg}")
                if "daily limit" in error_msg.lower() or "throttle" in error_msg.lower():
                    st.warning(f"Alpha Vantage API daily limit reached for {statement_type} of {ticker_symbol}. Please try again later (max 25 calls/day for free tier).")
                else:
                    st.error(f"Alpha Vantage API {statement_type} error for {ticker_symbol}: {error_msg}. Please check the ticker or API key.")
                if attempt == retries:
                    return None
                continue

            # Alpha Vantage returns statements in lists under specific keys
            reports_key_map = {
                "Income Statement": ["annualReports", "quarterlyReports"],
                "Balance Sheet": ["annualReports", "quarterlyReports"],
                "Cash Flow": ["annualReports", "quarterlyReports"]
            }
            
            annual_reports = data.get(reports_key_map[statement_type][0], [])
            quarterly_reports = data.get(reports_key_map[statement_type][1], [])

            if not annual_reports and not quarterly_reports:
                print(f"No {statement_type} data found for {ticker_symbol}.")
                if attempt == retries:
                    st.info(f"No {statement_type} data found for {ticker_symbol}. Data might be unavailable or API limit reached.")
                return None
            
            return annual_reports, quarterly_reports

        except requests.exceptions.RequestException as req_err:
            print(f"Attempt {attempt}/{retries}: Network or API request error for {statement_type} {ticker_symbol}: {req_err}")
            if attempt == retries:
                st.error(f"⚠️ Network error or API issue for {statement_type} of {ticker_symbol}. Please check your internet connection or try again later.")
                return None
        except json.JSONDecodeError as json_err:
            print(f"Attempt {attempt}/{retries}: JSON Decode Error for {statement_type} {ticker_symbol}: {json_err}. Response content starts with: {response.text[:200]}...")
            if attempt == retries:
                st.error(f"⚠️ Received invalid data from API for {statement_type} of {ticker_symbol}. Please try again later.")
                return None
        except Exception as e:
            print(f"Attempt {attempt}/{retries}: An unexpected error occurred while fetching {statement_type} for {ticker_symbol}: {e}")
            if attempt == retries:
                st.error(f"⚠️ An unexpected error occurred while fetching {statement_type} for {ticker_symbol}. Please try again later.")
                return None
    return None # Fallback if all retries fail


def display_financials(ticker_symbol):
    st.subheader(f"Financial Statements for {ticker_symbol.upper()}")

    st.markdown("---")
    st.markdown("#### Income Statement")
    income_annual, income_quarterly = get_financial_statements_data(ticker_symbol, "Income Statement")
    if income_annual:
        st.markdown("##### Annual")
        df_annual_income = pd.DataFrame(income_annual).set_index('fiscalDateEnding')
        st.dataframe(df_annual_income.T)
    else:
        st.info("No annual income statement data available.")

    if income_quarterly:
        st.markdown("##### Quarterly")
        df_quarterly_income = pd.DataFrame(income_quarterly).set_index('fiscalDateEnding')
        st.dataframe(df_quarterly_income.T)
    else:
        st.info("No quarterly income statement data available.")

    st.markdown("---")
    st.markdown("#### Balance Sheet")
    balance_annual, balance_quarterly = get_financial_statements_data(ticker_symbol, "Balance Sheet")
    if balance_annual:
        st.markdown("##### Annual")
        df_annual_balance = pd.DataFrame(balance_annual).set_index('fiscalDateEnding')
        st.dataframe(df_annual_balance.T)
    else:
        st.info("No annual balance sheet data available.")

    if balance_quarterly:
        st.markdown("##### Quarterly")
        df_quarterly_balance = pd.DataFrame(balance_quarterly).set_index('fiscalDateEnding')
        st.dataframe(df_quarterly_balance.T)
    else:
        st.info("No quarterly balance sheet data available.")

    st.markdown("---")
    st.markdown("#### Cash Flow Statement")
    cashflow_annual, cashflow_quarterly = get_financial_statements_data(ticker_symbol, "Cash Flow")
    if cashflow_annual:
        st.markdown("##### Annual")
        df_annual_cashflow = pd.DataFrame(cashflow_annual).set_index('fiscalDateEnding')
        st.dataframe(df_annual_cashflow.T)
    else:
        st.info("No annual cash flow statement data available.")

    if cashflow_quarterly:
        st.markdown("##### Quarterly")
        df_quarterly_cashflow = pd.DataFrame(cashflow_quarterly).set_index('fiscalDateEnding')
        st.dataframe(df_quarterly_cashflow.T)
    else:
        st.info("No quarterly cash flow statement data available.")
