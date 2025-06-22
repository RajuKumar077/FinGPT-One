import streamlit as st
import pandas as pd
import yfinance as yf # Using yfinance for financial statements

@st.cache_data(ttl=3600, show_spinner=False)
def get_financial_statements_data_yf(ticker_symbol):
    """
    Fetches financial statements (Income Statement, Balance Sheet, Cash Flow)
    from yfinance for a given ticker.
    
    Returns a tuple of (income_statements_df, balance_sheets_df, cash_flow_statements_df).
    Each DataFrame will be empty if data is not available or if there's an error.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # yfinance provides these as pandas DataFrames
        income_stmt = ticker.income_stmt
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow
        
        # Convert index to string for consistent display and transpose for better readability
        # Also, ensure values are numeric for display
        income_stmt_df = pd.DataFrame(income_stmt).apply(pd.to_numeric, errors='coerce').astype(str) if income_stmt is not None else pd.DataFrame()
        balance_sheet_df = pd.DataFrame(balance_sheet).apply(pd.to_numeric, errors='coerce').astype(str) if balance_sheet is not None else pd.DataFrame()
        cash_flow_df = pd.DataFrame(cash_flow).apply(pd.to_numeric, errors='coerce').astype(str) if cash_flow is not None else pd.DataFrame()

        # Transpose for a more readable format in Streamlit
        return income_stmt_df.T, balance_sheet_df.T, cash_flow_df.T

    except Exception as e:
        st.warning(f"⚠️ Could not load financial statements for {ticker_symbol} from yfinance: {e}. Data might be unavailable or the yfinance scraping method has changed.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Return empty DataFrames on error


def display_financials(ticker_symbol, api_key=None): # api_key is no longer primarily used here, but kept for function signature consistency
    st.subheader(f"Financial Statements for {ticker_symbol.upper()}")
    st.warning("⚠️ **Disclaimer for Financials:** This data is sourced from `yfinance`, which unofficially scrapes Yahoo Finance. While generally reliable, the availability and format of fundamental data (Income Statement, Balance Sheet, Cash Flow) can be inconsistent, incomplete, or prone to breaking changes. **Do not rely on this data for critical financial analysis or investment decisions.** Always cross-verify with official company filings and reputable financial data services.")

    income_annual_df, balance_annual_df, cashflow_annual_df = get_financial_statements_data_yf(ticker_symbol)

    st.markdown("---")
    st.markdown("#### Income Statement (Annual)")
    if not income_annual_df.empty:
        # Format the index (dates) for display
        income_annual_df.index = income_annual_df.index.strftime('%Y-%m-%d')
        st.dataframe(income_annual_df, use_container_width=True)
    else:
        st.info("No annual income statement data available.")

    st.markdown("---")
    st.markdown("#### Balance Sheet (Annual)")
    if not balance_annual_df.empty:
        balance_annual_df.index = balance_annual_df.index.strftime('%Y-%m-%d')
        st.dataframe(balance_annual_df, use_container_width=True)
    else:
        st.info("No annual balance sheet data available.")

    st.markdown("---")
    st.markdown("#### Cash Flow Statement (Annual)")
    if not cashflow_annual_df.empty:
        cashflow_annual_df.index = cashflow_annual_df.index.strftime('%Y-%m-%d')
        st.dataframe(cashflow_annual_df, use_container_width=True)
    else:
        st.info("No annual cash flow statement data available.")

    # Note: yfinance primarily provides annual data for these statements.
    # Quarterly data is often available but requires separate calls and more complex parsing if needed.
    # For a simple, robust free solution, annual is a good start.
