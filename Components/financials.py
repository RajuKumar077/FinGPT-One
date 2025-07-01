import streamlit as st
import pandas as pd
import requests
import json
import time
import plotly.graph_objects as go

@st.cache_data(ttl=3600, show_spinner=False)
def get_financial_statements_data(ticker_symbol, statement_type, api_key, retries=3, initial_delay=1):
    """
    Fetches financial statements from Financial Modeling Prep (FMP) API.

    Args:
        ticker_symbol (str): The stock ticker symbol.
        statement_type (str): Type of statement ('Income Statement', 'Balance Sheet', 'Cash Flow').
        api_key (str): FMP API key.
        retries (int): Number of retry attempts.
        initial_delay (float): Initial retry delay in seconds.

    Returns:
        tuple: (annual_reports, quarterly_reports) as lists, or (None, None) if failed.
    """
    if not ticker_symbol or not isinstance(ticker_symbol, str):
        st.error("❌ Invalid ticker symbol. Please enter a valid ticker (e.g., 'AAPL').")
        return None, None

    if not api_key or api_key == "YOUR_FMP_KEY":
        st.error("❌ Missing or invalid FMP_API_KEY in `app.py`. Financial statements cannot be loaded.")
        return None, None

    function_map = {
        "Income Statement": "income-statement",
        "Balance Sheet": "balance-sheet-statement",
        "Cash Flow": "cash-flow-statement"
    }
    
    fmp_function_path = function_map.get(statement_type)
    if not fmp_function_path:
        st.error(f"❌ Invalid statement type: {statement_type}. Choose 'Income Statement', 'Balance Sheet', or 'Cash Flow'.")
        return None, None

    base_url = f"https://financialmodelingprep.com/api/v3/{fmp_function_path}/{ticker_symbol.upper()}"
    params = {"apikey": api_key}
    
    annual_reports = []
    quarterly_reports = []

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                time.sleep(initial_delay * (2 ** (attempt - 1)))

            # Fetch annual reports
            response_annual = requests.get(base_url, params={"period": "annual", **params}, timeout=20)
            response_annual.raise_for_status()
            data_annual = response_annual.json()
            if isinstance(data_annual, list):
                annual_reports = data_annual

            # Fetch quarterly reports
            response_quarterly = requests.get(base_url, params={"period": "quarter", **params}, timeout=20)
            response_quarterly.raise_for_status()
            data_quarterly = response_quarterly.json()
            if isinstance(data_quarterly, list):
                quarterly_reports = data_quarterly

            if not annual_reports and not quarterly_reports:
                if attempt == retries:
                    st.info(f"⚠️ No {statement_type} data found for {ticker_symbol.upper()} on FMP's free tier.")
                    return None, None
                continue

            return annual_reports, quarterly_reports

        except requests.exceptions.HTTPError as http_err:
            if attempt == retries:
                if http_err.response.status_code == 429:
                    st.error("⚠️ FMP API rate limit reached (250 requests/day). Try again later.")
                elif http_err.response.status_code in [401, 403]:
                    st.error("❌ Invalid FMP API key. Verify `FMP_API_KEY` in `app.py`.")
                else:
                    st.error(f"⚠️ FMP {statement_type} HTTP error: {http_err} (Status: {http_err.response.status_code})")
                return None, None
        except requests.exceptions.ConnectionError:
            if attempt == retries:
                st.error("⚠️ FMP connection error. Check your internet connection.")
                return None, None
        except requests.exceptions.Timeout:
            if attempt == retries:
                st.error("⚠️ FMP request timed out. Try again later.")
                return None, None
        except json.JSONDecodeError:
            if attempt == retries:
                st.error("⚠️ FMP returned invalid data. Check API key or try again later.")
                return None, None
        except Exception as e:
            if attempt == retries:
                st.error(f"⚠️ FMP {statement_type} unexpected error: {e}")
                return None, None
    return None, None

def display_statement_df(df, period_type, statement_type):
    """Displays a financial statement DataFrame or a message if empty."""
    if df is None or df.empty:
        st.info(f"⚠️ No {period_type} {statement_type} data available.")
        return None

    # Drop non-financial columns
    cols_to_drop = ['cik', 'finalLink', 'fillingDate', 'acceptedDate', 'period', 'link', 'reportedCurrency', 'symbol']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # Convert to numeric where possible
    df = df.apply(pd.to_numeric, errors='coerce', downcast='float')

    # Sort by date if available
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date', ascending=False)
        df = df.set_index('date')
    else:
        df.index = [f"Report {i+1}" for i in range(len(df))]

    # Format numbers for display (e.g., in millions)
    df = df.apply(lambda x: x / 1e6 if x.dtype in ['float64', 'int64'] else x)  # Convert to millions
    df = df.round(2).fillna('N/A')

    st.dataframe(df.T, use_container_width=True)  # Transpose for dates as columns
    return df

def plot_key_metrics(df, statement_type, ticker):
    """Plots key metrics for the statement over time."""
    if df is None or df.empty or 'date' not in df.columns:
        return

    key_metrics = {
        "Income Statement": ["revenue", "netIncome"],
        "Balance Sheet": ["totalAssets", "totalLiabilities"],
        "Cash Flow": ["operatingCashFlow", "freeCashFlow"]
    }
    
    metrics = key_metrics.get(statement_type, [])
    if not metrics:
        return

    fig = go.Figure()
    for metric in metrics:
        if metric in df.columns and df[metric].notnull().any():
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(df['date']),
                y=df[metric] / 1e6,  # Convert to millions
                mode='lines+markers',
                name=metric.replace('netIncome', 'Net Income').replace('totalAssets', 'Total Assets')
                           .replace('totalLiabilities', 'Total Liabilities')
                           .replace('operatingCashFlow', 'Operating Cash Flow')
                           .replace('freeCashFlow', 'Free Cash Flow')
            ))

    if fig.data:  # Only show plot if there are valid traces
        fig.update_layout(
            title=f"📈 Key {statement_type} Metrics for {ticker}",
            xaxis_title="Date",
            yaxis_title="Value (Millions USD)",
            template='plotly_dark',
            height=400,
            showlegend=True,
            font=dict(family="Inter", size=12, color="#E0E0E0")
        )
        st.plotly_chart(fig, use_container_width=True)

def display_financials(ticker_symbol, fmp_api_key):
    """Main function to display financial statements for a ticker."""
    if not ticker_symbol or not isinstance(ticker_symbol, str):
        st.error("❌ Invalid ticker symbol. Please enter a valid ticker (e.g., 'AAPL').")
        return

    ticker_symbol = ticker_symbol.strip().upper()
    st.subheader(f"Financial Statements for {ticker_symbol}")
    st.warning("""
    ⚠️ **Disclaimer for Financial Statements:**
    Data sourced from Financial Modeling Prep's free API tier.
    - **Limits:** 250 requests/day, limited historical data for some tickers.
    - **Accuracy:** Data may be incomplete or delayed.
    - **Use:** **Do NOT rely on this for investment decisions.** Verify with official filings (e.g., SEC EDGAR).
    """)

    st.markdown("---")

    for statement_type in ["Income Statement", "Balance Sheet", "Cash Flow"]:
        st.markdown(f"#### {statement_type}")
        with st.spinner(f"Loading {statement_type}..."):
            annual_reports, quarterly_reports = get_financial_statements_data(ticker_symbol, statement_type, fmp_api_key)
            
            st.markdown("##### Annual")
            annual_df = display_statement_df(pd.DataFrame(annual_reports) if annual_reports else None, "annual", statement_type)
            if annual_df is not None:
                plot_key_metrics(annual_df, statement_type, ticker_symbol)
            
            st.markdown("##### Quarterly")
            quarterly_df = display_statement_df(pd.DataFrame(quarterly_reports) if quarterly_reports else None, "quarterly", statement_type)
            if quarterly_df is not None:
                plot_key_metrics(quarterly_df, statement_type, ticker_symbol)

        st.markdown("---")
