import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import time  # Import time for delays


def display_financials(ticker):
    """Displays company's financial statements (Income, Balance, Cash Flow) and key ratios."""
    st.markdown(f"<h3 class='section-title'>Financial Statements for {ticker.upper()}</h3>", unsafe_allow_html=True)

    view_option = st.radio("Select Period", ["Annual", "Quarterly"], horizontal=True, key=f"financial_period_{ticker}")

    stock = yf.Ticker(ticker)
    # Increased delay to 3 seconds to more aggressively mitigate 401 errors
    time.sleep(3)

    # Fetch financial data based on selected period
    @st.cache_data(ttl=86400, show_spinner=False)  # Cache financial data
    def get_data(period_type, current_ticker):  # Add current_ticker to cache key
        # Use a fresh Ticker instance inside the cached function if possible, or ensure it's passed
        # For simplicity, using the 'stock' object from outer scope, assuming it's stable.
        # If issues persist, consider creating yf.Ticker(current_ticker) inside this cached function.
        # However, to avoid passing 'stock' object (which might cause caching issues if not hashable)
        # and to ensure a fresh fetch within the cached function, let's instantiate yf.Ticker here.
        local_stock_instance = yf.Ticker(current_ticker)
        time.sleep(1)  # Add a small delay for this specific fetch inside the cached function
        if period_type == "Annual":
            income_data = local_stock_instance.financials.T
            balance_data = local_stock_instance.balance_sheet.T
            cashflow_data = local_stock_instance.cashflow.T
        else:  # Quarterly
            income_data = local_stock_instance.quarterly_financials.T
            balance_data = local_stock_instance.quarterly_balance_sheet.T
            cashflow_data = local_stock_instance.quarterly_cashflow.T

        # Ensure timezone-naive for consistency if the index is DatetimeIndex and has timezone info
        for df_data in [income_data, balance_data, cashflow_data]:
            # Only attempt to localize if the index is a DatetimeIndex and has timezone info
            if not df_data.empty and isinstance(df_data.index, pd.DatetimeIndex) and df_data.index.tz is not None:
                df_data.index = df_data.index.tz_localize(None)
        return income_data, balance_data, cashflow_data

    # Pass ticker to get_data function to ensure cache invalidation when ticker changes
    income_data, balance_data, cashflow_data = get_data(view_option, ticker)

    if income_data.empty and balance_data.empty and cashflow_data.empty:
        st.info("Financial data not available for this ticker or period.")
        return

    st.markdown("<h4>Income Statement</h4>", unsafe_allow_html=True)
    st.dataframe(income_data)

    st.markdown("<h4>Balance Sheet</h4>", unsafe_allow_html=True)
    st.dataframe(balance_data)

    st.markdown("<h4>Cash Flow</h4>", unsafe_allow_html=True)
    st.dataframe(cashflow_data)

    st.markdown("<h4>Key Financial Ratios</h4>", unsafe_allow_html=True)

    # Fetch stock info specifically for ratios, ensuring it's also within a robust block
    try:
        # Use a fresh Ticker instance here as well, with a delay
        info_stock_instance = yf.Ticker(ticker)
        time.sleep(1)  # Delay before fetching info
        info = info_stock_instance.info
    except Exception as e:
        st.error(f"Could not fetch key financial ratios for {ticker}: {e}")
        info = {}  # Provide an empty dict if info fetching fails

    ratios = {
        "Gross Margin": info.get("grossMargins"),
        "Operating Margin": info.get("operatingMargins"),
        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),
        "Debt to Equity": info.get("debtToEquity"),
        "Current Ratio": info.get("currentRatio"),
        "Free Cash Flow (TTM)": info.get("freeCashflow")
    }
    ratios_df = pd.DataFrame(ratios.items(), columns=["Metric", "Value"])
    # Format ratio values
    ratios_df['Value'] = ratios_df['Value'].apply(
        lambda x: f"{x * 100:.2f}%" if isinstance(x, (float, int)) and x <= 1 else f"{x:,.2f}" if isinstance(x, (float,
                                                                                                                 int)) else 'N/A')
    st.table(ratios_df)

    try:
        if not income_data.empty and 'Total Revenue' in income_data.columns and 'Net Income' in income_data.columns:
            st.markdown("<h4>Revenue & Net Income Trend</h4>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=income_data.index, y=income_data['Total Revenue'], name='Revenue', mode='lines+markers'))
            fig.add_trace(
                go.Scatter(x=income_data.index, y=income_data['Net Income'], name='Net Income', mode='lines+markers'))
            fig.update_layout(template='plotly_dark', title="Revenue & Net Income Over Time",
                              xaxis_title="Date", yaxis_title="Amount",
                              font=dict(family="Inter", size=12, color="#E0E0E0"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Revenue or Net Income data not available for chart.")
    except Exception as e:
        st.info(f"Could not generate Revenue & Net Income chart: {e}")

# Note: The if __name__ == "__main__": block is removed as this file is now imported as a module.
