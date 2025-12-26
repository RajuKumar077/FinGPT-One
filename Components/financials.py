import streamlit as st
import pandas as pd
from yahooquery import Ticker
import plotly.graph_objects as go

# ---------------------------
# 1️⃣ Advanced Data Fetching (Lifetime Free)
# ---------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_financials(ticker_symbol):
    """Fetches all financial data in one go using yahooquery."""
    try:
        t = Ticker(ticker_symbol)
        # Fetching all core data blocks
        data = {
            "Income Statement": {
                "annual": t.income_statement(frequency="annual"),
                "quarterly": t.income_statement(frequency="quarterly")
            },
            "Balance Sheet": {
                "annual": t.balance_sheet(frequency="annual"),
                "quarterly": t.balance_sheet(frequency="quarterly")
            },
            "Cash Flow": {
                "annual": t.cash_flow(frequency="annual"),
                "quarterly": t.cash_flow(frequency="quarterly")
            },
            "profile": t.asset_profile,
            "price": t.price
        }
        return data
    except Exception as e:
        st.error(f"❌ Yahoo Finance Error: {e}")
        return None

# ---------------------------
# 2️⃣ Smart Cleaning & Formatting
# ---------------------------
def clean_financial_df(df):
    """Cleans yahooquery multi-index dataframes for display."""
    if df is None or isinstance(df, dict) or (isinstance(df, pd.DataFrame) and df.empty):
        return None
    
    # yahooquery returns a dataframe where 'asOfDate' is a column
    if 'asOfDate' in df.columns:
        df = df.rename(columns={"asOfDate": "Date"}).set_index("Date")
    
    # Remove metadata columns that clutter the table
    cols_to_drop = ['periodType', 'currencyCode', 'symbol']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # Convert to numeric and sort by newest date first
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.sort_index(ascending=False)
    return df

def format_for_display(df):
    """Prepares DF for Streamlit: transposes and scales units to B/M."""
    if df is None: return None
    
    # Transpose so dates are columns (FMP style)
    disp_df = df.T
    
    def scale_val(val):
        if not isinstance(val, (int, float)) or pd.isna(val): return val
        if abs(val) >= 1e9: return f"{val/1e9:.2f}B"
        if abs(val) >= 1e6: return f"{val/1e6:.2f}M"
        return f"{val:,.2f}"

    return disp_df.applymap(scale_val)

# ---------------------------
# 3️⃣ Professional Visualization
# ---------------------------
def plot_advanced_metrics(df, statement_type):
    """Creates trend charts for key items."""
    if df is None or df.empty: return

    metric_map = {
        "Income Statement": ["TotalRevenue", "NetIncomeCommonStockholders"],
        "Balance Sheet": ["TotalAssets", "TotalLiabilitiesNetMinorityInterest"],
        "Cash Flow": ["OperatingCashFlow", "FreeCashFlow"]
    }
    
    metrics = metric_map.get(statement_type, [])
    fig = go.Figure()

    for m in metrics:
        if m in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df[m], 
                name=m.replace("CommonStockholders", "").replace("NetMinorityInterest", ""),
                mode='lines+markers'
            ))

    fig.update_layout(
        template="plotly_dark", height=250,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.2),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# 4️⃣ Main Application UI
# ---------------------------
def display_financials(ticker_symbol):
    """Main UI component to be called from app.py."""
    ticker_symbol = ticker_symbol.strip().upper()
    
    with st.spinner(f"Sourcing free lifetime data for {ticker_symbol}..."):
        raw_data = fetch_yahoo_financials(ticker_symbol)

    if not raw_data:
        st.error("No data found.")
        return

    # Overview Section (Replaces FMP Profile)
    price_info = raw_data["price"].get(ticker_symbol, {})
    prof_info = raw_data["profile"].get(ticker_symbol, {})
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Price", f"${price_info.get('regularMarketPrice', 0):.2f}")
    col2.metric("Market Cap", f"${price_info.get('marketCap', 0)/1e9:.2f}B")
    col3.metric("Sector", prof_info.get('sector', 'N/A'))
    
    st.divider()

    # Statement Navigation
    menu = ["Income Statement", "Balance Sheet", "Cash Flow"]
    tabs = st.tabs(menu)

    for i, statement_type in enumerate(menu):
        with tabs[i]:
            period_choice = st.radio(f"Period ({statement_type})", ["Annual", "Quarterly"], horizontal=True, key=f"rad_{i}")
            
            period_key = period_choice.lower()
            raw_df = raw_data[statement_type][period_key]
            clean_df = clean_financial_df(raw_df)
            
            if clean_df is not None:
                plot_advanced_metrics(clean_df, statement_type)
                st.markdown("**Detailed Report**")
                st.dataframe(format_for_display(clean_df), use_container_width=True)
            else:
                st.error("Data not available for this period.")
