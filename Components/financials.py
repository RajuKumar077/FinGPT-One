import streamlit as st
import pandas as pd
from yahooquery import Ticker
import plotly.graph_objects as go

# ---------------------------
# 1️⃣ Advanced Data Fetching
# ---------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo_financials(ticker_symbol):
    """Fetches all financial data in one go to minimize network calls."""
    try:
        t = Ticker(ticker_symbol)
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
    
    # Ensure 'asOfDate' is the index
    if 'asOfDate' in df.columns:
        df = df.rename(columns={"asOfDate": "Date"}).set_index("Date")
    
    # Remove metadata columns
    cols_to_drop = ['periodType', 'currencyCode', 'symbol']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # Convert to numeric and sort
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.sort_index(ascending=False)
    return df

def format_for_display(df):
    """Prepares DF for Streamlit table: transposes and scales units."""
    if df is None: return None
    
    # Transpose so dates are columns
    disp_df = df.T
    
    # Smart Scaling: If values are huge, convert to Millions
    def scale_val(val):
        if not isinstance(val, (int, float)) or pd.isna(val): return val
        if abs(val) >= 1e9: return f"{val/1e9:.2f}B"
        if abs(val) >= 1e6: return f"{val/1e6:.2f}M"
        return f"{val:.2f}"

    return disp_df.applymap(scale_val)

# ---------------------------
# 3️⃣ Professional Visualization
# ---------------------------
def plot_advanced_metrics(df, statement_type, ticker):
    """Creates high-quality trend charts."""
    if df is None or df.empty: return

    # Mapping Yahoo keys to readable labels
    metric_map = {
        "Income Statement": ["TotalRevenue", "NetIncomeCommonStockholders"],
        "Balance Sheet": ["TotalAssets", "TotalLiabilitiesNetMinorityInterest"],
        "Cash Flow": ["OperatingCashFlow", "FreeCashFlow"]
    }
    
    metrics = metric_map.get(statement_type, [])
    fig = go.Figure()

    for m in metrics:
        if m in df.columns:
            # We use the original numeric DF for plotting
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df[m], 
                name=m.replace("CommonStockholders", "").replace("NetMinorityInterest", ""),
                mode='lines+markers',
                hovertemplate='%{x}<br>$%{y:,.0f}'
            ))

    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# 4️⃣ Main Application Logic
# ---------------------------
def display_financials(ticker_symbol):
    """Main function to display financial statements."""
    ticker_symbol = ticker_symbol.strip().upper()
    
    # UI Header
    st.title(f"📊 Financial Analysis: {ticker_symbol}")
    
    with st.spinner(f"Retrieving global data for {ticker_symbol}..."):
        raw_data = fetch_yahoo_financials(ticker_symbol)

    if not raw_data or not isinstance(raw_data.get("price"), dict):
        st.error(f"Could not find data for {ticker_symbol}. Please check the ticker.")
        return

    # --- Company Overview Card ---
    price = raw_data["price"].get(ticker_symbol, {})
    profile = raw_data["profile"].get(ticker_symbol, {})
    
    with st.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"${price.get('regularMarketPrice', 0):.2f}")
        c2.metric("Market Cap", f"${price.get('marketCap', 0)/1e9:.2f}B")
        c3.metric("Currency", price.get('currency', 'USD'))
        st.caption(f"**Sector:** {profile.get('sector')} | **Industry:** {profile.get('industry')}")
    
    st.divider()

    # --- Statement Tabs ---
    menu = ["Income Statement", "Balance Sheet", "Cash Flow"]
    main_tabs = st.tabs(menu)

    for i, statement_type in enumerate(menu):
        with main_tabs[i]:
            st.subheader(f"{statement_type} Trends")
            
            # Sub-tabs for Annual vs Quarterly
            period_tabs = st.tabs(["📅 Annual Reports", "🕒 Quarterly Reports"])
            
            for p_idx, period in enumerate(["annual", "quarterly"]):
                with period_tabs[p_idx]:
                    raw_df = raw_data[statement_type][period]
                    clean_df = clean_financial_df(raw_df)
                    
                    if clean_df is not None:
                        # Chart
                        plot_advanced_metrics(clean_df, statement_type, ticker_symbol)
                        # Data Table
                        st.markdown("### Data Table")
                        st.dataframe(format_for_display(clean_df), use_container_width=True)
                    else:
                        st.info(f"No {period} {statement_type} data found.")

    st.success("✅ Analysis complete. All data sourced from Yahoo Finance.")
