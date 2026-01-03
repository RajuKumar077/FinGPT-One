import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

@st.cache_data(ttl=3600, show_spinner="🔄 Fetching institutional-grade data...")
def fetch_tiingo_fundamentals(ticker, api_key):
    """Fetch financial statements from Tiingo API"""
    if not api_key or api_key == "your_tiingo_code_here":
        st.error("❌ Missing or invalid TIINGO_API_KEY.")
        return None
        
    url = f"https://api.tiingo.com/tiingo/fundamentals/{ticker}/statements?token={api_key}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.error(f"❌ No fundamental data for {ticker}")
        else:
            st.error(f"❌ API Error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

def process_financial_data(raw_data):
    """Process and structure financial data"""
    records = []
    seen_periods = set()
    
    for period in raw_data:
        date = period.get('date')
        quarter = period.get('quarter', 0)
        year = period.get('year')
        statement_data = period.get('statementData', {})
        
        if quarter == 0:
            period_label = f"{year} Annual"
        else:
            period_label = f"{year} Q{quarter}"
        
        if period_label in seen_periods:
            continue
        seen_periods.add(period_label)
        
        row = {'Date': date, 'Period': period_label, 'Quarter': quarter, 'Year': year}
        
        # Extract all financial metrics
        for s_type in ['incomeStatement', 'balanceSheet', 'cashFlow', 'overview']:
            items = statement_data.get(s_type, [])
            for item in items:
                if 'dataCode' in item and 'value' in item:
                    row[item['dataCode']] = item['value']
        
        records.append(row)
    
    df = pd.DataFrame(records)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df.sort_values(['Year', 'Quarter'], ascending=True)
    return df

def calculate_financial_ratios(df):
    """Calculate key financial ratios"""
    metrics = {}
    latest = df.iloc[-1]
    
    # Profitability Ratios
    if 'netinc' in df.columns and 'revenue' in df.columns:
        metrics['Net Margin'] = (latest['netinc'] / latest['revenue']) * 100 if latest['revenue'] != 0 else 0
    
    if 'grossProfit' in df.columns and 'revenue' in df.columns:
        metrics['Gross Margin'] = (latest['grossProfit'] / latest['revenue']) * 100 if latest['revenue'] != 0 else 0
    
    if 'opinc' in df.columns and 'revenue' in df.columns:
        metrics['Operating Margin'] = (latest['opinc'] / latest['revenue']) * 100 if latest['revenue'] != 0 else 0
    
    # Liquidity Ratios
    if 'assetsCurrent' in df.columns and 'liabilitiesCurrent' in df.columns:
        metrics['Current Ratio'] = latest['assetsCurrent'] / latest['liabilitiesCurrent'] if latest['liabilitiesCurrent'] != 0 else 0
    
    # Leverage Ratios
    if 'debt' in df.columns and 'equity' in df.columns:
        metrics['Debt-to-Equity'] = latest['debt'] / latest['equity'] if latest['equity'] != 0 else 0
    
    # Efficiency Ratios
    if 'netinc' in df.columns and 'totalAssets' in df.columns:
        metrics['ROA'] = (latest['netinc'] / latest['totalAssets']) * 100 if latest['totalAssets'] != 0 else 0
    
    if 'netinc' in df.columns and 'equity' in df.columns:
        metrics['ROE'] = (latest['netinc'] / latest['equity']) * 100 if latest['equity'] != 0 else 0
    
    return metrics

def create_revenue_waterfall(df):
    """Create waterfall chart from Revenue to Net Income"""
    if len(df) == 0:
        return None
    
    latest = df.iloc[-1]
    period = latest['Period']
    
    # Build waterfall data
    labels = []
    values = []
    
    if 'revenue' in df.columns:
        labels.append('Revenue')
        values.append(latest['revenue'])
    
    if 'costRev' in df.columns:
        labels.append('Cost of Revenue')
        values.append(-latest['costRev'])
    
    if 'grossProfit' in df.columns:
        labels.append('Gross Profit')
        values.append(latest['grossProfit'])
    
    if 'opex' in df.columns:
        labels.append('Operating Expenses')
        values.append(-latest['opex'])
    
    if 'opinc' in df.columns:
        labels.append('Operating Income')
        values.append(latest['opinc'])
    
    if 'taxExp' in df.columns:
        labels.append('Taxes')
        values.append(-latest['taxExp'])
    
    if 'netinc' in df.columns:
        labels.append('Net Income')
        values.append(latest['netinc'])
    
    # Create waterfall
    fig = go.Figure(go.Waterfall(
        x=labels,
        y=values,
        measure=['absolute'] + ['relative'] * (len(labels) - 2) + ['total'],
        text=[f'${v/1e9:.2f}B' for v in values],
        textposition='outside',
        connector={'line': {'color': 'rgb(63, 63, 63)'}},
        decreasing={'marker': {'color': '#FF6B6B'}},
        increasing={'marker': {'color': '#4ECDC4'}},
        totals={'marker': {'color': '#45B7D1'}}
    ))
    
    fig.update_layout(
        title=f'💰 Profitability Waterfall - {period}',
        showlegend=False,
        template='plotly_dark',
        height=500,
        yaxis_title='USD',
        xaxis={'tickangle': -45}
    )
    
    return fig

def create_margin_trend(df):
    """Create margin trend analysis"""
    if 'revenue' not in df.columns or 'netinc' not in df.columns:
        return None
    
    df_calc = df.copy()
    df_calc['Gross Margin %'] = (df_calc['grossProfit'] / df_calc['revenue']) * 100 if 'grossProfit' in df.columns else None
    df_calc['Operating Margin %'] = (df_calc['opinc'] / df_calc['revenue']) * 100 if 'opinc' in df.columns else None
    df_calc['Net Margin %'] = (df_calc['netinc'] / df_calc['revenue']) * 100
    
    fig = go.Figure()
    
    if 'Gross Margin %' in df_calc.columns:
        fig.add_trace(go.Scatter(
            x=df_calc['Period'], y=df_calc['Gross Margin %'],
            name='Gross Margin', mode='lines+markers',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=8)
        ))
    
    if 'Operating Margin %' in df_calc.columns:
        fig.add_trace(go.Scatter(
            x=df_calc['Period'], y=df_calc['Operating Margin %'],
            name='Operating Margin', mode='lines+markers',
            line=dict(color='#45B7D1', width=3),
            marker=dict(size=8)
        ))
    
    fig.add_trace(go.Scatter(
        x=df_calc['Period'], y=df_calc['Net Margin %'],
        name='Net Margin', mode='lines+markers',
        line=dict(color='#96CEB4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='📊 Profitability Margins Trend',
        template='plotly_dark',
        hovermode='x unified',
        yaxis_title='Margin (%)',
        xaxis={'tickangle': -45},
        height=400
    )
    
    return fig

def create_balance_sheet_composition(df):
    """Create balance sheet composition chart"""
    if len(df) == 0:
        return None
    
    latest = df.iloc[-1]
    
    # Assets
    assets_data = []
    if 'assetsCurrent' in df.columns:
        assets_data.append({'Category': 'Current Assets', 'Value': latest['assetsCurrent']})
    if 'assetsNonCurrent' in df.columns:
        assets_data.append({'Category': 'Non-Current Assets', 'Value': latest['assetsNonCurrent']})
    
    # Liabilities & Equity
    liab_eq_data = []
    if 'liabilitiesCurrent' in df.columns:
        liab_eq_data.append({'Category': 'Current Liabilities', 'Value': latest['liabilitiesCurrent']})
    if 'liabilitiesNonCurrent' in df.columns:
        liab_eq_data.append({'Category': 'Non-Current Liabilities', 'Value': latest['liabilitiesNonCurrent']})
    if 'equity' in df.columns:
        liab_eq_data.append({'Category': 'Equity', 'Value': latest['equity']})
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'pie'}, {'type': 'pie'}]],
        subplot_titles=('Assets Composition', 'Liabilities & Equity')
    )
    
    # Assets pie
    if assets_data:
        df_assets = pd.DataFrame(assets_data)
        fig.add_trace(go.Pie(
            labels=df_assets['Category'],
            values=df_assets['Value'],
            hole=0.4,
            marker_colors=['#4ECDC4', '#45B7D1']
        ), row=1, col=1)
    
    # Liabilities & Equity pie
    if liab_eq_data:
        df_liab = pd.DataFrame(liab_eq_data)
        fig.add_trace(go.Pie(
            labels=df_liab['Category'],
            values=df_liab['Value'],
            hole=0.4,
            marker_colors=['#FF6B6B', '#FFA07A', '#96CEB4']
        ), row=1, col=2)
    
    fig.update_layout(
        title_text='⚖️ Balance Sheet Structure',
        template='plotly_dark',
        height=400,
        showlegend=True
    )
    
    return fig

def create_cash_flow_analysis(df):
    """Create cash flow analysis chart"""
    if 'ncfo' not in df.columns:
        return None
    
    fig = go.Figure()
    
    # Operating Cash Flow
    fig.add_trace(go.Bar(
        x=df['Period'], y=df['ncfo'],
        name='Operating CF',
        marker_color='#4ECDC4'
    ))
    
    # Investing Cash Flow
    if 'ncfi' in df.columns:
        fig.add_trace(go.Bar(
            x=df['Period'], y=df['ncfi'],
            name='Investing CF',
            marker_color='#45B7D1'
        ))
    
    # Financing Cash Flow
    if 'ncff' in df.columns:
        fig.add_trace(go.Bar(
            x=df['Period'], y=df['ncff'],
            name='Financing CF',
            marker_color='#96CEB4'
        ))
    
    # Free Cash Flow
    if 'freeCashFlow' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Period'], y=df['freeCashFlow'],
            name='Free Cash Flow',
            mode='lines+markers',
            line=dict(color='#FFD93D', width=3),
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title='💸 Cash Flow Analysis',
        template='plotly_dark',
        barmode='group',
        hovermode='x unified',
        yaxis_title='USD',
        xaxis={'tickangle': -45},
        height=450
    )
    
    return fig

def create_growth_metrics(df):
    """Calculate YoY and QoQ growth rates"""
    if len(df) < 2:
        return None
    
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    growth_data = []
    
    metrics_to_track = {
        'revenue': 'Revenue',
        'netinc': 'Net Income',
        'grossProfit': 'Gross Profit',
        'freeCashFlow': 'Free Cash Flow',
        'eps': 'EPS'
    }
    
    for key, label in metrics_to_track.items():
        if key in df.columns:
            try:
                prev_val = previous[key]
                curr_val = latest[key]
                if prev_val != 0:
                    growth = ((curr_val - prev_val) / abs(prev_val)) * 100
                    growth_data.append({
                        'Metric': label,
                        'Growth %': growth,
                        'Previous': prev_val,
                        'Current': curr_val
                    })
            except:
                pass
    
    if not growth_data:
        return None
    
    df_growth = pd.DataFrame(growth_data)
    
    fig = go.Figure(go.Bar(
        x=df_growth['Metric'],
        y=df_growth['Growth %'],
        text=[f"{x:+.1f}%" for x in df_growth['Growth %']],
        textposition='outside',
        marker_color=['#4ECDC4' if x > 0 else '#FF6B6B' for x in df_growth['Growth %']]
    ))
    
    fig.update_layout(
        title='📈 Period-over-Period Growth',
        template='plotly_dark',
        yaxis_title='Growth (%)',
        xaxis_title='',
        height=400,
        showlegend=False
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    
    return fig

def display_financials(ticker, api_key):
    """Main financial dashboard"""
    
    # Header
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0;'>📊 Financial Intelligence Dashboard</h1>
        <h2 style='color: #e0e0e0; margin-top: 10px;'>{ticker}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Fetch data
    raw_data = fetch_tiingo_fundamentals(ticker, api_key)
    
    if not raw_data or len(raw_data) == 0:
        st.error("❌ No financial data available")
        return
    
    # Process data
    df = process_financial_data(raw_data)
    
    if df.empty:
        st.error("❌ Could not process financial data")
        return
    
    # Calculate metrics
    ratios = calculate_financial_ratios(df)
    latest = df.iloc[-1]
    
    st.markdown("### 🎯 Executive Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if 'revenue' in df.columns and len(df) >= 2:
            rev_growth = ((latest['revenue'] - df.iloc[-2]['revenue']) / abs(df.iloc[-2]['revenue'])) * 100
            st.metric(
                "Revenue",
                f"${latest['revenue']/1e9:.2f}B",
                f"{rev_growth:+.1f}%"
            )
        elif 'revenue' in df.columns:
            st.metric("Revenue", f"${latest['revenue']/1e9:.2f}B")
    
    with col2:
        if 'netinc' in df.columns and len(df) >= 2:
            ni_growth = ((latest['netinc'] - df.iloc[-2]['netinc']) / abs(df.iloc[-2]['netinc'])) * 100
            st.metric(
                "Net Income",
                f"${latest['netinc']/1e9:.2f}B",
                f"{ni_growth:+.1f}%"
            )
        elif 'netinc' in df.columns:
            st.metric("Net Income", f"${latest['netinc']/1e9:.2f}B")
    
    with col3:
        if 'eps' in df.columns and len(df) >= 2:
            eps_growth = ((latest['eps'] - df.iloc[-2]['eps']) / abs(df.iloc[-2]['eps'])) * 100
            st.metric(
                "EPS",
                f"${latest['eps']:.2f}",
                f"{eps_growth:+.1f}%"
            )
        elif 'eps' in df.columns:
            st.metric("EPS", f"${latest['eps']:.2f}")
    
    with col4:
        if 'Net Margin' in ratios:
            st.metric("Net Margin", f"{ratios['Net Margin']:.1f}%")
    
    with col5:
        if 'freeCashFlow' in df.columns:
            st.metric("Free Cash Flow", f"${latest['freeCashFlow']/1e9:.2f}B")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Performance", "💰 Profitability", "⚖️ Balance Sheet", 
        "💸 Cash Flow", "📈 Growth", "🔢 Detailed Data"
    ])
    
    with tab1:
        st.subheader("Performance Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Revenue trend
            if 'revenue' in df.columns:
                fig_rev = go.Figure()
                fig_rev.add_trace(go.Scatter(
                    x=df['Period'], y=df['revenue']/1e9,
                    mode='lines+markers',
                    line=dict(color='#4ECDC4', width=3),
                    marker=dict(size=10),
                    fill='tonexty',
                    name='Revenue'
                ))
                fig_rev.update_layout(
                    title='Revenue Trend (Billions)',
                    template='plotly_dark',
                    height=350,
                    xaxis={'tickangle': -45}
                )
                st.plotly_chart(fig_rev, use_container_width=True)
        
        with col2:
            # Net Income trend
            if 'netinc' in df.columns:
                fig_ni = go.Figure()
                fig_ni.add_trace(go.Scatter(
                    x=df['Period'], y=df['netinc']/1e9,
                    mode='lines+markers',
                    line=dict(color='#96CEB4', width=3),
                    marker=dict(size=10),
                    fill='tonexty',
                    name='Net Income'
                ))
                fig_ni.update_layout(
                    title='Net Income Trend (Billions)',
                    template='plotly_dark',
                    height=350,
                    xaxis={'tickangle': -45}
                )
                st.plotly_chart(fig_ni, use_container_width=True)
        
        # Key ratios
        st.markdown("#### 📊 Key Financial Ratios")
        ratio_cols = st.columns(4)
        
        ratio_list = [
            ('ROE', '📈', '%'),
            ('ROA', '📊', '%'),
            ('Current Ratio', '💧', 'x'),
            ('Debt-to-Equity', '⚖️', 'x')
        ]
        
        for i, (ratio_name, icon, unit) in enumerate(ratio_list):
            if ratio_name in ratios:
                with ratio_cols[i]:
                    value = ratios[ratio_name]
                    st.markdown(f"""
                    <div style='background: #1e1e1e; padding: 20px; border-radius: 10px; text-align: center;'>
                        <h3 style='margin: 0;'>{icon}</h3>
                        <h2 style='color: #4ECDC4; margin: 10px 0;'>{value:.2f}{unit}</h2>
                        <p style='color: #888; margin: 0;'>{ratio_name}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Profitability Analysis")
        
        # Waterfall chart
        waterfall_fig = create_revenue_waterfall(df)
        if waterfall_fig:
            st.plotly_chart(waterfall_fig, use_container_width=True)
        
        # Margin trends
        margin_fig = create_margin_trend(df)
        if margin_fig:
            st.plotly_chart(margin_fig, use_container_width=True)
    
    with tab3:
        st.subheader("Balance Sheet Analysis")
        
        # Balance sheet composition
        bs_fig = create_balance_sheet_composition(df)
        if bs_fig:
            st.plotly_chart(bs_fig, use_container_width=True)
        
        # Key balance sheet metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'totalAssets' in df.columns:
                st.metric("Total Assets", f"${latest['totalAssets']/1e9:.2f}B")
        with col2:
            if 'totalLiabilities' in df.columns:
                st.metric("Total Liabilities", f"${latest['totalLiabilities']/1e9:.2f}B")
        with col3:
            if 'equity' in df.columns:
                st.metric("Total Equity", f"${latest['equity']/1e9:.2f}B")
    
    with tab4:
        st.subheader("Cash Flow Analysis")
        
        cf_fig = create_cash_flow_analysis(df)
        if cf_fig:
            st.plotly_chart(cf_fig, use_container_width=True)
    
    with tab5:
        st.subheader("Growth Metrics")
        
        growth_fig = create_growth_metrics(df)
        if growth_fig:
            st.plotly_chart(growth_fig, use_container_width=True)
    
    with tab6:
        st.subheader("Detailed Financial Statements")
        
        # Income Statement
        with st.expander("📋 Income Statement", expanded=False):
            inc_cols = ['revenue', 'costRev', 'grossProfit', 'opex', 'opinc', 'netinc', 'eps', 'epsDil']
            inc_cols = [c for c in inc_cols if c in df.columns]
            if inc_cols:
                display_df = df[['Period', 'Date'] + inc_cols].set_index('Period')
                st.dataframe(
                    display_df.style.format({col: "{:,.2f}" for col in inc_cols}),
                    use_container_width=True
                )
        
        # Balance Sheet
        with st.expander("📋 Balance Sheet", expanded=False):
            bal_cols = ['totalAssets', 'assetsCurrent', 'totalLiabilities', 'debt', 'equity', 'cashAndEq']
            bal_cols = [c for c in bal_cols if c in df.columns]
            if bal_cols:
                display_df = df[['Period', 'Date'] + bal_cols].set_index('Period')
                st.dataframe(
                    display_df.style.format({col: "{:,.2f}" for col in bal_cols}),
                    use_container_width=True
                )
        
        # Cash Flow
        with st.expander("📋 Cash Flow Statement", expanded=False):
            cf_cols = ['ncfo', 'ncfi', 'ncff', 'ncf', 'freeCashFlow', 'capex']
            cf_cols = [c for c in cf_cols if c in df.columns]
            if cf_cols:
                display_df = df[['Period', 'Date'] + cf_cols].set_index('Period')
                st.dataframe(
                    display_df.style.format({col: "{:,.2f}" for col in cf_cols}),
                    use_container_width=True
                )
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>📊 Data Source: Tiingo Fundamentals API | Latest Period: {latest['Period']} | Total Periods: {len(df)}</p>
        <p style='font-size: 12px;'>💡 All financial values in USD. Ratios calculated based on most recent period.</p>
    </div>
    """, unsafe_allow_html=True)