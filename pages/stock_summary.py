import streamlit as st
import yfinance as yf
import requests
import json
import time
import pandas as pd

@st.cache_data(ttl=3600, show_spinner=False)
def generate_llm_insight(company_name, description, industry, sector, gemini_api_key, retries=2, initial_delay=1):
    """
    Generates a concise AI-powered insight about the company using Google's Gemini API.
    """
    if not gemini_api_key or gemini_api_key == "YOUR_GEMINI_API_KEY":
        return "AI-powered insight unavailable: GEMINI_API_KEY not set."

    prompt = f"""
    Based on the following company information, provide a concise, insightful summary (2-3 sentences max) highlighting its core business, market position, and potential.
    Avoid financial advice, stock recommendations, or price predictions. Focus on qualitative understanding.

    Company Name: {company_name}
    Industry: {industry}
    Sector: {sector}
    Description: {description}
    """
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                time.sleep(sleep_time)

            response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=20)
            response.raise_for_status()
            result = response.json()

            if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                insight = result["candidates"][0]["content"]["parts"][0]["text"]
                return f"**AI-Powered Insight:** {insight.strip()}"
            else:
                if "error" in result:
                    if "API key not valid" in result["error"]["message"]:
                        st.error("❌ Gemini API key is invalid. Please check your GEMINI_API_KEY in app.py.")
                    elif "quota" in result["error"]["message"] or "rate limit" in result["error"]["message"]:
                        st.warning("⚠️ Gemini API rate limit hit. AI insight unavailable.")
                    else:
                        st.error(f"❌ Gemini API Error: {result['error']['message']}")
                if attempt == retries:
                    return "AI-powered insight could not be generated."
                continue
            
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                st.warning("⚠️ Gemini API rate limit hit. AI insight unavailable.")
            elif http_err.response.status_code in [400, 401, 403]:
                st.error(f"❌ Gemini API error: {http_err.response.status_code}. Check GEMINI_API_KEY.")
            else:
                st.error(f"❌ Gemini API HTTP error: {http_err}")
            if attempt == retries:
                return "AI-powered insight could not be generated."
        except requests.exceptions.RequestException as req_err:
            st.error(f"❌ Gemini API request error: {req_err}")
            if attempt == retries:
                return "AI-powered insight could not be generated."
        except Exception as e:
            st.error(f"❌ Gemini API unexpected error: {e}")
            if attempt == retries:
                return "AI-powered insight could not be generated."
    
    return "AI-powered insight could not be generated."

@st.cache_data(ttl=3600, show_spinner=False)
def get_company_overview(ticker_symbol, fmp_api_key, retries=3, initial_delay=1):
    """
    Fetches company overview data, primarily from yfinance, with FMP as a fallback.
    """
    company_info = {}
    
    try:
        yf_ticker = yf.Ticker(ticker_symbol)
        info = yf_ticker.info
        if info:
            company_info.update({
                'symbol': info.get('symbol', ticker_symbol),
                'companyName': info.get('longName', info.get('shortName', ticker_symbol)),
                'exchange': info.get('exchange', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'description': info.get('longBusinessSummary', 'N/A'),
                'ceo': info.get('ceo', 'N/A'),
                'website': info.get('website', 'N/A'),
                'country': info.get('country', 'N/A'),
                'currency': info.get('currency', 'N/A'),
                'marketCap': info.get('marketCap'),
                'peRatio': info.get('trailingPE'),
                'beta': info.get('beta'),
                'dividendYield': info.get('dividendYield', 0),
                'eps': info.get('trailingEps'),
                'bookValue': info.get('bookValue'),
                '52WeekHigh': info.get('fiftyTwoWeekHigh'),
                '52WeekLow': info.get('fiftyTwoWeekLow'),
                'priceToBookRatio': info.get('priceToBook'),
                'profitMargin': info.get('profitMargins'),
                'revenue': info.get('totalRevenue'),
                'ebitda': info.get('ebitda'),
                'sharesOutstanding': info.get('sharesOutstanding'),
                'volume': info.get('volume'),
                'price': info.get('currentPrice'),
                'fiscalYearEnd': info.get('fiscalYearEnd', 'N/A')
            })
    except Exception as e:
        st.warning(f"⚠️ yfinance failed for {ticker_symbol}: {e}. Trying FMP.")

    if fmp_api_key and fmp_api_key != "YOUR_FMP_KEY":
        base_url_fmp = f"https://financialmodelingprep.com/api/v3/profile/{ticker_symbol}"
        params_fmp = {"apikey": fmp_api_key}
        for attempt in range(retries + 1):
            try:
                if attempt > 0:
                    time.sleep(initial_delay * (2 ** (attempt - 1)))
                response_fmp = requests.get(base_url_fmp, params=params_fmp, timeout=10)
                response_fmp.raise_for_status()
                data_fmp = response_fmp.json()
                if data_fmp and isinstance(data_fmp, list) and data_fmp[0]:
                    fmp_profile = data_fmp[0]
                    company_info.update({
                        k: company_info.get(k) or fmp_profile.get(k)
                        for k in ['companyName', 'description', 'website', 'country', 'currency',
                                  'exchange', 'industry', 'sector', 'marketCap', 'peRatio',
                                  'beta', 'dividendYield', 'eps', 'bookValue', 'priceToBookRatio',
                                  'profitMargin', 'revenue', 'ebitda', 'sharesOutstanding', 'volume',
                                  'price', 'ceo', 'fiscalYearEnd']
                    })
                    break
                else:
                    if attempt == retries:
                        st.info(f"FMP profile data not found for {ticker_symbol}.")
            except requests.exceptions.RequestException as req_err:
                if attempt == retries:
                    st.error(f"⚠️ FMP request error for {ticker_symbol}: {req_err}")
            except Exception as e:
                if attempt == retries:
                    st.error(f"⚠️ FMP unexpected error for {ticker_symbol}: {e}")

    if not company_info.get('companyName'):
        st.error(f"❌ Could not retrieve company information for {ticker_symbol}.")
        return None
    return company_info

def format_value(value, is_currency=False):
    """Helper function to format large numbers for display."""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        num = float(value)
        if not is_currency and abs(num) < 1000 and num != int(num):
            return f"{num:.2f}"
        if abs(num) >= 1e12:
            return f"${num/1e12:.2f}T" if is_currency else f"{num/1e12:.2f}T"
        elif abs(num) >= 1e9:
            return f"${num/1e9:.2f}B" if is_currency else f"{num/1e9:.2f}B"
        elif abs(num) >= 1e6:
            return f"${num/1e6:.2f}M" if is_currency else f"{num/1e6:.2f}M"
        elif abs(num) >= 1e3:
            return f"${num/1e3:.2f}K" if is_currency else f"{num/1e3:.2f}K"
        elif num == int(num):
            return f"${int(num):,}" if is_currency else f"{int(num):,}"
        else:
            return f"${num:.2f}" if is_currency else f"{num:.2f}"
    except (ValueError, TypeError):
        return str(value)

def display_stock_summary(ticker_symbol, fmp_api_key, gemini_api_key):
    st.subheader(f"Company Overview for {ticker_symbol.upper()}")
    overview = get_company_overview(ticker_symbol, fmp_api_key)
    if overview:
        st.markdown(f"**{overview.get('companyName', 'N/A')} ({overview.get('symbol', 'N/A')})**")
        st.markdown(f"**Exchange:** {overview.get('exchange', 'N/A')} | **Industry:** {overview.get('industry', 'N/A')} | **Sector:** {overview.get('sector', 'N/A')}")
        st.markdown(f"**Description:** {overview.get('description', 'N/A')}")
        st.markdown("---")

        if gemini_api_key and gemini_api_key != "YOUR_GEMINI_API_KEY":
            with st.spinner("Generating AI-powered company insight..."):
                ai_insight_text = generate_llm_insight(
                    company_name=overview.get('companyName', ticker_symbol),
                    description=overview.get('description', 'No description available.'),
                    industry=overview.get('industry', 'N/A'),
                    sector=overview.get('sector', 'N/A'),
                    gemini_api_key=gemini_api_key
                )
                st.markdown(ai_insight_text)
            st.markdown("---")

        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Market Capitalization", format_value(overview.get('marketCap'), is_currency=True))
            st.metric("PE Ratio", format_value(overview.get('peRatio')))
            st.metric("Beta", format_value(overview.get('beta')))
            st.metric("52 Week High", format_value(overview.get('52WeekHigh'), is_currency=True))
            st.metric("Current Price", format_value(overview.get('price'), is_currency=True))
        with col2:
            dividend_yield = overview.get('dividendYield')
            st.metric("Dividend Yield", f"{float(dividend_yield) * 100:.2f}%" if dividend_yield and pd.notna(dividend_yield) else "N/A")
            st.metric("EPS", format_value(overview.get('eps'), is_currency=True))
            st.metric("Book Value", format_value(overview.get('bookValue'), is_currency=True))
            st.metric("52 Week Low", format_value(overview.get('52WeekLow'), is_currency=True))
            st.metric("Price to Book Ratio", format_value(overview.get('priceToBookRatio')))
        with col3:
            profit_margin = overview.get('profitMargin')
            st.metric("Profit Margin", f"{float(profit_margin) * 100:.2f}%" if profit_margin and pd.notna(profit_margin) else "N/A")
            st.metric("Total Revenue", format_value(overview.get('revenue'), is_currency=True))
            st.metric("EBITDA", format_value(overview.get('ebitda'), is_currency=True))
            st.metric("Shares Outstanding", format_value(overview.get('sharesOutstanding')))
            st.metric("Current Volume", format_value(overview.get('volume')))

        st.markdown("---")
        st.subheader("Company Details")
        st.write(f"**CEO:** {overview.get('ceo', 'N/A')}")
        st.write(f"**Website:** [{overview.get('website', 'N/A')}]({overview.get('website', '#')})")
        st.write(f"**Country:** {overview.get('country', 'N/A')}")
        st.write(f"**Currency:** {overview.get('currency', 'N/A')}")
        st.write(f"**Fiscal Year End:** {overview.get('fiscalYearEnd', 'N/A')}")
        
        st.warning("""
        ⚠️ **Disclaimer:** Data sourced from yfinance and Financial Modeling Prep (FMP) free tiers. Information may be incomplete, delayed, or subject to API limits. For critical decisions, consult official company reports.  
        🧠 **AI Insight Disclaimer:** Generated by Google Gemini. Not financial advice. Verify facts independently.
        """)
    else:
        st.info(f"Could not load company overview for {ticker_symbol}. Check ticker or API keys.")
