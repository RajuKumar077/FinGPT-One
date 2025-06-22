import streamlit as st
import yfinance as yf # Using yfinance for core company info
import requests # Still using requests for FMP fallback
import json
import time

@st.cache_data(ttl=3600, show_spinner=False)
def get_company_overview(ticker_symbol, api_key, retries=3, initial_delay=1):
    """
    Fetches company overview data, primarily from yfinance, with FMP as a fallback
    for some direct profile fields that yfinance might not expose easily.
    """
    company_info = {}
    yf_ticker = yf.Ticker(ticker_symbol)

    try:
        # Attempt to get info from yfinance (more reliable for common fields)
        info = yf_ticker.info
        if info:
            company_info['symbol'] = info.get('symbol', ticker_symbol)
            company_info['companyName'] = info.get('longName', info.get('shortName', ticker_symbol))
            company_info['exchange'] = info.get('exchange', 'N/A')
            company_info['industry'] = info.get('industry', 'N/A')
            company_info['sector'] = info.get('sector', 'N/A') # yfinance has sector
            company_info['description'] = info.get('longBusinessSummary', 'N/A')
            company_info['ceo'] = info.get('ceo', 'N/A') # yfinance has CEO
            company_info['website'] = info.get('website', 'N/A')
            company_info['country'] = info.get('country', 'N/A')
            company_info['currency'] = info.get('currency', 'N/A')
            company_info['marketCap'] = info.get('marketCap', None) # Renamed for consistency
            company_info['peRatio'] = info.get('trailingPE', None) # yfinance uses trailingPE
            company_info['beta'] = info.get('beta', None)
            company_info['dividendYield'] = info.get('dividendYield', 0) # Already a ratio
            company_info['eps'] = info.get('trailingEps', None) # yfinance uses trailingEps
            company_info['bookValue'] = info.get('bookValue', None)
            company_info['52WeekHigh'] = info.get('fiftyTwoWeekHigh', None)
            company_info['52WeekLow'] = info.get('fiftyTwoWeekLow', None)
            company_info['priceToBookRatio'] = info.get('priceToBook', None) # yfinance uses priceToBook
            company_info['profitMargin'] = info.get('profitMargins', None) # yfinance uses profitMargins
            company_info['revenue'] = info.get('totalRevenue', None) # yfinance uses totalRevenue
            company_info['ebitda'] = info.get('ebitda', None)
            company_info['sharesOutstanding'] = info.get('sharesOutstanding', None)
            company_info['volume'] = info.get('volume', None) # Current day's volume
            company_info['price'] = info.get('currentPrice', None) # Current price
            company_info['fiscalYearEnd'] = info.get('fiscalYearEnd', 'N/A') # yfinance has fiscalYearEnd
            
    except Exception as e:
        print(f"Warning: Could not get full info from yfinance for {ticker_symbol}: {e}")
        st.warning(f"⚠️ Could not load comprehensive company overview for {ticker_symbol} from yfinance. Falling back to FMP for some fields, data might be limited.")
        
    # --- FMP Fallback for some direct fields if yfinance misses them or for consistency ---
    # This part can be simplified or removed if yfinance provides everything needed.
    # Keeping it as a backup for specific profile fields.
    base_url_fmp = f"https://financialmodelingprep.com/api/v3/profile/{ticker_symbol}"
    params_fmp = {"apikey": api_key}

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                time.sleep(initial_delay * (2 ** (attempt - 1)))
            
            response_fmp = requests.get(base_url_fmp, params=params_fmp, timeout=10)
            response_fmp.raise_for_status()
            data_fmp = response_fmp.json()

            if data_fmp and isinstance(data_fmp, list) and data_fmp[0]:
                fmp_profile = data_fmp[0]
                # Overlay FMP data, prioritizing yfinance where possible for common fields
                company_info['companyName'] = company_info.get('companyName', fmp_profile.get('companyName'))
                company_info['description'] = company_info.get('description', fmp_profile.get('description'))
                company_info['website'] = company_info.get('website', fmp_profile.get('website'))
                company_info['country'] = company_info.get('country', fmp_profile.get('country'))
                company_info['currency'] = company_info.get('currency', fmp_profile.get('currency'))
                company_info['exchange'] = company_info.get('exchange', fmp_profile.get('exchange'))
                company_info['industry'] = company_info.get('industry', fmp_profile.get('industry'))
                company_info['sector'] = company_info.get('sector', fmp_profile.get('sector'))

                # FMP specific metrics that might augment or replace yfinance where yf is less direct
                company_info['mktCap'] = company_info.get('marketCap', fmp_profile.get('mktCap')) # FMP name is mktCap
                company_info['peRatio'] = company_info.get('peRatio', fmp_profile.get('peRatio'))
                company_info['beta'] = company_info.get('beta', fmp_profile.get('beta'))
                company_info['dividendYield'] = company_info.get('dividendYield', fmp_profile.get('dividendYield'))
                company_info['eps'] = company_info.get('eps', fmp_profile.get('eps'))
                company_info['bookValue'] = company_info.get('bookValue', fmp_profile.get('bookValue'))
                company_info['priceToBookRatio'] = company_info.get('priceToBookRatio', fmp_profile.get('priceToBookRatio'))
                company_info['profitMargin'] = company_info.get('profitMargin', fmp_profile.get('profitMargin'))
                company_info['revenue'] = company_info.get('revenue', fmp_profile.get('revenue'))
                company_info['ebitda'] = company_info.get('ebitda', fmp_profile.get('ebitda'))
                company_info['sharesOutstanding'] = company_info.get('sharesOutstanding', fmp_profile.get('sharesOutstanding'))
                company_info['volume'] = company_info.get('volume', fmp_profile.get('volume'))
                company_info['price'] = company_info.get('price', fmp_profile.get('price'))
                # FMP also has 'ceo', but yfinance's is often more reliable
                company_info['ceo'] = company_info.get('ceo', fmp_profile.get('ceo'))
                company_info['fiscalYearEnd'] = company_info.get('fiscalYearEnd', fmp_profile.get('lastDiv', 'N/A')) # FMP profile doesn't have direct fiscalYearEnd, lastDiv is a weak proxy
            break # Break on successful FMP fetch
        except requests.exceptions.RequestException as req_err:
            print(f"FMP profile fallback attempt {attempt}/{retries}: Network error: {req_err}")
            if attempt == retries:
                st.warning(f"⚠️ FMP API fallback for company overview failed for {ticker_symbol}.")
        except json.JSONDecodeError as json_err:
            print(f"FMP profile fallback attempt {attempt}/{retries}: JSON Decode Error: {json_err}")
            if attempt == retries:
                st.warning(f"⚠️ FMP API fallback for company overview returned invalid data for {ticker_symbol}.")
        except Exception as e:
            print(f"FMP profile fallback attempt {attempt}/{retries}: Unexpected error: {e}")
            if attempt == retries:
                st.warning(f"⚠️ An unexpected error occurred during FMP fallback for {ticker_symbol}.")
    
    if not company_info.get('companyName'): # Final check if company name is still missing
        st.error(f"❌ Could not retrieve basic company information for {ticker_symbol}. It might be an invalid ticker or data is not publicly available.")
        return None # Return None if essential info is still missing

    return company_info


def format_value(value, is_currency=False):
    """Helper function to format large numbers for display."""
    if value is None or value == 'None' or value == '' or pd.isna(value):
        return "N/A"
    try:
        num = float(value)
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


def display_stock_summary(ticker_symbol, api_key): # api_key passed for FMP fallback
    st.subheader(f"Company Overview for {ticker_symbol.upper()}")
    
    overview = get_company_overview(ticker_symbol, api_key) # Pass api_key here

    if overview:
        st.markdown(f"**{overview.get('companyName', 'N/A')} ({overview.get('symbol', 'N/A')})**")
        st.markdown(f"**Exchange:** {overview.get('exchange', 'N/A')} | **Industry:** {overview.get('industry', 'N/A')} | **Sector:** {overview.get('sector', 'N/A')}")
        st.markdown(f"**Description:** {overview.get('description', 'N/A')}")

        st.markdown("---")
        st.subheader("Key Metrics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Market Capitalization", format_value(overview.get('mktCap'), is_currency=True))
            st.metric("PE Ratio", format_value(overview.get('peRatio')))
            st.metric("Beta", format_value(overview.get('beta')))
            st.metric("52 Week High", format_value(overview.get('52WeekHigh'), is_currency=True))
            st.metric("Current Price", format_value(overview.get('price'), is_currency=True))
            
        with col2:
            st.metric("Dividend Yield", f"{float(overview.get('dividendYield', 0)) * 100:.2f}%" if overview.get('dividendYield') is not None else "N/A")
            st.metric("EPS", format_value(overview.get('eps'), is_currency=True))
            st.metric("Book Value", format_value(overview.get('bookValue'), is_currency=True))
            st.metric("52 Week Low", format_value(overview.get('52WeekLow'), is_currency=True))
            st.metric("Price to Book Ratio", format_value(overview.get('priceToBookRatio')))
            
        with col3:
            st.metric("Profit Margin", f"{float(overview.get('profitMargin', 0)) * 100:.2f}%" if overview.get('profitMargin') is not None else "N/A")
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
        
        st.warning("⚠️ **Disclaimer for Company Overview:** This data is sourced from free APIs (primarily yfinance). While efforts are made to provide accurate information, free fundamental data can be incomplete, delayed, or subject to changes in the data source's scraping methods. For critical financial decisions, always consult official company reports and reputable financial data providers.")

    else:
        st.info(f"Could not load company overview for {ticker_symbol}. Data might be unavailable or ticker is incorrect.")

