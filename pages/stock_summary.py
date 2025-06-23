import streamlit as st
import yfinance as yf
import requests
import json
import time
import pandas as pd # Import pandas for pd.isna

@st.cache_data(ttl=3600, show_spinner=False) # Cache LLM insights for an hour
def generate_llm_insight(company_name, description, industry, sector, gemini_api_key, retries=2, initial_delay=1):
    """
    Generates a concise AI-powered insight about the company using Google's Gemini API.
    """
    if not gemini_api_key or gemini_api_key == "YOUR_GEMINI_API_KEY":
        return "AI-powered insight unavailable: GEMINI_API_KEY not set."

    # Construct a detailed prompt for the LLM
    prompt = f"""
    Based on the following company information, provide a concise, insightful summary (2-3 sentences max) highlighting its core business, market position, and potential.
    Avoid financial advice, stock recommendations, or price predictions. Focus on qualitative understanding.

    Company Name: {company_name}
    Industry: {industry}
    Sector: {sector}
    Description: {description}
    """
    
    # Define the payload for the Gemini API request
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    # Gemini API endpoint
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                print(f"Retrying Gemini API call (attempt {attempt}/{retries}). Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)

            response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload, timeout=20)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()

            # Check if candidates and content parts exist
            if result.get("candidates") and len(result["candidates"]) > 0 and \
               result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts") and \
               len(result["candidates"][0]["content"]["parts"]) > 0:
                insight = result["candidates"][0]["content"]["parts"][0]["text"]
                return f"**AI-Powered Insight:** {insight.strip()}"
            else:
                print(f"Gemini API returned no content or unexpected structure: {result}")
                if "error" in result and "message" in result["error"]:
                    if "API key not valid" in result["error"]["message"]:
                        st.error("❌ Gemini API key is invalid. Please check your GEMINI_API_KEY in app.py.")
                    elif "quota" in result["error"]["message"] or "rate limit" in result["error"]["message"]:
                        st.warning("⚠️ Gemini API rate limit hit. AI insight may be temporarily unavailable.")
                    else:
                        st.error(f"❌ Gemini API Error: {result['error']['message']}")
                if attempt == retries: # If final retry failed
                    return "AI-powered insight could not be generated due to empty or malformed response."
                continue # Retry if response is empty/malformed, not specifically an error
            
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                st.warning("⚠️ Gemini API rate limit hit. AI insight may be temporarily unavailable.")
            elif http_err.response.status_code in [400, 401, 403]:
                 st.error(f"❌ Gemini API error: {http_err.response.status_code}. Please check your GEMINI_API_KEY and API usage.")
            else:
                st.error(f"❌ Gemini API: HTTP error occurred: {http_err}. Status: {http_err.response.status_code}")
            if attempt == retries: return "AI-powered insight could not be generated (HTTP error)."

        except requests.exceptions.ConnectionError as conn_err:
            st.error(f"❌ Gemini API: Connection error occurred: {conn_err}. Please check your internet connection.")
            if attempt == retries: return "AI-powered insight could not be generated (connection error)."
        except requests.exceptions.Timeout as timeout_err:
            st.error(f"❌ Gemini API: Request timed out. The server might be slow or unresponsive. Please try again.")
            if attempt == retries: return "AI-powered insight could not be generated (timeout)."
        except json.JSONDecodeError as json_err:
            st.error(f"❌ Gemini API: Received invalid data from API. Please try again later. Error: {json_err}")
            if attempt == retries: return "AI-powered insight could not be generated (invalid response)."
        except Exception as e:
            st.error(f"❌ Gemini API: An unexpected error occurred: {e}")
            if attempt == retries: return "AI-powered insight could not be generated (unexpected error)."
    
    return "AI-powered insight could not be generated." # Fallback if all retries fail


@st.cache_data(ttl=3600, show_spinner=False) # Cache overview data for 1 hour
def get_company_overview(ticker_symbol, fmp_api_key, retries=3, initial_delay=1):
    """
    Fetches company overview data, primarily from yfinance, with FMP as a fallback
    for some direct profile fields or in case yfinance fails for certain tickers.
    """
    company_info = {}
    
    # --- Attempt to get info from yfinance first ---
    try:
        yf_ticker = yf.Ticker(ticker_symbol)
        info = yf_ticker.info # This call itself might have network issues or return empty
        
        if info:
            company_info['symbol'] = info.get('symbol', ticker_symbol)
            company_info['companyName'] = info.get('longName', info.get('shortName', ticker_symbol))
            company_info['exchange'] = info.get('exchange', 'N/A')
            company_info['industry'] = info.get('industry', 'N/A')
            company_info['sector'] = info.get('sector', 'N/A')
            company_info['description'] = info.get('longBusinessSummary', 'N/A')
            company_info['ceo'] = info.get('ceo', 'N/A')
            company_info['website'] = info.get('website', 'N/A')
            company_info['country'] = info.get('country', 'N/A')
            company_info['currency'] = info.get('currency', 'N/A')
            company_info['marketCap'] = info.get('marketCap', None) # yfinance uses marketCap
            company_info['peRatio'] = info.get('trailingPE', None) # yfinance uses trailingPE
            company_info['beta'] = info.get('beta', None)
            company_info['dividendYield'] = info.get('dividendYield', 0)
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
            company_info['fiscalYearEnd'] = info.get('fiscalYearEnd', 'N/A')
            
        else:
            print(f"yfinance.info returned empty for {ticker_symbol}. Trying FMP fallback.")

    except Exception as e:
        print(f"Warning: Error fetching yfinance.info for {ticker_symbol}: {e}. Attempting FMP fallback.")
        
    # --- FMP Fallback/Augmentation ---
    # Only try FMP if FMP_API_KEY is set and valid
    if fmp_api_key and fmp_api_key != "YOUR_FMP_KEY":
        base_url_fmp = f"https://financialmodelingprep.com/api/v3/profile/{ticker_symbol}"
        params_fmp = {"apikey": fmp_api_key}

        for attempt in range(retries + 1):
            try:
                if attempt > 0:
                    time.sleep(initial_delay * (2 ** (attempt - 1)))
                
                response_fmp = requests.get(base_url_fmp, params=params_fmp, timeout=10)
                response_fmp.raise_for_status() # Raise HTTPError for bad responses
                data_fmp = response_fmp.json()

                if data_fmp and isinstance(data_fmp, list) and data_fmp[0]:
                    fmp_profile = data_fmp[0]
                    # Overlay/augment company_info from FMP, prioritizing yfinance if it existed
                    company_info['companyName'] = company_info.get('companyName') or fmp_profile.get('companyName')
                    company_info['description'] = company_info.get('description') or fmp_profile.get('description')
                    company_info['website'] = company_info.get('website') or fmp_profile.get('website')
                    company_info['country'] = company_info.get('country') or fmp_profile.get('country')
                    company_info['currency'] = company_info.get('currency') or fmp_profile.get('currency')
                    company_info['exchange'] = company_info.get('exchange') or fmp_profile.get('exchange')
                    company_info['industry'] = company_info.get('industry') or fmp_profile.get('industry')
                    company_info['sector'] = company_info.get('sector') or fmp_profile.get('sector')

                    # FMP specific metrics that might augment or replace yfinance where yf is less direct
                    company_info['marketCap'] = company_info.get('marketCap') or fmp_profile.get('mktCap') 
                    company_info['peRatio'] = company_info.get('peRatio') or fmp_profile.get('peRatio')
                    company_info['beta'] = company_info.get('beta') or fmp_profile.get('beta')
                    company_info['dividendYield'] = company_info.get('dividendYield') or fmp_profile.get('dividendYield')
                    company_info['eps'] = company_info.get('eps') or fmp_profile.get('eps')
                    company_info['bookValue'] = company_info.get('bookValue') or fmp_profile.get('bookValue')
                    company_info['priceToBookRatio'] = company_info.get('priceToBookRatio') or fmp_profile.get('priceToBookRatio')
                    company_info['profitMargin'] = company_info.get('profitMargin') or fmp_profile.get('profitMargin')
                    company_info['revenue'] = company_info.get('revenue') or fmp_profile.get('revenue')
                    company_info['ebitda'] = company_info.get('ebitda') or fmp_profile.get('ebitda')
                    company_info['sharesOutstanding'] = company_info.get('sharesOutstanding') or fmp_profile.get('sharesOutstanding')
                    company_info['volume'] = company_info.get('volume') or fmp_profile.get('volume')
                    company_info['price'] = company_info.get('price') or fmp_profile.get('price')
                    company_info['ceo'] = company_info.get('ceo') or fmp_profile.get('ceo')
                    company_info['fiscalYearEnd'] = company_info.get('fiscalYearEnd', fmp_profile.get('lastDiv', 'N/A')) 
                    break 
                else:
                    print(f"FMP profile returned empty for {ticker_symbol}.")
                    if attempt == retries:
                        st.info(f"FMP profile data not found for {ticker_symbol}. Displaying available data from yfinance only.")
                    continue 

            except requests.exceptions.HTTPError as http_err:
                if http_err.response.status_code == 429:
                    st.warning(f"⚠️ FMP API rate limit hit for company profile. Some data might be missing.")
                elif http_err.response.status_code in [401, 403]:
                     st.error("❌ FMP API key is invalid or unauthorized for company profile. Please check your FMP_API_KEY.")
                else:
                    st.error(f"❌ FMP Company Profile: HTTP error occurred: {http_err}. Status: {http_err.response.status_code}")
                if attempt == retries: return company_info 
            except requests.exceptions.ConnectionError as conn_err:
                st.error(f"❌ FMP Company Profile: Connection error occurred: {conn_err}. Please check your internet connection.")
                if attempt == retries: return company_info
            except requests.exceptions.Timeout as timeout_err:
                st.error(f"❌ FMP Company Profile: Request timed out. The server might be slow or unresponsive. Please try again.")
                if attempt == retries: return company_info
            except json.JSONDecodeError as json_err:
                st.error(f"❌ FMP Company Profile: Received invalid data from API. Please try again later. Error: {json_err}")
                if attempt == retries: return company_info
            except Exception as e:
                st.error(f"❌ FMP Company Profile: An unexpected error occurred: {e}")
                if attempt == retries: return company_info
    
    if not company_info.get('companyName') and not company_info.get('symbol'): 
        st.error(f"❌ Could not retrieve basic company information for {ticker_symbol}. It might be an invalid ticker or data is not publicly available from either source.")
        return None 

    return company_info


def format_value(value, is_currency=False):
    """Helper function to format large numbers for display."""
    if value is None or value == 'None' or value == '' or pd.isna(value):
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
        else: # For small numbers, format to 2 decimal places
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

        # --- AI-Powered Insight Section ---
        if gemini_api_key != "YOUR_GEMINI_API_KEY":
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
        # --- End AI-Powered Insight Section ---


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
            st.metric("Dividend Yield", f"{float(dividend_yield) * 100:.2f}%" if dividend_yield is not None and pd.notna(dividend_yield) else "N/A")
            st.metric("EPS", format_value(overview.get('eps'), is_currency=True))
            st.metric("Book Value", format_value(overview.get('bookValue'), is_currency=True))
            st.metric("52 Week Low", format_value(overview.get('52WeekLow'), is_currency=True))
            st.metric("Price to Book Ratio", format_value(overview.get('priceToBookRatio')))
            
        with col3:
            profit_margin = overview.get('profitMargin')
            st.metric("Profit Margin", f"{float(profit_margin) * 100:.2f}%" if profit_margin is not None and pd.notna(profit_margin) else "N/A")
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
        ⚠️ **Disclaimer for Company Overview:**
        This data is primarily sourced from yfinance and may be augmented by Financial Modeling Prep's (FMP) free tier.
        While efforts are made to provide accurate information, free fundamental data can be:
        - Incomplete or delayed for certain tickers.
        - Subject to changes in the data source's scraping methods or API limits.
        - May not always reflect the absolute latest official figures.
        
        **For critical financial decisions, always consult official company reports and reputable paid financial data providers.**
        
        ---
        
        🧠 **Disclaimer for AI-Powered Insight:**
        The AI-generated insight is created by an experimental Large Language Model (Google Gemini).
        - **Generative Nature:** It summarizes information found in the company description and classification.
        - **Not Financial Advice:** It is **NOT** financial advice, a recommendation, or a prediction of future performance.
        - **Limitations:** AI models can sometimes generate inaccurate, biased, or nonsensical information. Always verify facts.
        
        **Do NOT use this AI insight for actual investment decisions.** It is for informational and illustrative purposes only.
        """)

    else:
        st.info(f"Could not load complete company overview for {ticker_symbol}. Data might be unavailable or ticker is incorrect. Please ensure your FMP API key is correctly set in `app.py` if you wish to use FMP as a fallback.")
