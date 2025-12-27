import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from dotenv import load_dotenv
from textblob import TextBlob
import plotly.express as px
import time
import re
from collections import Counter
from datetime import datetime, timedelta

# --- 1. HYBRID SECRETS LOADING ---
load_dotenv()

def get_secret(key_name):
    """Safely fetch secrets from Streamlit Cloud or local .env without crashing."""
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name)

# --- 2. API KEYS ---
TIINGO_API_KEY = get_secret("TIINGO_API_KEY")
FMP_API_KEY = get_secret("FMP_API_KEY")
ALPHA_VANTAGE_API_KEY = get_secret("ALPHA_VANTAGE_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
NEWS_API_KEY = get_secret("NEWS_API_KEY")

# --- 3. COMPONENT IMPORTS ---
try:
    from Components.stock_summary import display_stock_summary
    from Components.probabilistic_stock_model import display_probabilistic_models
    from Components.forecast_module import display_forecasting
    from Components.financials import display_financials
except ImportError as e:
    st.error(f"⚠️ Component Import Error: {e}")

# --- 4. APP CONFIGURATION ---
st.set_page_config(page_title="FinGPT One - Pro", layout="wide", page_icon="📈")

# --- 5. THE DATA ENGINE ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(symbol):
    """Multi-layer failover data engine."""
    # LAYER 1: TIINGO
    if TIINGO_API_KEY and TIINGO_API_KEY != "your_tiingo_code_here":
        try:
            url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?token={TIINGO_API_KEY}"
            res = requests.get(url, timeout=5).json()
            if isinstance(res, list) and len(res) > 0:
                df = pd.DataFrame(res)
                df['date'] = pd.to_datetime(df['date']).dt.date
                df = df.rename(columns={'adjClose': 'Close', 'adjHigh': 'High', 'adjLow': 'Low', 'adjOpen': 'Open', 'adjVolume': 'Volume'})
                return df.set_index('date').sort_index(), "Tiingo Institutional"
        except: pass

    # LAYER 2: ALPHA VANTAGE
    if ALPHA_VANTAGE_API_KEY:
        try:
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
            data = requests.get(url, timeout=5).json()
            if "Time Series (Daily)" in data:
                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                df.index = pd.to_datetime(df.index).date
                df = df.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "6. volume": "Volume"}).astype(float)
                return df.sort_index(), "Alpha Vantage"
        except: pass

    # LAYER 3: YFINANCE
    try:
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(period="2y")
        if not df.empty:
            df.index = df.index.date
            return df, "Yahoo Finance"
    except: pass

    return pd.DataFrame(), None

# --- 6. NEWS SENTIMENT FUNCTIONS ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_news_articles(query, news_api_key, total_articles=50, retries=3, initial_delay=0.5):
    if not news_api_key or news_api_key == "YOUR_NEWSAPI_KEY":
        st.error("❌ Invalid or missing NewsAPI key.")
        return []

    articles = []
    page = 1
    page_size = min(20, total_articles)
    from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    while len(articles) < total_articles and page <= (total_articles // page_size) + 1:
        url = (
            f"https://newsapi.org/v2/everything?q={query}&language=en"
            f"&sortBy=relevancy&pageSize={min(page_size, total_articles - len(articles))}"
            f"&page={page}&from={from_date}&apiKey={news_api_key}"
        )
        for attempt in range(retries + 1):
            try:
                if attempt > 0:
                    time.sleep(initial_delay * (2 ** (attempt - 1)))
                response = requests.get(url, timeout=20)
                response.raise_for_status()
                json_data = response.json()
                page_articles = json_data.get("articles", [])
                if not page_articles:
                    break
                articles.extend(page_articles)
                page += 1
                break
            except Exception as e:
                if attempt == retries:
                    st.error(f"⚠️ NewsAPI error: {e}")
                    return []
    return articles[:total_articles]

@st.cache_data(ttl=86400, show_spinner=False)
def get_company_name_from_ticker(ticker, fmp_api_key, retries=3, initial_delay=0.5):
    if not fmp_api_key or fmp_api_key == "YOUR_FMP_KEY":
        st.warning("⚠️ FMP API key missing. Using ticker.")
        return ticker
    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={fmp_api_key}"
    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                time.sleep(initial_delay * (2 ** (attempt - 1)))
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
            if data and isinstance(data, list) and data[0]:
                return data[0].get('companyName', ticker)
            return ticker
        except Exception as e:
            if attempt == retries:
                st.warning(f"⚠️ FMP profile fetch failed: {e}")
                return ticker
    return ticker

def analyze_sentiment(text):
    if not text or not text.strip():
        return 0.0
    return TextBlob(text).sentiment.polarity

def generate_sentiment_summary(sentiments):
    if not sentiments:
        return "No sentiment data available."
    tags = ['Positive' if s > 0.05 else 'Negative' if s < -0.05 else 'Neutral' for s in sentiments]
    count = Counter(tags)
    avg_sent = np.mean(sentiments)
    trend = "🟢 Positive" if avg_sent > 0.05 else "🔴 Negative" if avg_sent < -0.05 else "⚪ Neutral"
    summary = (
        f"Out of **{len(sentiments)}** articles:\n"
        f"- ✅ {count.get('Positive', 0)} Positive\n"
        f"- 🔻 {count.get('Negative', 0)} Negative\n"
        f"- ⚪ {count.get('Neutral', 0)} Neutral\n\n"
        f"Average Sentiment: `{avg_sent:.2f}`\nOverall: {trend}"
    )
    return summary

def create_sentiment_timeline(sentiments, dates):
    if not sentiments or not dates or len(sentiments) != len(dates):
        return None
    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Sentiment': sentiments,
        'Color': ['🟢 Positive' if s > 0.05 else '🔴 Negative' if s < -0.05 else '⚪ Neutral' for s in sentiments]
    })
    fig = px.scatter(df, x='Date', y='Sentiment', color='Color',
                     color_discrete_map={'🟢 Positive': 'limegreen', '🔴 Negative': 'tomato', '⚪ Neutral': 'lightgray'},
                     title="📈 Sentiment Timeline", hover_data={'Sentiment': ':.3f'})
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    return fig

def create_word_cloud_data(articles):
    all_text = " ".join(article.get('description', '') or article.get('content', '') for article in articles)
    if not all_text.strip():
        return []
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    stop_words = set(["the","and","for","with","that","this","from","your","have","are","will","its","their","you"])
    words = [w for w in words if w not in stop_words]
    return Counter(words).most_common(20)

def display_news_sentiment(ticker, news_api_key, fmp_api_key):
    if not ticker:
        st.error("❌ Invalid ticker.")
        return
    ticker = ticker.strip().upper()
    st.markdown(f"<h3>News Sentiment for {ticker}</h3>", unsafe_allow_html=True)
    num_articles = st.slider("Number of articles to analyze", 10, 100, 30, step=10, key=f"news_{ticker}")
    company_name = get_company_name_from_ticker(ticker, fmp_api_key)
    query = f'"{company_name}" OR "{ticker}"'
    articles = fetch_news_articles(query, news_api_key, total_articles=num_articles)
    if not articles:
        st.info("No news found.")
        return

    sentiments, dates, sources, urls, titles = [], [], [], [], []
    for article in articles:
        text = article.get('description', '') or article.get('content', '')
        if not text.strip():
            continue
        sentiments.append(analyze_sentiment(text))
        titles.append(article.get('title', 'No Title'))
        sources.append(article.get('source', {}).get('name', 'Unknown'))
        urls.append(article.get('url', '#'))
        try:
            dates.append(pd.to_datetime(article.get('publishedAt', datetime.now())))
        except:
            dates.append(datetime.now())

    if not sentiments:
        st.info("No valid content for sentiment analysis.")
        return

    st.markdown(generate_sentiment_summary(sentiments))
    timeline_fig = create_sentiment_timeline(sentiments, dates)
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True)

    word_freq = create_word_cloud_data(articles)
    if word_freq:
        st.markdown("### Most Mentioned Keywords")
        st.dataframe(pd.DataFrame(word_freq, columns=['Keyword','Frequency']), use_container_width=True)

    st.markdown("### Latest Articles")
    for t, s, src, url, d in zip(titles, sentiments, sources, urls, dates):
        label = "Positive" if s > 0.05 else "Negative" if s < -0.05 else "Neutral"
        color_map = {'Positive':'limegreen','Negative':'tomato','Neutral':'lightgray'}
        icon_map = {'Positive':'🟢','Negative':'🔴','Neutral':'⚪'}
        date_str = d.strftime('%Y-%m-%d') if isinstance(d, datetime) else 'N/A'
        st.markdown(f"""
        <div style="border: 2px solid {color_map[label]}; padding:10px; border-radius:8px; margin-bottom:8px;">
        <h4>{icon_map[label]} {label} Sentiment ({s:.3f})</h4>
        <a href="{url}" target="_blank">📰 {t}</a><br>
        <small>📺 {src} | 📅 {date_str}</small>
        </div>
        """, unsafe_allow_html=True)

    st.warning("⚠️ Disclaimer: News sentiment is for informational purposes only. Do NOT use for investment decisions.")

# --- 7. MAIN INTERFACE ---
def main():
    st.sidebar.title("💎 FinGPT One")
    query = st.sidebar.text_input("Enter Ticker:", value="AAPL").strip().upper()
    
    # Add News Sentiment page
    page = st.sidebar.radio("Dashboard Navigation", [
        "Stock Summary", "Forecasting", "Probabilistic Models", "Financials", "News Sentiment"
    ])

    if query:
        if page != "News Sentiment":
            # Fetch stock data only if needed
            if 'data' not in st.session_state or st.session_state.get('last_ticker') != query:
                with st.spinner(f"Connecting to Markets for {query}..."):
                    df, src = fetch_stock_data(query)
                    st.session_state.data = df
                    st.session_state.source = src
                    st.session_state.last_ticker = query

            hist_data = st.session_state.data
            source = st.session_state.source

            if not hist_data.empty:
                st.sidebar.info(f"Connected via {source}")
                
                if page == "Stock Summary":
                    display_stock_summary(query, hist_data, FMP_API_KEY, ALPHA_VANTAGE_API_KEY, GEMINI_API_KEY)
                elif page == "Forecasting":
                    display_forecasting(hist_data, query)
                elif page == "Probabilistic Models":
                    display_probabilistic_models(hist_data)
                elif page == "Financials":
                    display_financials(query)
            else:
                st.error(f"Could not find data for {query}.")
        else:
            # News Sentiment Page
            display_news_sentiment(query, NEWS_API_KEY, FMP_API_KEY)

if __name__ == "__main__":
    main()
