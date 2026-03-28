import importlib
import os
import time
import re
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(
    page_title="FinGPT One - Pro",
    layout="wide",
    page_icon="💎"
)

CSS_STYLES = """
<style>
    :root {
        --font-ui: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", sans-serif;
        --glass-bg: rgba(17, 24, 39, 0.72);
        --glass-bg-strong: rgba(15, 23, 42, 0.84);
        --glass-border: rgba(148, 163, 184, 0.20);
        --text-primary: #F8FAFC;
        --text-secondary: rgba(226, 232, 240, 0.76);
        --text-muted: rgba(148, 163, 184, 0.88);
        --accent-blue: #7CC3FF;
        --accent-mint: #6EE7C8;
        --accent-gold: #F7C66C;
        --accent-rose: #F2A7B5;
        --shadow-soft: 0 30px 80px rgba(2, 6, 23, 0.38);
        --backdrop: blur(20px) saturate(150%);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(59, 130, 246, 0.14), transparent 28%),
            radial-gradient(circle at top right, rgba(16, 185, 129, 0.10), transparent 22%),
            radial-gradient(circle at bottom center, rgba(244, 114, 182, 0.06), transparent 26%),
            linear-gradient(180deg, #040814 0%, #0b1220 28%, #0f172a 100%);
        font-family: var(--font-ui);
        color: var(--text-primary);
        min-height: 100vh;
    }

    .stApp *, .stMarkdown, .stMarkdown *, div, span, p, h1, h2, h3, h4, h5, h6, label {
        color: var(--text-primary) !important;
        font-family: var(--font-ui) !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(9,14,26,0.96) 0%, rgba(13,20,36,0.94) 100%) !important;
        border-right: 1px solid var(--glass-border) !important;
        backdrop-filter: var(--backdrop);
    }

    [data-testid="stSidebar"] > div {
        background: transparent !important;
        padding-top: 1.6rem !important;
    }

    [data-testid="stSidebar"] h1 {
        font-weight: 800 !important;
        font-size: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        letter-spacing: -0.04em;
    }

    h1 {
        font-weight: 800 !important;
        font-size: 2.8rem !important;
        margin: 0 0 0.8rem 0 !important;
        letter-spacing: -0.05em;
        line-height: 1.2 !important;
    }

    h2 {
        font-weight: 700 !important;
        font-size: 1.9rem !important;
        margin: 1.4rem 0 1rem 0 !important;
        letter-spacing: -0.04em;
    }

    h3 {
        font-weight: 600 !important;
        font-size: 1.28rem !important;
        margin: 1.2rem 0 0.8rem 0 !important;
    }

    h4 {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    .block-container {
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    div[data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
        gap: 1rem !important;
    }

    div[data-testid="column"] > div {
        height: 100%;
    }

    [data-testid="stToolbar"], footer, #MainMenu {
        visibility: hidden;
    }

    .hero-panel,
    .glass-card,
    div[data-testid="stMetric"],
    [data-testid="stDataFrame"],
    [data-testid="stPlotlyChart"],
    .stAlert,
    .stTabs [data-baseweb="tab-panel"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.88) 0%, rgba(15,23,42,0.72) 100%);
        backdrop-filter: var(--backdrop);
        -webkit-backdrop-filter: var(--backdrop);
        border: 1px solid var(--glass-border);
        box-shadow: var(--shadow-soft);
        border-radius: 28px;
    }

    .hero-panel {
        padding: 2.15rem 2.3rem;
        margin-bottom: 1.4rem;
    }

    .hero-panel h1 {
        margin-bottom: 0.6rem !important;
    }

    .hero-panel p {
        margin: 0;
        max-width: 760px;
        color: var(--text-secondary) !important;
        font-size: 1.05rem;
        line-height: 1.7;
    }

    .hero-kicker,
    .section-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.72rem;
        border-radius: 999px;
        background: rgba(30, 41, 59, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.20);
        color: var(--text-secondary) !important;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.85rem;
    }

    .glass-card {
        padding: 1.25rem 1.45rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(15,23,42,0.92) 0%, rgba(17,24,39,0.82) 100%);
        backdrop-filter: var(--backdrop);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 24px;
        padding: 1.05rem 1.1rem;
        min-height: 132px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: var(--shadow-soft);
        width: 100%;
    }

    .metric-label {
        color: var(--text-secondary) !important;
        font-size: 0.88rem;
        letter-spacing: 0.02em;
    }

    .metric-value {
        font-size: 1.7rem;
        line-height: 1.2;
        margin: 0.55rem 0 0.35rem 0;
        font-weight: 700;
    }

    .metric-delta {
        color: var(--text-muted) !important;
        font-size: 0.92rem;
        line-height: 1.45;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 999px !important;
        border: 1px solid rgba(148,163,184,0.20) !important;
        background: linear-gradient(180deg, rgba(30,41,59,0.96) 0%, rgba(15,23,42,0.92) 100%) !important;
        color: #F5F7FB !important;
        padding: 0.62rem 1rem !important;
        box-shadow: 0 14px 28px rgba(2,6,23,0.28) !important;
    }

    .stTextInput input,
    .stSelectbox [data-baseweb="select"],
    .stMultiSelect [data-baseweb="select"] {
        border-radius: 18px !important;
        background: rgba(15,23,42,0.9) !important;
        border: 1px solid rgba(148,163,184,0.18) !important;
    }

    [data-testid="stMetric"] {
        padding: 1rem 1.1rem;
        min-height: 132px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
        margin-bottom: 0.75rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        background: rgba(15,23,42,0.88);
        border: 1px solid rgba(148,163,184,0.18);
        padding: 0.45rem 0.9rem;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(30,41,59,0.98) !important;
        border-color: rgba(124,195,255,0.28) !important;
    }

    .stAlert {
        padding: 1rem 1.1rem !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        padding: 1rem 1.1rem 1.2rem 1.1rem;
    }

    [data-testid="stPlotlyChart"] > div,
    [data-testid="stDataFrame"] > div {
        border-radius: 24px !important;
        overflow: hidden;
    }

    .stMarkdown p {
        line-height: 1.65;
    }
</style>
"""

st.markdown(CSS_STYLES, unsafe_allow_html=True)

load_dotenv()

def get_secret(key_name: str):
    """Load secret from Streamlit Cloud or environment variable."""
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name)

TIINGO_API_KEY = get_secret("TIINGO_API_KEY")
FMP_API_KEY = get_secret("FMP_API_KEY")
ALPHA_VANTAGE_API_KEY = get_secret("ALPHA_VANTAGE_API_KEY")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
NEWS_API_KEY = get_secret("NEWS_API_KEY")

def load_component(module_name: str, function_name: str):
    """Import a dashboard component only when the user opens that page."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except Exception as exc:
        st.error(f"Component Import Error: {module_name}.{function_name} -> {exc}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(symbol: str):
    """Fetch stock data with multi-layer failover: yfinance -> Tiingo -> Alpha Vantage -> FMP -> Synthetic."""
    import yfinance as yf

    symbol = symbol.strip().upper()

    periods_to_try = ["1y", "2y", "5y"]
    for period in periods_to_try:
        try:
            ticker_obj = yf.Ticker(symbol)
            df = ticker_obj.history(period=period, auto_adjust=True, prepost=False)
            if not df.empty and len(df) > 50:
                df.index = df.index.date
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                return df, f"Yahoo Finance (Real Market Data - {period})"
        except Exception:
            continue

    try:
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(period="max", auto_adjust=True, prepost=False)
        if not df.empty and len(df) > 50:
            df = df.tail(1256)
            df.index = df.index.date
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            return df, "Yahoo Finance (Real Market Data - Max, trimmed)"
    except Exception:
        pass

    if TIINGO_API_KEY and TIINGO_API_KEY != "your_tiingo_code_here":
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")
            url = (
                f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?"
                f"token={TIINGO_API_KEY}&startDate={start_date}&endDate={end_date}&resampleFreq=daily"
            )
            res = requests.get(url, timeout=10).json()
            if isinstance(res, list) and len(res) > 50:
                df = pd.DataFrame(res)
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df = df.rename(
                    columns={
                        "adjClose": "Close",
                        "adjHigh": "High",
                        "adjLow": "Low",
                        "adjOpen": "Open",
                        "adjVolume": "Volume",
                    }
                )
                df = df.set_index("date").sort_index()
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                return df, "Tiingo Institutional (Real Market Data)"
        except Exception:
            pass

    if ALPHA_VANTAGE_API_KEY and ALPHA_VANTAGE_API_KEY != "YOUR_ALPHA_VANTAGE_API_KEY":
        try:
            url = (
                "https://www.alphavantage.co/query?"
                f"function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
            )
            data = requests.get(url, timeout=10).json()
            ts = data.get("Time Series (Daily)")
            if ts and len(ts) > 50:
                df = pd.DataFrame.from_dict(ts, orient="index")
                df.index = pd.to_datetime(df.index).date
                df = df.rename(
                    columns={
                        "1. open": "Open",
                        "2. high": "High",
                        "3. low": "Low",
                        "4. close": "Close",
                        "5. adjusted close": "Close",
                        "6. volume": "Volume",
                    }
                ).astype(float)
                df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
                df = df.tail(1256)
                return df, "Alpha Vantage (Real Market Data)"
        except Exception:
            pass

    if FMP_API_KEY and FMP_API_KEY != "YOUR_FMP_KEY":
        try:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=1825)).strftime("%Y-%m-%d")
            url = (
                f"https://financialmodelingprep.com/api/v3/historical-price-full/"
                f"{symbol}?from={start_date}&to={end_date}&apikey={FMP_API_KEY}"
            )
            res = requests.get(url, timeout=10).json()
            if isinstance(res, dict) and "historical" in res and len(res["historical"]) > 50:
                df = pd.DataFrame(res["historical"])
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df = df.rename(
                    columns={
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    }
                )
                df = df.set_index("date").sort_index()
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                return df, "Financial Modeling Prep (Real Market Data)"
        except Exception:
            pass

    try:
        trading_days = 1256
        start_date = datetime.now() - timedelta(days=int(trading_days * 1.5))
        dates = pd.bdate_range(start=start_date, periods=trading_days)

        np.random.seed(abs(hash(symbol)) % (2**32))
        initial_price = 50.0 + (abs(hash(symbol)) % 200)

        prices = []
        current_price = initial_price
        for _ in range(trading_days):
            daily_return = np.random.normal(0.0005, 0.02)
            current_price *= 1 + daily_return
            prices.append(max(current_price, 1.0))

        df = pd.DataFrame(
            {
                "Open": prices,
                "High": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                "Low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                "Close": prices,
                "Volume": np.random.randint(1_000_000, 50_000_000, trading_days),
            },
            index=dates.date,
        )

        return df, "Sample Data (Testing Mode - Check API Keys!)"
    except Exception:
        return pd.DataFrame(), None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_news_articles(query: str, news_api_key: str, total_articles: int = 50,
                        retries: int = 3, initial_delay: float = 0.5):
    if not news_api_key or news_api_key == "YOUR_NEWSAPI_KEY":
        st.error("Invalid or missing NewsAPI key.")
        return []

    articles = []
    page = 1
    page_size = min(20, total_articles)
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    while len(articles) < total_articles and page <= (total_articles // page_size) + 1:
        url = (
            "https://newsapi.org/v2/everything?"
            f"q={query}&language=en&sortBy=relevancy&pageSize="
            f"{min(page_size, total_articles - len(articles))}"
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
                    st.error(f"NewsAPI error: {e}")
                    return []
    return articles[:total_articles]

@st.cache_data(ttl=86400, show_spinner=False)
def get_company_name_from_ticker(ticker: str, fmp_api_key: str,
                                 retries: int = 3, initial_delay: float = 0.5) -> str:
    if not fmp_api_key or fmp_api_key == "YOUR_FMP_KEY":
        st.warning("FMP API key missing. Using ticker as company name.")
        return ticker

    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={fmp_api_key}"
    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                time.sleep(initial_delay * (2 ** (attempt - 1)))
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0 and data[0]:
                return data[0].get("companyName", ticker)
            return ticker
        except Exception as e:
            if attempt == retries:
                st.warning(f"FMP profile fetch failed: {e}")
                return ticker
    return ticker

def analyze_sentiment(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    from textblob import TextBlob

    return TextBlob(text).sentiment.polarity

def generate_sentiment_summary(sentiments):
    if not sentiments:
        return "No sentiment data available."
    tags = [
        "Positive" if s > 0.05 else "Negative" if s < -0.05 else "Neutral"
        for s in sentiments
    ]
    count = Counter(tags)
    avg_sent = float(np.mean(sentiments))
    trend = (
        "🟢 Positive" if avg_sent > 0.05
        else "🔴 Negative" if avg_sent < -0.05
        else "⚪ Neutral"
    )
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
    import plotly.express as px

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Sentiment": sentiments,
            "Color": [
                "🟢 Positive" if s > 0.05 else "🔴 Negative" if s < -0.05 else "⚪ Neutral"
                for s in sentiments
            ],
        }
    )
    fig = px.scatter(
        df,
        x="Date",
        y="Sentiment",
        color="Color",
        color_discrete_map={
            "🟢 Positive": "limegreen",
            "🔴 Negative": "tomato",
            "⚪ Neutral": "lightgray",
        },
        title="📈 Sentiment Timeline",
        hover_data={"Sentiment": ":.3f"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig

def create_word_cloud_data(articles):
    all_text = " ".join(
        article.get("description", "") or article.get("content", "")
        for article in articles
    )
    if not all_text.strip():
        return []
    words = re.findall(r"\b[a-zA-Z]{3,}\b", all_text.lower())
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from",
        "your", "have", "are", "will", "its", "their", "you",
    }
    words = [w for w in words if w not in stop_words]
    return Counter(words).most_common(20)

def display_news_sentiment(ticker: str, news_api_key: str, fmp_api_key: str):
    if not ticker:
        st.error("Invalid ticker.")
        return
    ticker = ticker.strip().upper()

    st.markdown(
        f"<h3>📰 News Sentiment for {ticker}</h3>",
        unsafe_allow_html=True,
    )

    num_articles = st.slider(
        "Number of articles to analyze",
        10,
        100,
        30,
        step=10,
        key=f"news_{ticker}",
    )

    company_name = get_company_name_from_ticker(ticker, fmp_api_key)
    query = f'"{company_name}" OR "{ticker}"'
    articles = fetch_news_articles(query, news_api_key, total_articles=num_articles)

    if not articles:
        st.info("No news found.")
        return

    sentiments, dates, sources, urls, titles = [], [], [], [], []
    for article in articles:
        text = article.get("description", "") or article.get("content", "")
        if not text.strip():
            continue
        sentiments.append(analyze_sentiment(text))
        titles.append(article.get("title", "No Title"))
        sources.append(article.get("source", {}).get("name", "Unknown"))
        urls.append(article.get("url", "#"))
        try:
            dates.append(pd.to_datetime(article.get("publishedAt", datetime.now())))
        except Exception:
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
        st.markdown("### 🔑 Most Mentioned Keywords")
        st.dataframe(
            pd.DataFrame(word_freq, columns=["Keyword", "Frequency"]),
            use_container_width=True,
        )

    st.markdown("### 📋 Latest Articles")
    for t, s, src, url, d in zip(titles, sentiments, sources, urls, dates):
        label = "Positive" if s > 0.05 else "Negative" if s < -0.05 else "Neutral"
        color_map = {
            "Positive": "#10B981",
            "Negative": "#EF4444",
            "Neutral": "#6B7280",
        }
        icon_map = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}
        date_str = d.strftime("%Y-%m-%d") if isinstance(d, datetime) else "N/A"

        st.markdown(
            f"""
            <div class="glass-card" style="border: 2px solid {color_map[label]};">
                <h4 style="margin: 0 0 0.5rem 0;">
                    {icon_map[label]} {label} Sentiment ({s:.3f})
                </h4>
                <a href="{url}" target="_blank"
                   style="color: var(--accent-blue); text-decoration: none; font-weight: 600;">
                    📰 {t}
                </a><br>
                <small style="color: var(--text-secondary);">
                    📺 {src} | 📅 {date_str}
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.warning(
        "Disclaimer: News sentiment is for informational purposes only. "
        "Do NOT use for investment decisions."
    )

POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "JPM", "V",
    "UNH", "MA", "PG", "HD", "JNJ", "AVGO", "XOM", "CVX", "KO", "PEP", "COST",
    "MRK", "ABBV", "WMT", "CRM", "TMO", "ACN", "DHR", "NFLX", "ADBE", "TXN",
    "ABT", "WFC", "QCOM", "NKE", "PM", "CSCO", "INTC", "PFE", "ORCL", "AMGN",
    "DIS", "VZ", "T", "BMY", "GILD", "SBUX", "MDT", "HON", "UPS", "LMT", "NEE",
    "RTX", "UNP", "CAT", "GE", "LOW", "SPGI", "AXP", "GS", "MS", "BAC",
]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_suggestions(query: str, api_key: str,
                             retries: int = 5, initial_delay: float = 1.0) -> list:
    if not query or not api_key or api_key == "YOUR_ALPHA_VANTAGE_API_KEY":
        return []

    base_url = "https://www.alphavantage.co/query"
    params = {"function": "SYMBOL_SEARCH", "keywords": query.strip(), "apikey": api_key}

    for attempt in range(retries):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                time.sleep(sleep_time)
            else:
                time.sleep(initial_delay)

            response = requests.get(base_url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                return []
            if "Information" in data and "premium endpoint" in str(data["Information"]).lower():
                return []

            best_matches = data.get("bestMatches", [])
            suggestions = [
                f"{item.get('1. symbol', '')} - {item.get('2. name', 'Unknown')} "
                f"({item.get('4. region', 'Unknown')})"
                for item in best_matches[:10]
                if item.get("1. symbol")
            ]
            return [s for s in suggestions if s]
        except Exception:
            continue

    return []

@st.cache_data(ttl=300, show_spinner=False)
def search_tickers(search_term: str, api_key: str):
    search_term = search_term.strip().upper()
    if len(search_term) < 1:
        return []

    suggestions = fetch_ticker_suggestions(search_term, api_key)
    if suggestions:
        symbols = [s.split(" - ")[0] for s in suggestions if " - " in s]
        return symbols[:10]

    return [ticker for ticker in POPULAR_TICKERS if search_term in ticker][:10]

def main():
    st.sidebar.title("FinGPT One")

    if "ticker_input" not in st.session_state:
        st.session_state.ticker_input = "AAPL"
    if "requested_ticker" not in st.session_state:
        st.session_state.requested_ticker = ""

    def update_ticker_from_input():
        new_value = st.session_state.temp_ticker_input.upper().strip()
        if new_value:
            st.session_state.ticker_input = new_value

    def update_ticker_from_suggestion():
        st.session_state.ticker_input = st.session_state.suggestion_select.upper().strip()
        st.session_state.temp_ticker_input = st.session_state.ticker_input

    st.sidebar.text_input(
        "Enter Ticker:",
        value=st.session_state.ticker_input,
        key="temp_ticker_input",
        on_change=update_ticker_from_input,
        help="Type a ticker symbol (e.g., AAPL) for auto-suggestions",
    )

    pending_query = st.session_state.ticker_input

    if len(pending_query) >= 2:
        suggestions = search_tickers(pending_query, ALPHA_VANTAGE_API_KEY)
        if suggestions:
            st.sidebar.markdown("### 🔍 Suggestions:")
            st.sidebar.selectbox(
                "Pick a ticker:",
                options=suggestions,
                index=0 if pending_query.upper() in [s.upper() for s in suggestions] else 0,
                key="suggestion_select",
                on_change=update_ticker_from_suggestion,
                help="Select to auto-fill the ticker",
            )
            pending_query = st.session_state.ticker_input

    if st.sidebar.button("Load Dashboard", type="primary", use_container_width=True):
        st.session_state.requested_ticker = pending_query

    query = st.session_state.requested_ticker

    st.sidebar.markdown(f"**Typed: `{pending_query}`**")
    st.sidebar.markdown(f"**Loaded: `{query or 'None'}`**")

    page = st.sidebar.radio(
        "Dashboard Navigation",
        ["Stock Summary", "Forecasting", "Probabilistic Models", "Financials", "News Sentiment"],
    )

    if not query:
        st.markdown(
            """
            <section class="hero-panel">
                <div class="hero-kicker">Glass Dashboard</div>
                <h1>Market intelligence, redesigned.</h1>
                <p>
                    Enter a ticker in the sidebar and load the dashboard to explore cleaner charts,
                    upgraded models, and a lighter Apple-inspired interface.
                </p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        return

    if page == "News Sentiment":
        display_news_sentiment(query, NEWS_API_KEY, FMP_API_KEY)
        return

    if "data" not in st.session_state or st.session_state.get("last_ticker") != query:
        with st.spinner(f"Connecting to Markets for {query}..."):
            df, src = fetch_stock_data(query)
            if df.empty:
                st.error(
                    f"No data available for {query}. It may be delisted or temporarily "
                    "unavailable. Try another ticker."
                )
                st.session_state.data = pd.DataFrame()
                st.session_state.source = None
                st.session_state.last_ticker = query
            else:
                st.session_state.data = df
                st.session_state.source = src
                st.session_state.last_ticker = query
                st.success(f"Data loaded from {src}")

    hist_data = st.session_state.get("data", pd.DataFrame())
    source = st.session_state.get("source", None)

    if hist_data.empty:
        st.error(f"Could not find data for {query}. Please check the ticker symbol.")
        return

    st.markdown(
        f"""
        <div class="glass-card">
            <div class="section-kicker">Live Session</div>
            <div style="display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;">
                <div>
                    <h3 style="margin:0 0 0.3rem 0;">{query} loaded</h3>
                    <p style="margin:0;color:var(--text-secondary);">Page: {page}</p>
                </div>
                <div style="text-align:right;">
                    <p style="margin:0;color:var(--text-secondary);">Data source</p>
                    <p style="margin:0;font-weight:600;">{source or 'Unknown'}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "Stock Summary":
        display_stock_summary = load_component("Components.stock_summary", "display_stock_summary")
        if display_stock_summary:
            display_stock_summary(query, hist_data, FMP_API_KEY, ALPHA_VANTAGE_API_KEY, GEMINI_API_KEY)
    elif page == "Forecasting":
        display_forecasting = load_component("Components.forecast_module", "display_forecasting")
        if display_forecasting:
            display_forecasting(hist_data, query)
    elif page == "Probabilistic Models":
        display_probabilistic_models = load_component(
            "Components.probabilistic_stock_model",
            "display_probabilistic_models",
        )
        if display_probabilistic_models:
            display_probabilistic_models(hist_data)
    elif page == "Financials":
        display_financials = load_component("Components.financials", "display_financials")
        if display_financials:
            display_financials(query, TIINGO_API_KEY)

if __name__ == "__main__":
    main()
