import os
import time
import re
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import streamlit as st
from dotenv import load_dotenv
from textblob import TextBlob
import plotly.express as px

st.set_page_config(
    page_title="FinGPT One - Pro",
    layout="wide",
    page_icon="💎"
)

CSS_STYLES = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.08);
        --accent-blue: #3B82F6;
        --accent-purple: #8B5CF6;
        --accent-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --text-primary: #FFFFFF;
        --text-secondary: rgba(255, 255, 255, 0.7);
        --text-muted: rgba(255, 255, 255, 0.5);
        --bg-dark: #1a1f3a;
        --bg-card: rgba(30, 35, 58, 0.6);
    }

    .stApp {
        background: linear-gradient(135deg, #1e2139 0%, #2d3561 50%, #1e2139 100%);
        background-size: 200% 200%;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
        animation: gradientFlow 20s ease infinite;
    }

    @keyframes gradientFlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    .stApp *, .stMarkdown, .stMarkdown *, div, span, p, h1, h2, h3, h4, h5, h6, label {
        color: var(--text-primary) !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3a 0%, #151829 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    [data-testid="stSidebar"] > div {
        background: transparent !important;
        padding-top: 2rem !important;
    }

    [data-testid="stSidebar"] h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
        margin-bottom: 2rem !important;
        letter-spacing: -0.5px;
    }

    h1 {
        font-weight: 800 !important;
        font-size: 2.5rem !important;
        margin: 1.5rem 0 2rem 0 !important;
        letter-spacing: -1px;
        line-height: 1.2 !important;
    }

    h2 {
        font-weight: 700 !important;
        font-size: 1.8rem !important;
        margin: 2rem 0 1.5rem 0 !important;
        letter-spacing: -0.5px;
    }

    h3 {
        font-weight: 600 !important;
        font-size: 1.4rem !important;
        margin: 1.5rem 0 1rem 0 !important;
    }

    h4 {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }

    [data-testid="stHeaderActionElements"],
    [data-testid="stHeaderActionElements"] *,
    [data-testid="stHeaderActionElements"] button,
    [data-testid="stHeaderActionElements"] svg,
    [data-testid="stHeaderActionElements"] a {
        color: white !important;
        fill: white !important;
        stroke: white !important;
    }

    [data-testid="stHeaderActionElements"] a:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 4px !important;
    }

    .glass-card {
        background: var(--bg-card);
        border-radius: 16px;
        border: 1px solid var(--glass-border);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
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

try:
    from Components.stock_summary import display_stock_summary
    from Components.probabilistic_stock_model import display_probabilistic_models
    from Components.forecast_module import display_forecasting
    from Components.financials import display_financials
except ImportError as e:
    st.error(f"Component Import Error: {e}")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(symbol: str):
    """Fetch stock data with multi-layer failover: yfinance -> Tiingo -> Alpha Vantage -> FMP -> Synthetic."""
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
    st.sidebar.title("💎 FinGPT One")

    if "ticker_input" not in st.session_state:
        st.session_state.ticker_input = "AAPL"

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

    query = st.session_state.ticker_input

    if len(query) >= 2:
        suggestions = search_tickers(query, ALPHA_VANTAGE_API_KEY)
        if suggestions:
            st.sidebar.markdown("### 🔍 Suggestions:")
            st.sidebar.selectbox(
                "Pick a ticker:",
                options=suggestions,
                index=0 if query.upper() in [s.upper() for s in suggestions] else 0,
                key="suggestion_select",
                on_change=update_ticker_from_suggestion,
                help="Select to auto-fill the ticker",
            )
            query = st.session_state.ticker_input

    st.sidebar.markdown(f"**Selected: `{query}`**")

    page = st.sidebar.radio(
        "Dashboard Navigation",
        ["Stock Summary", "Forecasting", "Probabilistic Models", "Financials", "News Sentiment"],
    )

    if not query:
        st.error("Please enter a valid ticker symbol.")
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

    if page == "Stock Summary":
        display_stock_summary(query, hist_data, FMP_API_KEY, ALPHA_VANTAGE_API_KEY, GEMINI_API_KEY)
    elif page == "Forecasting":
        display_forecasting(hist_data, query)
    elif page == "Probabilistic Models":
        display_probabilistic_models(hist_data)
    elif page == "Financials":
        display_financials(query, TIINGO_API_KEY)

if __name__ == "__main__":
    main()
