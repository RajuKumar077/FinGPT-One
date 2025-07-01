import streamlit as st
import requests
from textblob import TextBlob
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd
import numpy as np
import json
import time
import re

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_news_articles(query, news_api_key, total_articles=50, retries=3, initial_delay=0.5):
    """Fetches news articles from NewsAPI.com."""
    if not news_api_key or news_api_key == "YOUR_NEWSAPI_KEY":
        st.error("❌ Invalid or missing NewsAPI key in `app.py`. News articles cannot be loaded.")
        return []

    articles = []
    page = 1
    page_size = min(20, total_articles)  # NewsAPI max page size
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
            except requests.exceptions.HTTPError as http_err:
                if attempt == retries:
                    if http_err.response.status_code == 429:
                        st.error("⚠️ NewsAPI rate limit reached (100 requests/day for free tier). Try again later.")
                    elif http_err.response.status_code == 401:
                        st.error("❌ Invalid NewsAPI key. Verify `NEWS_API_KEY` in `app.py`.")
                    else:
                        st.error(f"⚠️ NewsAPI HTTP error: {http_err} (Status: {http_err.response.status_code})")
                    return []
            except requests.exceptions.ConnectionError as conn_err:
                if attempt == retries:
                    st.error(f"⚠️ NewsAPI connection error: {conn_err}. Check your internet connection.")
                    return []
            except requests.exceptions.Timeout:
                if attempt == retries:
                    st.error("⚠️ NewsAPI request timed out. Server may be slow.")
                    return []
            except json.JSONDecodeError:
                if attempt == retries:
                    st.error("⚠️ NewsAPI returned invalid data. Try again later.")
                    return []
            except Exception as e:
                if attempt == retries:
                    st.error(f"⚠️ NewsAPI unexpected error: {e}")
                    return []
    
    return articles[:total_articles]

@st.cache_data(ttl=86400, show_spinner=False)
def get_company_name_from_ticker(ticker, fmp_api_key, retries=3, initial_delay=0.5):
    """Fetches the company's long name using Financial Modeling Prep (FMP) profile endpoint."""
    if not fmp_api_key or fmp_api_key == "YOUR_FMP_KEY":
        st.warning("⚠️ FMP API key missing. Using ticker for news search.")
        return ticker

    base_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}"
    params = {"apikey": fmp_api_key}

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                time.sleep(initial_delay * (2 ** (attempt - 1)))
            response = requests.get(base_url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            if data and isinstance(data, list) and data[0]:
                return data[0].get('companyName', ticker)
            else:
                if attempt == retries:
                    st.info(f"⚠️ FMP profile data not found for {ticker}. Using ticker for news search.")
                return ticker
        except requests.exceptions.HTTPError as http_err:
            if attempt == retries:
                if http_err.response.status_code == 429:
                    st.error("⚠️ FMP rate limit reached (250 requests/day). Using ticker for news search.")
                elif http_err.response.status_code in [401, 403]:
                    st.error(f"❌ Invalid FMP API key. Verify `FMP_API_KEY` in `app.py`.")
                else:
                    st.error(f"⚠️ FMP HTTP error: {http_err}")
                return ticker
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, json.JSONDecodeError) as e:
            if attempt == retries:
                st.error(f"⚠️ FMP error: {e}. Using ticker for news search.")
                return ticker
        except Exception as e:
            if attempt == retries:
                st.error(f"⚠️ FMP unexpected error: {e}. Using ticker for news search.")
                return ticker
    return ticker

def analyze_sentiment(text):
    """Analyzes sentiment of text using TextBlob."""
    if not text or not text.strip():
        return 0.0
    return TextBlob(text).sentiment.polarity

def generate_sentiment_summary(sentiments):
    """Generates a text summary of overall sentiment."""
    if not sentiments:
        return "No sentiment data available for summary."

    sentiment_tags = ['Positive' if s > 0.05 else 'Negative' if s < -0.05 else 'Neutral' for s in sentiments]
    count = Counter(sentiment_tags)
    summary = (
        f"Out of **{len(sentiments)}** news articles analyzed:\n\n"
        f"- ✅ **{count.get('Positive', 0)}** Positive\n"
        f"- 🔻 **{count.get('Negative', 0)}** Negative\n"
        f"- ⚪ **{count.get('Neutral', 0)}** Neutral\n"
    )
    avg_sent = np.mean(sentiments)
    trend = (
        "🟢 Overall sentiment is **Positive**" if avg_sent > 0.05 else
        "🔴 Overall sentiment is **Negative**" if avg_sent < -0.05 else
        "⚪ Overall sentiment is **Neutral**"
    )
    return summary + f"\n\n**Average Sentiment Score**: `{avg_sent:.2f}`\n\n{trend}"

def create_sentiment_timeline(sentiments, parsed_dates):
    """Create a timeline chart showing sentiment over time."""
    if not sentiments or not parsed_dates or len(sentiments) != len(parsed_dates):
        return None

    df = pd.DataFrame({
        'Date': pd.to_datetime(parsed_dates),
        'Sentiment': sentiments,
        'Color': ['🟢 Positive' if s > 0.05 else '🔴 Negative' if s < -0.05 else '⚪ Neutral' for s in sentiments]
    })

    fig = px.scatter(df, x='Date', y='Sentiment', color='Color',
                     color_discrete_map={'🟢 Positive': 'limegreen', '🔴 Negative': 'tomato', '⚪ Neutral': 'lightgray'},
                     title="📈 Sentiment Timeline Analysis",
                     hover_data={'Sentiment': ':.3f'})

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.5)")
    fig.add_hline(y=0.05, line_dash="dot", line_color="rgba(50,205,50,0.3)")
    fig.add_hline(y=-0.05, line_dash="dot", line_color="rgba(255,99,71,0.3)")

    fig.update_layout(
        template='plotly_dark',
        height=400,
        xaxis_title="📅 Publication Date",
        yaxis_title="💫 Sentiment Score",
        showlegend=True,
        font=dict(family="Inter", size=12, color="#E0E0E0")
    )
    return fig

def create_sentiment_heatmap(sentiments, dates, sources):
    """Create a heatmap showing sentiment by source over time."""
    if not sentiments or not dates or not sources or len(sentiments) != len(dates) or len(sentiments) != len(sources):
        return None

    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Source': sources,
        'Sentiment': sentiments
    })

    df.dropna(subset=['Date'], inplace=True)
    if df.empty:
        return None

    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year
    df['Week_Year'] = df['Year'].astype(str) + '-W' + df['Week'].astype(str).str.zfill(2)

    heatmap_data = df.groupby(['Source', 'Week_Year'])['Sentiment'].mean().reset_index()
    if heatmap_data.empty:
        return None

    pivot_table = heatmap_data.pivot(index='Source', columns='Week_Year', values='Sentiment')
    pivot_table = pivot_table.reindex(columns=sorted(pivot_table.columns))

    fig = px.imshow(
        pivot_table.values,
        labels=dict(x="Time Period", y="News Source", color="Avg Sentiment"),
        x=pivot_table.columns,
        y=pivot_table.index,
        color_continuous_scale='RdYlGn',
        range_color=[-1, 1],
        title="🌡️ Sentiment Heatmap by Source & Time"
    )

    fig.update_layout(template='plotly_dark', height=400, font=dict(family="Inter", size=10))
    return fig

def create_word_cloud_data(articles):
    """Extracts key words for word frequency analysis."""
    all_text = " ".join(article.get('description', '') or article.get('content', '') for article in articles)
    if not all_text.strip():
        return []

    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())  # Words with 3+ letters
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
        'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with', 'this', 'have', 'been', 'each',
        'which', 'their', 'time', 'there', 'could', 'other', 'also', 'just', 'like', 'about', 'more',
        'what', 'when', 'where', 'how', 'than', 'into', 'such', 'make', 'over', 'even', 'most', 'some', 'much',
        'stock', 'market', 'company', 'investor', 'investors', 'shares', 'firm', 'group', 'new', 'report',
        'news', 'billion', 'million', 'trillion', 'analyst', 'price', 'trading', 'business', 'financial',
        'economy', 'economic', 'percent', 'said', 'say', 'says', 'year', 'week', 'day', 'month', 'quarter',
        'growth', 'earnings', 'revenue', 'profit', 'loss', 'deal', 'acquisition', 'update', 'share', 'dividend',
        'index', 'nasdaq', 'nyse', 'ceo', 'executive', 'management', 'board', 'fund', 'capital', 'investment',
        'traders', 'analysts', 'rate', 'interest', 'inflation', 'gdp', 'forecast', 'outlook', 'future',
        'potential', 'risk', 'opportunity', 'performance', 'results', 'profitability', 'guidance'
    }
    
    words = [word for word in words if word not in stop_words]
    word_freq = Counter(words).most_common(20)
    return word_freq

def create_sentiment_metrics_cards(sentiments, articles):
    """Creates metric cards for sentiment analysis."""
    col1, col2, col3, col4 = st.columns(4)

    if not sentiments:
        for col in [col1, col2, col3, col4]:
            with col:
                st.markdown("<div class='metric-card'><h3 class='card-icon'>-</h3><h4 class='card-title'>No Data</h4><h2 class='card-value'>N/A</h2></div>", unsafe_allow_html=True)
        return

    avg_sentiment = np.mean(sentiments)
    volatility = np.std(sentiments)
    positive_ratio = len([s for s in sentiments if s > 0.05]) / len(sentiments) * 100
    total_articles = len(articles)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 class="card-icon">📊</h3>
            <h4 class="card-title">Avg Sentiment</h4>
            <h2 class="card-value">{avg_sentiment:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 class="card-icon">📈</h3>
            <h4 class="card-title">Volatility</h4>
            <h2 class="card-value">{volatility:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 class="card-icon">✅</h3>
            <h4 class="card-title">Positive %</h4>
            <h2 class="card-value">{positive_ratio:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 class="card-icon">📰</h3>
            <h4 class="card-title">Total Articles</h4>
            <h2 class="card-value">{total_articles}</h2>
        </div>
        """, unsafe_allow_html=True)

def display_news_sentiment(ticker, news_api_key, fmp_api_key):
    """Main function to display news sentiment analysis for a given ticker."""
    if not ticker or not isinstance(ticker, str):
        st.error("❌ Invalid ticker symbol. Please enter a valid ticker (e.g., 'AAPL').")
        return

    ticker = ticker.strip().upper()
    st.markdown(f"<h3 class='section-title'>News Sentiment Analysis for {ticker}</h3>", unsafe_allow_html=True)

    num_articles = st.slider("Number of Articles to Analyze", min_value=10, max_value=100, value=30, step=10, key=f"num_articles_sentiment_{ticker}")

    if news_api_key == "YOUR_NEWSAPI_KEY" or fmp_api_key == "YOUR_FMP_KEY":
        st.error("❌ Missing or invalid `NEWS_API_KEY` or `FMP_API_KEY` in `app.py`. Update API keys to use news sentiment analysis.")
        return

    with st.spinner("Fetching and analyzing news..."):
        company_name = get_company_name_from_ticker(ticker, fmp_api_key)
        search_query = f'"{company_name}" OR "{ticker}"'
        articles = fetch_news_articles(search_query, news_api_key, total_articles=num_articles)

    if not articles:
        st.info("🔍 No recent news articles found. Check ticker, API key, or try a broader query.")
        return

    sentiments, parsed_dates, sources, urls, titles = [], [], [], [], []
    for article in articles:
        text = article.get('description', '') or article.get('content', '')
        if not text.strip():
            continue
        sentiment_score = analyze_sentiment(text)
        sentiments.append(sentiment_score)
        titles.append(article.get('title', 'No Title'))
        sources.append(article.get('source', {}).get('name', 'Unknown Source'))
        urls.append(article.get('url', '#'))
        try:
            date_str = article.get('publishedAt', '')
            parsed_dates.append(pd.to_datetime(date_str) if date_str else datetime.now())
        except ValueError:
            parsed_dates.append(datetime.now())

    if not sentiments:
        st.info("⚠️ No valid articles with content found for sentiment analysis.")
        return

    st.markdown("<h4 class='section-subtitle'>Sentiment Summary</h4>", unsafe_allow_html=True)
    st.markdown(generate_sentiment_summary(sentiments))
    
    create_sentiment_metrics_cards(sentiments, articles)

    st.markdown("<h4 class='section-subtitle'>Sentiment Charts</h4>", unsafe_allow_html=True)
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("<h5>Sentiment Timeline</h5>", unsafe_allow_html=True)
        timeline_fig = create_sentiment_timeline(sentiments, parsed_dates)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.info("⚠️ Not enough data to generate Sentiment Timeline.")
    
    with chart_cols[1]:
        st.markdown("<h5>Sentiment Heatmap by Source</h5>", unsafe_allow_html=True)
        heatmap_fig = create_sentiment_heatmap(sentiments, parsed_dates, sources)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.info("⚠️ Not enough data to generate Sentiment Heatmap.")

    st.markdown("---")
    st.markdown("<h4 class='section-subtitle'>Key Topics & Articles</h4>", unsafe_allow_html=True)
    word_freq = create_word_cloud_data(articles)
    if word_freq:
        st.markdown("<h5>Most Mentioned Keywords:</h5>", unsafe_allow_html=True)
        words_df = pd.DataFrame(word_freq, columns=['Keyword', 'Frequency'])
        st.dataframe(words_df, use_container_width=True)
    else:
        st.info("⚠️ No significant keywords found.")

    st.markdown("<h5>Latest News Articles:</h5>", unsafe_allow_html=True)
    for title, sentiment_score, source, url, date in zip(titles, sentiments, sources, urls, parsed_dates):
        if not title.strip() or not url.strip():
            continue
        label = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"
        color_map = {'Positive': 'limegreen', 'Negative': 'tomato', 'Neutral': 'lightgray'}
        icon_map = {'Positive': '🟢', 'Negative': '🔴', 'Neutral': '⚪'}
        date_str = date.strftime('%Y-%m-%d') if isinstance(date, datetime) else 'N/A'

        st.markdown(f"""
        <div class="news-card" style="border: 2px solid {color_map[label]};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h4 style="margin: 0; color: {color_map[label]};">{icon_map[label]} {label} Sentiment</h4>
                <span style="background: {color_map[label]}; padding: 5px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; color: white;">
                    {sentiment_score:.3f}
                </span>
            </div>
            <h3 style="margin: 10px 0;">
                <a href="{url}" target="_blank" class="news-link">
                    📰 {title}
                </a>
            </h3>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px;">
                <span style="color:#B0BEC5; font-size: 14px;">📺 {source}</span>
                <span style="color:#B0BEC5; font-size: 14px;">📅 {date_str}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.warning("""
    **Disclaimer for News Sentiment Analysis:**
    - **Data Source:** News articles from NewsAPI.com (free tier: 100 requests/day, 30-day history).
    - **Sentiment Accuracy:** TextBlob may misinterpret financial jargon or sarcasm.
    - **Limitations:** News may be delayed or biased. Not all sources are covered.
    - **Do NOT use for investment decisions.** Verify with official sources.
    """)
