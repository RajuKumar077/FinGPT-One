import streamlit as st
import requests
from textblob import TextBlob
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
from collections import Counter
# import yfinance as yf # Not used, we will use Alpha Vantage to get company name
import pandas as pd
import numpy as np
import json # Added json import for Alpha Vantage response parsing
import time # Added time for Alpha Vantage rate limiting

# Alpha Vantage API Key (should be the same as in app.py)
ALPHA_VANTAGE_API_KEY = "9NBXSBBIYEBJHBIP"

# NEWS_API_KEY is expected to be passed from app.py or retrieved from a global config
# For modularity, it's better to pass it or have a common config file.
# We will receive it as an argument in display_news_sentiment.

# --- News Fetching and Sentiment Analysis ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_news_articles(company_name, news_api_key, total_articles=50):
    """Fetches news articles from NewsAPI.com."""
    articles = []
    page = 1
    page_size = 20
    # Limiting pages to 5 to avoid excessive API calls
    while len(articles) < total_articles and page <= 5: 
        url = (
            f"https://newsapi.org/v2/everything?q={company_name}&language=en"
            f"&sortBy=publishedAt&pageSize={min(page_size, total_articles - len(articles))}"
            f"f"&page={page}&apiKey={news_api_key}" # Fixed extra f-string prefix
        )
        try:
            response = requests.get(url)
            response.raise_for_status() # Raise an exception for HTTP errors
            page_articles = response.json().get("articles", [])
            if not page_articles:
                break
            articles.extend(page_articles)
            page += 1
        except Exception as e:
            st.warning(f"❌ Error fetching news: {e}. Please check your API key or internet connection.")
            break
    return articles[:total_articles]

def generate_sentiment_summary(sentiments):
    """Generates a text summary of overall sentiment."""
    sentiment_tags = ['Positive' if s > 0.05 else 'Negative' if s < -0.05 else 'Neutral' for s in sentiments]
    count = Counter(sentiment_tags)
    summary = (
        f"Out of **{len(sentiments)}** news articles analyzed:\n\n"
        f"- ✅ **{count.get('Positive', 0)}** are Positive\n"
        f"- 🔻 **{count.get('Negative', 0)}** are Negative\n"
        f"- ⚪ **{count.get('Neutral', 0)}** are Neutral\n"
    )
    avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0
    if avg_sent > 0.05:
        trend = "🟢 Overall sentiment is **Positive**"
    elif avg_sent < -0.05:
        trend = "🔴 Overall sentiment is **Negative**"
    else:
        trend = "⚪ Overall sentiment is **Neutral**"
    return summary + f"\n\n**Average Sentiment Score**: `{avg_sent:.2f}`\n\n{trend}"

@st.cache_data(ttl=86400, show_spinner=False)
def get_company_name_from_ticker(ticker, retries=3, initial_delay=0.5):
    """Fetches the company's long name using Alpha Vantage OVERVIEW endpoint."""
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                print(f"Retrying company name fetch for {ticker} (attempt {attempt}/{retries}). Waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            else:
                time.sleep(initial_delay) # Initial delay before first API call

            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                error_msg = data["Error Message"]
                print(f"Alpha Vantage API Overview Error (for company name) for {ticker}: {error_msg}")
                if "daily limit" in error_msg.lower() or "throttle" in error_msg.lower():
                    st.warning(f"Alpha Vantage API daily limit reached for company name fetch of {ticker}. Falling back to ticker symbol. (max 25 calls/day for free tier).")
                return ticker # Fallback to ticker symbol
            
            return data.get('Name', ticker)
        except requests.exceptions.RequestException as req_err:
            print(f"Attempt {attempt}/{retries}: Network or API request error for company name {ticker}: {req_err}. Falling back to ticker symbol.")
            if attempt == retries:
                st.warning(f"⚠️ Failed to get company name for {ticker}. Using ticker symbol for news search. Error: {req_err}")
            return ticker # Fallback to ticker symbol
        except json.JSONDecodeError as json_err:
            print(f"Attempt {attempt}/{retries}: JSON Decode Error for company name {ticker}: {json_err}. Falling back to ticker symbol. Response content starts with: {response.text[:200]}...")
            if attempt == retries:
                st.warning(f"⚠️ Received invalid data for company name {ticker}. Using ticker symbol for news search. Error: {json_err}")
            return ticker # Fallback to ticker symbol
        except Exception as e:
            print(f"Attempt {attempt}/{retries}: An unexpected error occurred while fetching company name for {ticker}: {e}. Falling back to ticker symbol.")
            if attempt == retries:
                st.warning(f"⚠️ An unexpected error occurred for company name {ticker}. Using ticker symbol for news search. Error: {e}")
            return ticker # Fallback to ticker symbol
    return ticker # Fallback return if all retries fail


# --- Sentiment Visualizations ---
def create_sentiment_timeline(sentiments, dates):
    """Create a timeline chart showing sentiment over time."""
    df = pd.DataFrame({
        'Date': dates,
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
    df = pd.DataFrame({
        'Date': [datetime.strptime(d, '%b %d, %Y') if isinstance(d, str) else d for d in dates],
        'Source': sources,
        'Sentiment': sentiments
    })

    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year
    df['Week_Year'] = df['Year'].astype(str) + '-W' + df['Week'].astype(str)

    heatmap_data = df.groupby(['Source', 'Week_Year'])['Sentiment'].mean().reset_index()

    if len(heatmap_data) > 0:
        pivot_table = heatmap_data.pivot(index='Source', columns='Week_Year', values='Sentiment')

        fig = px.imshow(pivot_table.values,
                        labels=dict(x="Time Period", y="News Source", color="Avg Sentiment"),
                        x=pivot_table.columns,
                        y=pivot_table.index,
                        color_continuous_scale='RdYlGn',
                        title="🌡️ Sentiment Heatmap by Source & Time")

        fig.update_layout(template='plotly_dark', height=400, font=dict(family="Inter", size=10))
        return fig
    return None

def create_word_cloud_data(articles):
    """Extracts key words for word frequency analysis."""
    import re
    all_text = ""
    for article in articles:
        text = article.get('description', '') or article.get('content', '')
        all_text += " " + text
    words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
    stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which',
                  'their', 'time', 'would', 'there', 'could', 'other', 'also', 'just', 'like', 'about', 'more', 'what',
                  'when', 'where', 'how', 'than', 'into', 'such', 'make', 'over', 'even', 'most', 'some', 'much'}
    words = [word for word in words if word not in stop_words]
    word_freq = Counter(words).most_common(15)
    return word_freq

def create_sentiment_metrics_cards(sentiments, articles):
    """Creates advanced metric cards for sentiment analysis."""
    col1, col2, col3, col4 = st.columns(4)

    avg_sentiment = np.mean(sentiments) if sentiments else 0
    volatility = np.std(sentiments) if sentiments else 0
    positive_ratio = len([s for s in sentiments if s > 0.05]) / len(sentiments) * 100 if sentiments else 0
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

def display_news_sentiment(ticker, news_api_key):
    """Main function to display news sentiment analysis for a given ticker."""
    st.markdown(f"<h3 class='section-title'>News Sentiment Analysis for {ticker.upper()}</h3>", unsafe_allow_html=True)

    num_articles = st.slider("Number of Articles to Analyze", min_value=10, max_value=100, value=30, step=10, key=f"num_articles_sentiment_{ticker}")

    with st.spinner("Fetching and analyzing news..."):
        # Get company name using Alpha Vantage
        company_name = get_company_name_from_ticker(ticker)
        
        # Dynamic industry keywords or broader search query could be implemented
        # The search for NewsAPI is broad, it searches for (company_name OR ticker OR keywords)
        industry_keywords = ["stock", "market", "economy", "invest", "share", "financial"] # Added 'financial'
        search_query = f'"{company_name}" OR "{ticker}"' # Prioritize exact name/ticker
        # Add keywords for broader search, ensuring unique terms
        unique_keywords = [k for k in industry_keywords if k.lower() not in company_name.lower() and k.lower() != ticker.lower()]
        if unique_keywords:
            search_query += f' OR {" OR ".join(unique_keywords)}'

        articles = fetch_news_articles(search_query, news_api_key, total_articles=num_articles)

    if not articles:
        st.info("🔍 No recent news articles found for this ticker or query. Try a different ticker or broaden your search parameters.")
        return

    sentiments, dates, sources, urls, titles = [], [], [], [], []
    parsed_dates = []

    for article in articles:
        text = article.get('description') or article.get('content') or ""
        if not text.strip():
            continue
        sentiment_score = TextBlob(text).sentiment.polarity
        sentiments.append(sentiment_score)
        titles.append(article.get('title'))
        sources.append(article.get('source', {}).get('name', 'Unknown'))
        urls.append(article.get('url'))

        try:
            date = datetime.strptime(article.get('publishedAt', ''), '%Y-%m-%dT%H:%M:%SZ')
            dates.append(date.strftime('%b %d, %Y'))
            parsed_dates.append(date)
        except:
            dates.append(article.get('publishedAt'))
            parsed_dates.append(datetime.now()) # Fallback to current time if parsing fails

    if not sentiments: # No valid articles found after parsing and filtering
        st.info("No valid articles found for sentiment analysis after processing.")
        return

    st.markdown("<h4 class='section-subtitle'>Sentiment Summary</h4>", unsafe_allow_html=True)
    st.markdown(generate_sentiment_summary(sentiments))
    
    create_sentiment_metrics_cards(sentiments, articles)

    st.markdown("<h4 class4='section-subtitle'>Sentiment Charts</h4>", unsafe_allow_html=True) # Typo fixed from h44
    
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("<h5>Sentiment Timeline</h5>", unsafe_allow_html=True)
        timeline_fig = create_sentiment_timeline(sentiments, parsed_dates)
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    with chart_cols[1]:
        st.markdown("<h5>Sentiment Heatmap by Source</h5>", unsafe_allow_html=True)
        heatmap_fig = create_sentiment_heatmap(sentiments, dates, sources)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.info("Not enough data to generate Sentiment Heatmap.")

    st.markdown("<h4 class='section-subtitle'>Key Topics & Articles</h4>", unsafe_allow_html=True)
    word_freq = create_word_cloud_data(articles)
    if word_freq:
        st.markdown("<h5>Most Mentioned Keywords:</h5>", unsafe_allow_html=True)
        words_df = pd.DataFrame(word_freq, columns=['Keyword', 'Frequency'])
        st.dataframe(words_df)
    else:
        st.info("No significant keywords found for analysis.")

    st.markdown("<h5>Latest News Articles:</h5>", unsafe_allow_html=True)
    for i in range(len(articles)):
        article = articles[i]
        if article.get('url'):
            sentiment_score = TextBlob(article.get('description') or article.get('content') or '').sentiment.polarity
            label = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"
            color_map = {'Positive': 'limegreen', 'Negative': 'tomato', 'Neutral': 'lightgray'}
            icon_map = {'Positive': '🟢', 'Negative': '🔴', 'Neutral': '⚪'}

            st.markdown(f"""
            <div class="news-card" style="border: 2px solid {color_map[label]};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h4 style="margin: 0; color: {color_map[label]};">{icon_map[label]} {label} Sentiment</h4>
                    <span style="background: {color_map[label]}; padding: 5px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; color: white;">
                        {sentiment_score:.3f}
                    </span>
                </div>
                <h3 style="margin: 10px 0;">
                    <a href="{article.get('url')}" target="_blank" class="news-link">
                        📰 {article.get('title', 'No Title')}
                    </a>
                </h3>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px;">
                    <span style="color:#B0BEC5; font-size: 14px;">📺 {article.get('source', {}).get('name', 'Unknown')}</span>
                    <span style="color:#B0BEC5; font-size: 14px;">📅 {article.get('publishedAt', 'N/A').split('T')[0]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
