import streamlit as st
import requests
from textblob import TextBlob
import plotly.express as px
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd
import numpy as np
import time
import re

st.set_page_config(page_title="News Sentiment Dashboard", layout="wide")

# -------------------------------
# API Fetching Functions
# -------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_news_articles(query, news_api_key, total_articles=50, retries=3, initial_delay=0.5):
    """Fetch news articles from NewsAPI.org."""
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
                page_articles = response.json().get("articles", [])
                if not page_articles:
                    break
                articles.extend(page_articles)
                page += 1
                break
            except requests.exceptions.RequestException as e:
                if attempt == retries:
                    st.error(f"⚠️ Error fetching news: {e}")
                    return []
    return articles[:total_articles]

@st.cache_data(ttl=86400, show_spinner=False)
def get_company_name(ticker, fmp_api_key, retries=3, initial_delay=0.5):
    """Fetch company name from FMP API."""
    if not fmp_api_key or fmp_api_key == "YOUR_FMP_KEY":
        st.warning("⚠️ FMP API key missing. Using ticker for search.")
        return ticker

    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}"
    params = {"apikey": fmp_api_key}

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                time.sleep(initial_delay * (2 ** (attempt - 1)))
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            return data[0].get('companyName', ticker) if data else ticker
        except requests.exceptions.RequestException:
            if attempt == retries:
                return ticker
    return ticker

# -------------------------------
# Sentiment Analysis Functions
# -------------------------------

def analyze_sentiment(text):
    if not text or not text.strip():
        return 0.0
    return TextBlob(text).sentiment.polarity

def summarize_sentiments(sentiments):
    if not sentiments:
        return "No sentiment data available."
    sentiment_tags = ['Positive' if s > 0.05 else 'Negative' if s < -0.05 else 'Neutral' for s in sentiments]
    count = Counter(sentiment_tags)
    avg_sent = np.mean(sentiments)
    trend = "🟢 Positive" if avg_sent > 0.05 else "🔴 Negative" if avg_sent < -0.05 else "⚪ Neutral"
    summary = (
        f"Analyzed **{len(sentiments)}** articles:\n\n"
        f"- ✅ Positive: {count.get('Positive',0)}\n"
        f"- 🔻 Negative: {count.get('Negative',0)}\n"
        f"- ⚪ Neutral: {count.get('Neutral',0)}\n\n"
        f"**Average Sentiment:** {avg_sent:.2f} → {trend}"
    )
    return summary

# -------------------------------
# Visualization Functions
# -------------------------------

def plot_timeline(sentiments, dates):
    if not sentiments or not dates:
        return None
    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Sentiment': sentiments,
        'Label': ['Positive' if s>0.05 else 'Negative' if s<-0.05 else 'Neutral' for s in sentiments]
    })
    fig = px.scatter(df, x='Date', y='Sentiment', color='Label',
                     color_discrete_map={'Positive':'limegreen','Negative':'tomato','Neutral':'lightgray'},
                     title="Sentiment Timeline")
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    return fig

def plot_heatmap(sentiments, dates, sources):
    if not sentiments or not dates or not sources:
        return None
    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Source': sources,
        'Sentiment': sentiments
    })
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year
    df['Week_Year'] = df['Year'].astype(str) + '-W' + df['Week'].astype(str).str.zfill(2)
    heatmap_data = df.groupby(['Source','Week_Year'])['Sentiment'].mean().reset_index()
    pivot = heatmap_data.pivot(index='Source', columns='Week_Year', values='Sentiment')
    fig = px.imshow(pivot.values,
                    labels=dict(x="Week-Year", y="Source", color="Avg Sentiment"),
                    x=pivot.columns, y=pivot.index,
                    color_continuous_scale='RdYlGn', range_color=[-1,1],
                    title="Sentiment Heatmap by Source")
    return fig

def extract_keywords(articles):
    text = " ".join(article.get('description','') or article.get('content','') for article in articles)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stop_words = set(["the","and","for","are","with","this","that","from","was","will","have","also","more"])
    keywords = [w for w in words if w not in stop_words]
    return Counter(keywords).most_common(20)

# -------------------------------
# Main App
# -------------------------------

def main():
    st.title("📰 News Sentiment Dashboard")
    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL):").upper().strip()
    news_api_key = st.secrets.get("NEWS_API_KEY", "YOUR_NEWSAPI_KEY")
    fmp_api_key = st.secrets.get("FMP_API_KEY", "YOUR_FMP_KEY")

    if ticker:
        num_articles = st.slider("Number of articles to analyze", 10, 100, 30, 10)
        with st.spinner("Fetching and analyzing news..."):
            company_name = get_company_name(ticker, fmp_api_key)
            search_query = f'"{company_name}" OR "{ticker}"'
            articles = fetch_news_articles(search_query, news_api_key, total_articles=num_articles)

        if not articles:
            st.info("No articles found. Try a different ticker or increase the article count.")
            return

        sentiments, dates, sources, urls, titles = [], [], [], [], []
        for art in articles:
            text = art.get('description','') or art.get('content','')
            if not text.strip():
                continue
            sentiments.append(analyze_sentiment(text))
            dates.append(pd.to_datetime(art.get('publishedAt', datetime.now())))
            sources.append(art.get('source',{}).get('name','Unknown'))
            urls.append(art.get('url','#'))
            titles.append(art.get('title','No Title'))

        st.subheader("📊 Sentiment Summary")
        st.markdown(summarize_sentiments(sentiments))

        col1, col2 = st.columns(2)
        with col1:
            timeline_fig = plot_timeline(sentiments, dates)
            if timeline_fig: st.plotly_chart(timeline_fig, use_container_width=True)
        with col2:
            heatmap_fig = plot_heatmap(sentiments, dates, sources)
            if heatmap_fig: st.plotly_chart(heatmap_fig, use_container_width=True)

        st.subheader("🔑 Key Keywords")
        keywords = extract_keywords(articles)
        if keywords:
            st.dataframe(pd.DataFrame(keywords, columns=['Keyword','Frequency']), use_container_width=True)

        st.subheader("📰 Latest Articles")
        for t,s,src,url,d in zip(titles,sentiments,sources,urls,dates):
            label = "Positive" if s>0.05 else "Negative" if s<-0.05 else "Neutral"
            color = {'Positive':'limegreen','Negative':'tomato','Neutral':'lightgray'}[label]
            st.markdown(f"**[{t}]({url})** | {label} ({s:.2f}) | {src} | {d.strftime('%Y-%m-%d')}", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
