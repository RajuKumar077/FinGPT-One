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

# --- News Fetching and Sentiment Analysis ---
@st.cache_data(ttl=3600, show_spinner=False) # Cache news articles for an hour
def fetch_news_articles(query, news_api_key, total_articles=50, retries=3, initial_delay=0.5):
    """Fetches news articles from NewsAPI.com."""
    if not news_api_key or news_api_key == "874ba654bdcd4aa7b68f7367a907cc2f":
        st.error("❌ NewsAPI_KEY is not set. News articles cannot be loaded.")
        return []

    articles = []
    page = 1
    page_size = 20 # Max articles per page for NewsAPI
    
    # Fetch articles published in the last 30 days
    from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    while len(articles) < total_articles and page <= (total_articles / page_size) + 1: # Limit pages to avoid excessive calls
        url = (
            f"https://newsapi.org/v2/everything?q={query}&language=en"
            f"&sortBy=relevancy&pageSize={min(page_size, total_articles - len(articles))}"
            f"&page={page}&from={from_date}&apiKey={news_api_key}"
        )
        
        for attempt in range(retries + 1):
            try:
                if attempt > 0:
                    time.sleep(initial_delay * (2 ** (attempt - 1)))
                    print(f"Retrying NewsAPI fetch (attempt {attempt}/{retries}). Waiting {initial_delay * (2 ** (attempt - 1)):.1f} seconds...")
                
                response = requests.get(url, timeout=15)
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
                json_data = response.json()
                page_articles = json_data.get("articles", [])
                
                if not page_articles:
                    break # No more articles or query returned nothing
                
                articles.extend(page_articles)
                page += 1
                break # Break from retry loop on success
            except requests.exceptions.HTTPError as http_err:
                if http_err.response.status_code == 429: # Rate limit
                    st.warning(f"⚠️ NewsAPI rate limit hit. Some news might not be loaded. Please wait and try again.")
                elif http_err.response.status_code == 401: # Unauthorized
                    st.error("❌ NewsAPI key is invalid or unauthorized. Please check your NEWS_API_KEY in app.py.")
                else:
                    st.error(f"❌ NewsAPI HTTP error occurred: {http_err}. Status: {http_err.response.status_code}")
                if attempt == retries: return [] # Final failure
            except requests.exceptions.ConnectionError as conn_err:
                st.error(f"❌ NewsAPI Connection error: {conn_err}. Check internet connection.")
                if attempt == retries: return []
            except requests.exceptions.Timeout as timeout_err:
                st.error(f"❌ NewsAPI Request timed out: {timeout_err}. Server might be slow.")
                if attempt == retries: return []
            except json.JSONDecodeError as json_err:
                st.error(f"❌ NewsAPI: Received invalid data. Error: {json_err}")
                if attempt == retries: return []
            except Exception as e:
                st.error(f"❌ NewsAPI: An unexpected error occurred: {e}")
                if attempt == retries: return []
    return articles[:total_articles] # Ensure we don't return more than requested


@st.cache_data(ttl=86400, show_spinner=False) # Cache company name for a day
def get_company_name_from_ticker(ticker, fmp_api_key, retries=3, initial_delay=0.5):
    """Fetches the company's long name using Financial Modeling Prep (FMP) profile endpoint."""
    if not fmp_api_key or fmp_api_key == "YOUR_FMP_KEY":
        return ticker # Fallback to ticker if key is not set

    base_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}"
    params = {"apikey": fmp_api_key}

    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                sleep_time = initial_delay * (2 ** (attempt - 1))
                time.sleep(sleep_time)

            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data and isinstance(data, list) and data[0]:
                return data[0].get('companyName', ticker)
            else:
                if attempt == retries:
                    st.info(f"FMP profile data not found for {ticker}. Using ticker symbol for news search.")
                return ticker # Fallback to ticker symbol
        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code == 429:
                st.warning(f"⚠️ FMP API rate limit hit for company name lookup. Using ticker symbol for news search.")
            elif http_err.response.status_code in [401, 403]:
                st.error("❌ FMP API key is invalid or unauthorized for company name lookup. Please check your FMP_API_KEY.")
            else:
                st.error(f"❌ FMP Company Name Lookup: HTTP error occurred: {http_err}. Using ticker symbol for news search.")
            if attempt == retries: return ticker
        except requests.exceptions.ConnectionError as conn_err:
            st.error(f"❌ FMP Company Name Lookup: Connection error: {conn_err}. Using ticker symbol for news search.")
            if attempt == retries: return ticker
        except requests.exceptions.Timeout as timeout_err:
            st.error(f"❌ FMP Company Name Lookup: Request timed out. Using ticker symbol for news search.")
            if attempt == retries: return ticker
        except json.JSONDecodeError as json_err:
            st.error(f"❌ FMP Company Name Lookup: Invalid data. Error: {json_err}. Using ticker symbol for news search.")
            if attempt == retries: return ticker
        except Exception as e:
            st.error(f"❌ FMP Company Name Lookup: Unexpected error: {e}. Using ticker symbol for news search.")
            if attempt == retries: return ticker
    return ticker # Fallback return if all retries fail


def analyze_sentiment(text):
    """Analyzes sentiment of text using TextBlob."""
    if not text:
        return 0.0 # Neutral sentiment for empty text
    return TextBlob(text).sentiment.polarity

def generate_sentiment_summary(sentiments):
    """Generates a text summary of overall sentiment."""
    if not sentiments:
        return "No sentiment data available for summary."

    sentiment_tags = ['Positive' if s > 0.05 else 'Negative' if s < -0.05 else 'Neutral' for s in sentiments]
    count = Counter(sentiment_tags)
    summary = (
        f"Out of **{len(sentiments)}** news articles analyzed:\n\n"
        f"- ✅ **{count.get('Positive', 0)}** are Positive\n"
        f"- 🔻 **{count.get('Negative', 0)}** are Negative\n"
        f"- ⚪ **{count.get('Neutral', 0)}** are Neutral\n"
    )
    avg_sent = sum(sentiments) / len(sentiments)
    if avg_sent > 0.05:
        trend = "🟢 Overall sentiment is **Positive**"
    elif avg_sent < -0.05:
        trend = "🔴 Overall sentiment is **Negative**"
    else:
        trend = "⚪ Overall sentiment is **Neutral**"
    return summary + f"\n\n**Average Sentiment Score**: `{avg_sent:.2f}`\n\n{trend}"

def create_sentiment_timeline(sentiments, parsed_dates):
    """Create a timeline chart showing sentiment over time."""
    if not sentiments or not parsed_dates or len(sentiments) != len(parsed_dates):
        return None

    df = pd.DataFrame({
        'Date': parsed_dates,
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
        'Date': [d.date() if isinstance(d, datetime) else datetime.strptime(d, '%b %d, %Y').date() if isinstance(d, str) else d for d in dates],
        'Source': sources,
        'Sentiment': sentiments
    })

    # Filter out entries where 'Date' might be NaT due to parsing errors
    df.dropna(subset=['Date'], inplace=True)
    if df.empty: return None

    # Calculate week and year for grouping
    df['Week'] = df['Date'].apply(lambda x: x.isocalendar()[1]).astype(int) # Get week number
    df['Year'] = df['Date'].apply(lambda x: x.year)
    df['Week_Year'] = df['Year'].astype(str) + '-W' + df['Week'].astype(str).str.zfill(2) # Ensure 2 digits for week

    heatmap_data = df.groupby(['Source', 'Week_Year'])['Sentiment'].mean().reset_index()

    if len(heatmap_data) > 0:
        # Sort columns (Week_Year) correctly for chronological order
        pivot_table = heatmap_data.pivot(index='Source', columns='Week_Year', values='Sentiment')
        pivot_table = pivot_table.reindex(columns=sorted(pivot_table.columns)) # Sort columns chronologically

        fig = px.imshow(pivot_table.values,
                        labels=dict(x="Time Period", y="News Source", color="Avg Sentiment"),
                        x=pivot_table.columns,
                        y=pivot_table.index,
                        color_continuous_scale='RdYlGn', # Red-Yellow-Green for sentiment
                        range_color=[-1, 1], # Set consistent color range for sentiment
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
    
    if not all_text.strip(): return []

    words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower()) # Words with 4 or more letters
    # More comprehensive list of common English stop words and financial stop words
    try:
        stop_words_list = requests.get("https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt", timeout=5).text.splitlines()
        stop_words = set(stop_words_list)
    except Exception as e:
        print(f"Could not load external stopwords: {e}. Using a default limited set.")
        stop_words = set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
            'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with', 'this', 'that', 'have', 'been', 'each',
            'which', 'their', 'time', 'would', 'there', 'could', 'other', 'also', 'just', 'like', 'about', 'more',
            'what', 'when', 'where', 'how', 'than', 'into', 'such', 'make', 'over', 'even', 'most', 'some', 'much'
        ])

    stop_words.update({
        'stock', 'market', 'company', 'investor', 'investors', 'shares', 'firm', 'group', 'new', 'report',
        'news', 'billion', 'million', 'trillion', 'analyst', 'price', 'trading', 'business', 'financial',
        'economy', 'economic', 'percent', 'would', 'could', 'should', 'also', 'said', 'say', 'says',
        'year', 'week', 'day', 'month', 'quarter', 'growth', 'earnings', 'revenue', 'profit', 'loss',
        'deal', 'acquisition', 'update', 'share', 'dividend', 'index', 'nasdaq', 'nyse', 'bse', 'nse',
        'ceo', 'executive', 'management', 'board', 'fund', 'fundmanager', 'capital', 'investment', 'money',
        'traders', 'analysts', 'rate', 'interest', 'inflation', 'gdp', 'forecast', 'outlook', 'future',
        'potential', 'risk', 'opportunity', 'performance', 'results', 'profitability', 'outlook', 'guidance',
        'analyst', 'analysts', 'estimates', 'target', 'rating', 'buy', 'sell', 'hold', 'downgrade', 'upgrade',
        'impact', 'effect', 'industry', 'sector', 'global', 'national', 'local', 'developments', 'future',
        'forward', 'looking', 'projected', 'expects', 'expected', 'anticipates', 'anticipate', 'may', 'might',
        'can', 'could', 'will', 'would', 'should', 'must', 'has', 'have', 'had', 'been', 'being', 'was', 'were'
    })
    
    words = [word for word in words if word not in stop_words]
    word_freq = Counter(words).most_common(20) # Top 20 words
    return word_freq

def create_sentiment_metrics_cards(sentiments, articles):
    """Creates advanced metric cards for sentiment analysis."""
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
    st.markdown(f"<h3 class='section-title'>News Sentiment Analysis for {ticker.upper()}</h3>", unsafe_allow_html=True)

    num_articles = st.slider("Number of Articles to Analyze", min_value=10, max_value=100, value=30, step=10, key=f"num_articles_sentiment_{ticker}")

    # Check if API keys are set before proceeding
    if news_api_key == "YOUR_NEWSAPI_KEY" or fmp_api_key == "YOUR_FMP_KEY":
        st.error("❌ NewsAPI_KEY or FMP_API_KEY is not set. Please update `app.py` with your API keys to use News Sentiment analysis.")
        return

    with st.spinner("Fetching and analyzing news..."):
        # Get company name using FMP
        company_name = get_company_name_from_ticker(ticker, fmp_api_key)
        
        # Construct a robust search query for NewsAPI
        search_query_parts = []
        if company_name and company_name != ticker: # Prioritize multi-word company name
            search_query_parts.append(f'"{company_name}"')
        search_query_parts.append(f'"{ticker}"') # Always include ticker

        # Add generic financial keywords only if not already covered by company name
        financial_keywords = ["stock market", "finance news", "investment", "shares", "company earnings", "industry outlook"]
        for keyword in financial_keywords:
            # Check if parts of the keyword are not already present in company_name or ticker
            is_present = False
            for part in keyword.split():
                if part.lower() in company_name.lower() or part.lower() in ticker.lower():
                    is_present = True
                    break
            if not is_present:
                search_query_parts.append(f'"{keyword}"')

        # Combine unique and relevant search terms
        search_query = " OR ".join(list(set(search_query_parts)))
        
        articles = fetch_news_articles(search_query, news_api_key, total_articles=num_articles)

    if not articles:
        st.info("🔍 No recent news articles found for this ticker or query. This could be due to: \n"
                "- No articles matching the refined query in the last 30 days.\n"
                "- NewsAPI.com rate limits being hit.\n"
                "- An invalid or expired NewsAPI_KEY.\n"
                "Try a different ticker or check your API key.")
        return

    sentiments, parsed_dates, sources, urls, titles = [], [], [], [], []

    for article in articles:
        text = article.get('description') or article.get('content') or ""
        if not text or not text.strip(): # Skip articles with empty content
            continue
            
        sentiment_score = analyze_sentiment(text)
        sentiments.append(sentiment_score)
        titles.append(article.get('title', 'No Title'))
        sources.append(article.get('source', {}).get('name', 'Unknown Source'))
        urls.append(article.get('url', '#'))

        try:
            # Parse date, handle potential errors
            date_str = article.get('publishedAt', '')
            if date_str:
                date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
                parsed_dates.append(date)
            else:
                parsed_dates.append(datetime.now()) # Fallback for missing date
        except ValueError:
            parsed_dates.append(datetime.now()) # Fallback for malformed date

    if not sentiments: # No valid articles found after parsing and filtering
        st.info("No valid articles found for sentiment analysis after processing (articles might have been empty or malformed).")
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
            st.info("Not enough data to generate Sentiment Timeline.")
    
    with chart_cols[1]:
        st.markdown("<h5>Sentiment Heatmap by Source</h5>", unsafe_allow_html=True)
        # Ensure dates passed to heatmap are in the correct format (e.g., '%b %d, %Y' string or datetime objects)
        formatted_dates_for_heatmap = [d.strftime('%b %d, %Y') for d in parsed_dates]
        heatmap_fig = create_sentiment_heatmap(sentiments, formatted_dates_for_heatmap, sources)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.info("Not enough data to generate Sentiment Heatmap (e.g., too few unique sources/dates).")

    st.markdown("---")
    st.markdown("<h4 class='section-subtitle'>Key Topics & Articles</h4>", unsafe_allow_html=True)
    word_freq = create_word_cloud_data(articles)
    if word_freq:
        st.markdown("<h5>Most Mentioned Keywords:</h5>", unsafe_allow_html=True)
        words_df = pd.DataFrame(word_freq, columns=['Keyword', 'Frequency'])
        st.dataframe(words_df, use_container_width=True)
    else:
        st.info("No significant keywords found for analysis (articles might be too short or too few).")

    st.markdown("<h5>Latest News Articles:</h5>", unsafe_allow_html=True)
    for i in range(len(articles)):
        article = articles[i]
        article_text = article.get('description') or article.get('content') or ""
        if not article_text.strip(): continue # Skip if article has no meaningful text
            
        sentiment_score = analyze_sentiment(article_text)
        label = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"
        color_map = {'Positive': 'limegreen', 'Negative': 'tomato', 'Neutral': 'lightgray'}
        icon_map = {'Positive': '🟢', 'Negative': '🔴', 'Neutral': '⚪'}

        # Ensure title and URL exist before displaying
        article_title = article.get('title', 'No Title Available')
        article_url = article.get('url', '#')
        article_source = article.get('source', {}).get('name', 'Unknown Source')
        article_pub_date = article.get('publishedAt', 'N/A').split('T')[0] # Only date part

        st.markdown(f"""
        <div class="news-card" style="border: 2px solid {color_map[label]};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h4 style="margin: 0; color: {color_map[label]};">{icon_map[label]} {label} Sentiment</h4>
                <span style="background: {color_map[label]}; padding: 5px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; color: white;">
                    {sentiment_score:.3f}
                </span>
            </div>
            <h3 style="margin: 10px 0;">
                <a href="{article_url}" target="_blank" class="news-link">
                    📰 {article_title}
                </a>
            </h3>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px;">
                <span style="color:#B0BEC5; font-size: 14px;">📺 {article_source}</span>
                <span style="color:#B0BEC5; font-size: 14px;">📅 {article_pub_date}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.warning("""
    **Disclaimer for News Sentiment Analysis:**
    - **Data Source:** News articles are sourced from NewsAPI.com, which may have limitations on historical depth, number of articles per query, and specific news providers in its free tier.
    - **Sentiment Accuracy:** Sentiment analysis is performed using a rule-based algorithm (`TextBlob`), which provides a generalized sentiment. It may not fully grasp nuanced financial language, sarcasm, or specific market context, and can produce inaccurate results.
    - **Freshness:** While efforts are made to get recent news, delays can occur.
    - **Bias:** News content itself can carry biases.
    
    **Do NOT use this sentiment analysis for making actual investment decisions.** It is for general informational purposes only.
    """)
