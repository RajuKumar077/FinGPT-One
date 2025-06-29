from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import time
from retrying import retry
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Load API keys from .env file
load_dotenv()
FMP_API_KEY = os.getenv("FMP_API_KEY", "5C9DnMCAzYam2ZPjNpOxKLFxUiGhrJDD")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "874ba654bdcd4aa7b68f7367a907cc2f")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "8UU32LX81NSED6CM")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAK8BevJ1wIrwMoYDsnCLQXdZlFglF92WE")

# Streamlit configuration
st.set_page_config(
    page_title="FinGPT One - Stock Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# Custom CSS for consistent styling
st.markdown("""
<style>
    .main { background-color: #1E1E1E; }
    .section-title { color: #00ACC1; font-size: 24px; font-weight: bold; margin-bottom: 10px; }
    .section-subtitle { color: #B0BEC5; font-size: 18px; font-weight: bold; margin-top: 20px; }
    .metric-card { 
        background-color: #2D2D2D; 
        padding: 15px; 
        border-radius: 10px; 
        text-align: center; 
        margin-bottom: 10px;
    }
    .metric-card .card-icon { font-size: 24px; margin-bottom: 5px; }
    .metric-card .card-title { font-size: 14px; color: #B0BEC5; margin-bottom: 5px; }
    .metric-card .card-value { font-size: 20px; font-weight: bold; color: #00ACC1; }
    .news-card { 
        background-color: #2D2D2D; 
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 15px;
    }
    .news-link { color: #00ACC1; text-decoration: none; }
    .news-link:hover { text-decoration: underline; }
    .sidebar .sidebar-content { background-color: #2D2D2D; }
    h1 { color: #00ACC1; font-family: 'Inter', sans-serif; }
    p { color: #B0BEC5; font-family: 'Inter', sans-serif; }
</style>
<link href='https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap' rel='stylesheet'>
""", unsafe_allow_html=True)

# Sidebar for ticker selection
st.sidebar.title("FinGPT One")
st.sidebar.markdown("### Stock Selection")
if ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
    st.sidebar.warning("⚠️ Alpha Vantage API key is missing. Ticker suggestions will not work.")
query = st.sidebar.text_input("Enter ticker or company name:", value="AAPL", key="ticker_input")
from pages.yahoo_autocomplete import fetch_ticker_suggestions
suggestions = fetch_ticker_suggestions(query, ALPHA_VANTAGE_API_KEY) if query else []
selected_suggestion = st.sidebar.selectbox("Select a stock:", [""] + suggestions, key="ticker_select")
ticker = selected_suggestion.split(" - ")[0].strip().upper() if selected_suggestion else query.strip().upper()

# Validate ticker and store in session state
if not ticker:
    st.warning("⚠️ Please enter or select a valid ticker to proceed.")
    st.stop()

st.session_state.current_ticker = ticker

# Main page header
st.markdown("""
    <h1>📈 FinGPT One - Stock Analysis Dashboard</h1>
    <p>Comprehensive AI-powered stock analysis tool</p>
""", unsafe_allow_html=True)
st.markdown(f"<h2 class='section-title'>Analysis for {ticker}</h2>", unsafe_allow_html=True)

# Note: Page-specific logic is handled in individual page files
