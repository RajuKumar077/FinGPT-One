# FinGPT-One - Stock Analysis Platform

Professional financial analysis platform with multi-source data aggregation, predictive modeling, and market intelligence.

## Overview

FinGPT-One is a comprehensive financial analysis platform that integrates multiple data sources, machine learning models, and visualizations to provide insights into stock market performance. Built with Streamlit, it features a modern UI with real-time data processing, predictive analytics, and sentiment analysis.

## Features

### Stock Summary Dashboard
- Interactive candlestick charts with multiple moving averages (5, 10, 20, 50-day)
- Volume analysis with color-coded bars
- RSI momentum indicator (14-day) with overbought/oversold levels
- Key metrics: price, high/low, volume, 50-day MA with percentage changes
- 30-day statistics: mean close, volatility, max drawdown, Sharpe ratio, win rate
- Optional AI summary integration via Google Gemini

### Forecasting Module
Multi-model time series forecasting with evaluation:

- **ARIMA**: AutoRegressive Integrated Moving Average for trend-based predictions
- **SARIMA**: Seasonal ARIMA for cyclical patterns
- **Prophet**: Handles seasonality, holidays, and changepoints
- **Exponential Smoothing**: Holt-Winters method for level and trend
- **Moving Average**: Baseline predictions

Features: Configurable horizons (7-90 days), confidence intervals, performance metrics (RMSE, MAE, MAPE, R²), residual analysis, model comparison, CSV export.

### Probabilistic Prediction Engine
Machine learning-powered stock direction prediction:

- **15+ Technical Indicators**: Price action features, moving average ratios, RSI, MACD, Bollinger Bands, ROC, volume ratios
- **Random Forest Classifier**: 150 estimators with balanced class weights, feature importance visualization, multi-horizon predictions (1-day, 3-day, 5-day)
- **Performance Metrics**: Accuracy, AUC-ROC, confusion matrices, classification reports
- **Backtesting**: Strategy vs Buy & Hold comparison with cumulative returns visualization

### Financial Intelligence Dashboard
Fundamental analysis with:

- **Financial Statements**: Income Statement, Balance Sheet, Cash Flow Statement
- **Financial Ratios**: Profitability (margins), Liquidity (current ratio), Leverage (debt-to-equity), Efficiency (ROA, ROE)
- **Visualizations**: Revenue waterfall charts, margin trends, balance sheet composition, cash flow analysis, growth metrics
- **Executive Summary**: Key metrics with period-over-period growth indicators

### News Sentiment Analysis
Real-time market sentiment:

- Article aggregation (up to 100 articles from NewsAPI)
- Sentiment scoring using TextBlob (-1 to +1 polarity)
- Visualizations: sentiment timeline, distribution, keyword frequency
- Article display with sentiment labels, sources, and dates
- Automatic ticker-to-company name conversion

### Ticker Autocomplete
- Alpha Vantage integration for real-time symbol search
- Popular tickers list (50+ major stocks)
- Auto-suggestions after 2+ characters
- Region and exchange information

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- API keys for data sources (see Security section)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/RajuKumar077/FinGPT-One.git
cd FinGPT-One
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml` and add your API keys:
```toml
TIINGO_API_KEY = "your_key_here"
FMP_API_KEY = "your_key_here"
ALPHA_VANTAGE_API_KEY = "your_key_here"
GEMINI_API_KEY = "your_key_here"
NEWS_API_KEY = "your_key_here"
```

Get API keys:
- [Tiingo](https://api.tiingo.com/)
- [FMP](https://site.financialmodelingprep.com/)
- [Alpha Vantage](https://www.alphavantage.co/)
- [Google Gemini](https://makersuite.google.com/)
- [NewsAPI](https://newsapi.org/)

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Enter a ticker symbol in the sidebar (e.g., AAPL, MSFT, TSLA)
2. Select a dashboard module:
   - Stock Summary: Price charts and technical indicators
   - Forecasting: Multi-model price predictions
   - Probabilistic Models: AI-powered direction predictions
   - Financials: Fundamental analysis and statements
   - News Sentiment: Market sentiment from news articles
3. Interact with visualizations: zoom, pan, hover for details, download forecast data

## Architecture

### Data Flow
```
User Input → Multi-Source Data Engine (5-layer failover)
    → Data Processing & Caching
    → Component Routing
    → Interactive Visualizations
```

### Data Sources (Multi-Layer Failover)
1. Yahoo Finance (via yfinance) - Primary, no API key required
2. Tiingo API - Institutional-grade data
3. Alpha Vantage - Stock data and symbol search
4. Financial Modeling Prep - Company profiles and financial statements
5. Synthetic Data - Fallback for testing

### Component Architecture
- Modular design with separate components for each feature
- Separation of concerns: data fetching, processing, visualization
- Reusable functions and utilities
- Session state management for efficient caching

## Security

This project implements secure secret management:

- API keys stored in `.streamlit/secrets.toml` (gitignored)
- Environment variable fallback for local development
- No hardcoded keys in source code
- Streamlit Cloud secrets integration

See [SECURITY.md](SECURITY.md) for detailed documentation.

## Technologies

- **Framework**: Streamlit 1.45.1, Python 3.9+
- **Data Processing**: Pandas 2.0.3, NumPy 1.26.4, yfinance 0.2.28
- **Machine Learning**: scikit-learn 1.3.0, statsmodels 0.12.0+, Prophet, ta
- **Visualization**: Plotly 5.20.0, Matplotlib 3.8.4, Seaborn 0.13.2
- **NLP**: TextBlob 0.18.0
- **Utilities**: requests 2.31.0, python-dotenv

## Project Structure

```
FinGPT-One/
├── app.py                          # Main application
├── Components/
│   ├── stock_summary.py           # OHLCV charts & indicators
│   ├── forecast_module.py         # Time series forecasting
│   ├── probabilistic_stock_model.py  # ML prediction engine
│   ├── financials.py             # Fundamental analysis
│   ├── news_sentiment.py          # News aggregation & sentiment
│   ├── fmp_autocomplete.py       # FMP ticker search
│   └── yahoo_autocomplete.py      # Yahoo Finance ticker search
├── .streamlit/
│   ├── secrets.toml              # API keys (gitignored)
│   └── secrets.toml.example       # Template
├── assets/
│   └── style.css                  # Custom CSS
├── requirements.txt               # Dependencies
├── SECURITY.md                    # Security documentation
└── README.md                      # This file
```

## Use Cases

- Day Traders: Technical analysis with RSI, MACD, candlestick patterns
- Swing Traders: Multi-day predictions with probabilistic models
- Long-term Investors: Fundamental analysis with financial statements
- Researchers: Time series forecasting with multiple models and backtesting
- Analysts: News sentiment analysis for market sentiment tracking

## Contributing

Contributions are welcome. Please submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Disclaimer

This application is for educational and informational purposes only. It is NOT financial advice.

- Stock market predictions are inherently uncertain
- Past performance does not guarantee future results
- Always do your own research before making investment decisions
- The authors are not responsible for any financial losses

## Contact

**Project Maintainer**: Raju Kumar

- GitHub: [@RajuKumar077](https://github.com/RajuKumar077)
- Project Link: [https://github.com/RajuKumar077/FinGPT-One](https://github.com/RajuKumar077/FinGPT-One)
