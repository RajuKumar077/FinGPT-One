import streamlit as st
import yfinance as yf
import pandas as pd  # Required for pd.DataFrame
import numpy as np  # For numerical operations


# Cache stock info for better performance
@st.cache_data(ttl=86400, show_spinner=False)  # Cache for 24 hours
def get_stock_info(ticker):
    """Fetches comprehensive stock information."""
    try:
        stock = yf.Ticker(ticker)
        # Ensure we fetch all necessary info for detailed display
        info = stock.info
        return info
    except Exception as e:
        st.error(f"Could not fetch stock information for {ticker}: {e}")
        return None


def format_value(value, prefix="", suffix="", decimal_places=2, is_percentage=False, is_currency=False):
    """Helper function to format numeric values nicely."""
    if isinstance(value, (int, float)):
        if is_percentage:
            return f"{value * 100:.{decimal_places}f}{suffix}"
        if is_currency:
            return f"{prefix}{value:,.{decimal_places}f}{suffix}"
        return f"{value:,.{decimal_places}f}{suffix}"
    return "N/A"


def display_stock_summary(ticker):
    """Displays company overview and key metrics for a given ticker."""
    info = get_stock_info(ticker)
    if info is None:
        st.info(f"No summary data available for {ticker}. Please try another ticker or check the symbol.")
        return

    st.markdown(f"<h3 class='section-title'>Company Profile for {ticker.upper()}</h3>", unsafe_allow_html=True)

    # Basic Company Info in a structured layout
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Name:** {info.get('longName', 'N/A')}")
        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
        st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
    with col2:
        # Provide a clickable link if website exists
        website = info.get('website', 'N/A')
        if website != 'N/A' and website.startswith('http'):
            st.markdown(f"**Website:** [{website}]({website})")
        else:
            st.markdown(f"**Website:** {website}")
        st.markdown(f"**Country:** {info.get('country', 'N/A')}")
        st.markdown(f"**Exchange:** {info.get('exchange', 'N/A')}")
        st.markdown(f"**Currency:** {info.get('currency', 'N/A')}")

    st.markdown("<h4>Business Summary:</h4>", unsafe_allow_html=True)
    st.markdown(info.get('longBusinessSummary', 'No business summary available.'))

    st.markdown(f"<h3 class='section-title'>Key Valuation Metrics</h3>", unsafe_allow_html=True)

    # Valuation Metrics in a visually appealing grid
    val_metrics = {
        "Market Cap": {'value': info.get('marketCap'), 'icon': '💰', 'prefix': '₹', 'is_currency': True,
                       'decimal_places': 0},
        "Enterprise Value": {'value': info.get('enterpriseValue'), 'icon': '🏢', 'prefix': '₹', 'is_currency': True,
                             'decimal_places': 0},
        "Trailing P/E": {'value': info.get('trailingPE'), 'icon': '📈', 'decimal_places': 2,
                         'condition': lambda v: "warning" if v > 30 else "success" if v < 10 else ""},
        "Forward P/E": {'value': info.get('forwardPE'), 'icon': '📊', 'decimal_places': 2,
                        'condition': lambda v: "warning" if v > 30 else "success" if v < 10 else ""},
        "PEG Ratio": {'value': info.get('pegRatio'), 'icon': '🚀', 'decimal_places': 2,
                      'condition': lambda v: "success" if 0 < v <= 1 else ""},
        "Price to Book": {'value': info.get('priceToBook'), 'icon': '📚', 'decimal_places': 2,
                          'condition': lambda v: "warning" if v > 3 else "success" if v < 1 else ""},
        "EV/EBITDA": {'value': info.get('enterpriseToEbitda'), 'icon': '�', 'decimal_places': 2},
        "Trailing EPS": {'value': info.get('trailingEps'), 'icon': '💵', 'prefix': '₹', 'is_currency': True,
                         'decimal_places': 2},
        "Book Value Per Share": {'value': info.get('bookValue'), 'icon': '📖', 'prefix': '₹', 'is_currency': True,
                                 'decimal_places': 2},
        "Debt to Equity": {'value': info.get('debtToEquity'), 'icon': '🔗', 'suffix': '%', 'decimal_places': 2,
                           'is_percentage': True,
                           'condition': lambda v: "warning" if v > 100 else "success" if v < 50 else ""},
    }

    cols = st.columns(3)
    metrics_list = list(val_metrics.items())
    for i, (label, data) in enumerate(metrics_list):
        with cols[i % 3]:
            value = data['value']
            icon = data['icon']
            prefix = data.get('prefix', '')
            suffix = data.get('suffix', '')
            decimal_places = data.get('decimal_places', 2)
            is_percentage = data.get('is_percentage', False)
            is_currency = data.get('is_currency', False)

            formatted_value = format_value(value, prefix=prefix, suffix=suffix,
                                           decimal_places=decimal_places,
                                           is_percentage=is_percentage, is_currency=is_currency)

            card_class = "info-badge"
            if 'condition' in data and isinstance(value, (int, float)):
                card_class += " " + data['condition'](value)

            st.markdown(f"""
                <div class="{card_class}">
                    <div class="badge-icon">{icon}</div>
                    <div class="badge-label">{label}</div>
                    <div class="badge-value">{formatted_value}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown(f"<h3 class='section-title'>Dividend & Share Statistics</h3>", unsafe_allow_html=True)

    div_share_metrics = {
        "Dividend Yield": {'value': info.get('dividendYield'), 'icon': '💸', 'suffix': '%', 'decimal_places': 2,
                           'is_percentage': True, 'condition': lambda v: "success" if v > 0.02 else ""},
        "Dividend Payout Ratio": {'value': info.get('payoutRatio'), 'icon': '📈', 'suffix': '%', 'decimal_places': 2,
                                  'is_percentage': True, 'condition': lambda v: "warning" if v > 0.7 else ""},
        "Last Dividend Value": {'value': info.get('lastDividendValue'), 'icon': '💲', 'prefix': '₹', 'is_currency': True,
                                'decimal_places': 2},
        "Ex-Dividend Date": {'value': info.get('exDividendDate'), 'icon': '📅', 'type': 'date'},
        "Dividend Date": {'value': info.get('dividendDate'), 'icon': '🗓️', 'type': 'date'},
        "Shares Outstanding": {'value': info.get('sharesOutstanding'), 'icon': '🔢', 'decimal_places': 0},
        "Float Shares": {'value': info.get('floatShares'), 'icon': '🌊', 'decimal_places': 0},
    }

    cols = st.columns(3)
    metrics_list = list(div_share_metrics.items())
    for i, (label, data) in enumerate(metrics_list):
        with cols[i % 3]:
            value = data['value']
            icon = data['icon']

            formatted_value = "N/A"
            card_class = "info-badge"

            if data.get('type') == 'date':
                if value:
                    try:
                        formatted_value = pd.to_datetime(value, unit='s').strftime('%Y-%m-%d')
                    except:
                        formatted_value = str(value)
                else:
                    formatted_value = "N/A"
            else:
                prefix = data.get('prefix', '')
                suffix = data.get('suffix', '')
                decimal_places = data.get('decimal_places', 2)
                is_percentage = data.get('is_percentage', False)
                is_currency = data.get('is_currency', False)
                formatted_value = format_value(value, prefix=prefix, suffix=suffix,
                                               decimal_places=decimal_places,
                                               is_percentage=is_percentage, is_currency=is_currency)
                if 'condition' in data and isinstance(value, (int, float)):
                    card_class += " " + data['condition'](value)

            st.markdown(f"""
                <div class="{card_class}">
                    <div class="badge-icon">{icon}</div>
                    <div class="badge-label">{label}</div>
                    <div class="badge-value">{formatted_value}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown(f"<h3 class='section-title'>Price Performance & Trading</h3>", unsafe_allow_html=True)

    perf_metrics = {
        "Day's Low": {'value': info.get('dayLow'), 'icon': '📉', 'prefix': '₹', 'is_currency': True,
                      'decimal_places': 2},
        "Day's High": {'value': info.get('dayHigh'), 'icon': '📈', 'prefix': '₹', 'is_currency': True,
                       'decimal_places': 2},
        "52-Week Low": {'value': info.get('fiftyTwoWeekLow'), 'icon': '🔻', 'prefix': '₹', 'is_currency': True,
                        'decimal_places': 2},
        "52-Week High": {'value': info.get('fiftyTwoWeekHigh'), 'icon': '🔺', 'prefix': '₹', 'is_currency': True,
                         'decimal_places': 2},
        "52-Week Change": {'value': info.get('52WeekChange'), 'icon': '🔄', 'suffix': '%', 'decimal_places': 2,
                           'is_percentage': True, 'condition': lambda v: "success" if v > 0 else "warning"},
        "Analyst Target Price": {'value': info.get('targetMeanPrice'), 'icon': '🎯', 'prefix': '₹', 'is_currency': True,
                                 'decimal_places': 2},
    }

    cols = st.columns(3)
    metrics_list = list(perf_metrics.items())
    for i, (label, data) in enumerate(metrics_list):
        with cols[i % 3]:
            value = data['value']
            icon = data['icon']
            prefix = data.get('prefix', '')
            suffix = data.get('suffix', '')
            decimal_places = data.get('decimal_places', 2)
            is_percentage = data.get('is_percentage', False)
            is_currency = data.get('is_currency', False)

            formatted_value = format_value(value, prefix=prefix, suffix=suffix,
                                           decimal_places=decimal_places,
                                           is_percentage=is_percentage, is_currency=is_currency)

            card_class = "info-badge"
            if 'condition' in data and isinstance(value, (int, float)):
                card_class += " " + data['condition'](value)

            st.markdown(f"""
                <div class="{card_class}">
                    <div class="badge-icon">{icon}</div>
                    <div class="badge-label">{label}</div>
                    <div class="badge-value">{formatted_value}</div>
                </div>
            """, unsafe_allow_html=True)

    # Visualizing 52-Week Range
    fifty_two_week_low = info.get('fiftyTwoWeekLow')
    fifty_two_week_high = info.get('fiftyTwoWeekHigh')
    current_price = info.get('currentPrice')

    if all(isinstance(x, (int, float)) for x in [fifty_two_week_low, fifty_two_week_high, current_price]) and \
            fifty_two_week_high > fifty_two_week_low:
        st.markdown("<h4 class='section-subtitle'>Current Price within 52-Week Range</h4>", unsafe_allow_html=True)

        # Calculate position as a percentage
        range_percent = (current_price - fifty_two_week_low) / (fifty_two_week_high - fifty_two_week_low)
        range_percent = np.clip(range_percent, 0, 1)  # Clip to ensure it's between 0 and 1

        st.markdown(f"""
        <div style="
            background-color: #2C2C2E; padding: 20px; border-radius: 15px; margin-top: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3); text-align: center;
            border: 1px solid rgba(102, 102, 102, 0.6); /* Slightly more visible border */
        ">
            <h5 style="color: #007AFF; margin-bottom: 15px; font-size: 1.1em;">
                Current Price: ₹{current_price:.2f} (52W Low: ₹{fifty_two_week_low:.2f} | 52W High: ₹{fifty_two_week_high:.2f})
            </h5>
            <div style="
                width: 100%; height: 30px; /* Increased height for better visibility */ background-color: #4A4A4C; border-radius: 10px; overflow: hidden;
                position: relative;
            ">
                <div style="
                    width: {range_percent * 100}%; height: 100%;
                    background: linear-gradient(90deg, #34C759, #8BC34A); /* Apple-like green gradient */
                    border-radius: 10px;
                "></div>
                <div style="
                    position: absolute; left: calc({range_percent * 100}% - 15px); top: 50%;
                    transform: translateY(-50%);
                    width: 30px; height: 30px; background-color: white; border-radius: 50%;
                    border: 3px solid #007AFF; box-shadow: 0 0 10px rgba(0, 122, 255, 0.7); /* Apple blue border/shadow */
                    display: flex; align-items: center; justify-content: center; font-size: 1.2em;
                    color: black; font-weight: bold;
                ">
                    <span style="font-size: 0.8em;">{range_percent * 100:.0f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("52-Week range data not available for visualization.")

