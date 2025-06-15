import streamlit as st

def render_header():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@600;700;900&display=swap');

            .header-container {
                text-align: center;
                padding: 30px 10px 10px 10px;
                font-family: 'Inter', sans-serif;
            }

            .header-title {
                font-size: 3em;
                font-weight: 900;
                background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.3rem;
            }

            .header-subtitle {
                font-size: 1.2rem;
                color: #555;
                font-weight: 600;
                letter-spacing: 0.5px;
            }

            .header-line {
                height: 2px;
                width: 60px;
                margin: 15px auto;
                background: linear-gradient(to right, #0077b6, #00b4d8);
                border-radius: 2px;
            }
        </style>

        <div class="header-container">
            <div class="header-title">FinSight AI</div>
            <div class="header-subtitle">Smarter Stock Insights • Minimal UI • Premium Forecasts</div>
            <div class="header-line"></div>
        </div>
        """,
        unsafe_allow_html=True
    )
