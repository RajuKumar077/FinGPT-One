import streamlit as st

def render_cards(current_price, proj_price, roi_pct, profit, proj_value, investment):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(card_html("📌 Current Price", f"₹{current_price:,.2f}", "#3b82f6"), unsafe_allow_html=True)

    with col2:
        st.markdown(card_html("📈 Projected Price", f"₹{proj_price:,.2f}", "#2563eb"), unsafe_allow_html=True)

    with col3:
        color = "#16a34a" if profit >= 0 else "#dc2626"
        st.markdown(card_html("💸 Estimated ROI", f"{roi_pct:.2f}%", color), unsafe_allow_html=True)

    st.markdown(f"""
        <div style='
            margin-top: 40px;
            padding: 24px 36px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 12px 30px rgba(59, 130, 246, 0.25);
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            text-align: center;
            user-select: none;
            color: #1e40af;
            font-size: 22px;
            font-weight: 700;
            border: 1px solid rgba(59, 130, 246, 0.35);
            transition: box-shadow 0.3s ease, transform 0.3s ease;
        ' onmouseover="this.style.transform='scale(1.03)'; this.style.boxShadow='0 18px 40px rgba(59, 130, 246, 0.4)';" onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 12px 30px rgba(59, 130, 246, 0.25)';">
            Projected Portfolio Value: <span style='color:#1d4ed8;'>₹{proj_value:,.2f}</span><br>
            <small style='font-weight: 500; color: #3b82f6;'>(Investment: ₹{investment:,})</small>
        </div>
    """, unsafe_allow_html=True)

def card_html(title, value, color):
    return f"""
    <div style='
        background: rgba(255, 255, 255, 0.22);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 18px;
        padding: 28px 24px;
        margin: 8px 4px;
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.08);
        border-left: 8px solid;
        border-image-slice: 1;
        border-image-source: linear-gradient(180deg, {color} 0%, #60a5fa 100%);
        cursor: default;
        user-select: none;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        transition: transform 0.35s cubic-bezier(.25,.8,.25,1), box-shadow 0.35s cubic-bezier(.25,.8,.25,1);
        text-align: center;
    '
    onmouseover="this.style.transform='translateY(-10px) scale(1.05)'; this.style.boxShadow='0 20px 40px rgba(0, 0, 0, 0.15)';"
    onmouseout="this.style.transform='translateY(0) scale(1)'; this.style.boxShadow='0 10px 22px rgba(0, 0, 0, 0.08)';"
    >
        <div style='color: {color}; font-weight: 800; font-size: 18px; text-shadow: 0 1px 2px rgba(0,0,0,0.12);'>{title}</div>
        <div style='font-size: 32px; font-weight: 900; margin-top: 12px; color: #1e293b; text-shadow: 0 2px 4px rgba(59,130,246,0.15);'>{value}</div>
    </div>
    """
