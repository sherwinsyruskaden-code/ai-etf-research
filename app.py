"""
AI ETF Research -- Streamlit Dashboard
Run with:  streamlit run app.py

Read-only: this dashboard queries existing DB data.
It does NOT retrain models, write to the database, or generate live signals.
"""
import streamlit as st

st.set_page_config(
    page_title="AI ETF Research Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
with st.sidebar:
    st.title("🔬 AI ETF Research")
    st.caption("Research Dashboard — Read-only")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "📋 Overview",
            "📈 ETF Explorer",
            "📉 Backtest Results",
            "🔭 Research / Methodology",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        """
        **⚠️ Disclaimer**

        All outputs shown are **historical / backtested** only.
        This dashboard does **not** generate live trading signals
        and does **not** recommend any investment action.

        **Kill-gate verdict:** FAILED
        """
    )

# Dispatch to page modules
if page == "📋 Overview":
    from dashboard import page_overview
    page_overview.render()

elif page == "📈 ETF Explorer":
    from dashboard import page_etf_explorer
    page_etf_explorer.render()

elif page == "📉 Backtest Results":
    from dashboard import page_backtest
    page_backtest.render()

elif page == "🔭 Research / Methodology":
    from dashboard import page_methodology
    page_methodology.render()
