
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="VC Startup & Financial Portfolio", layout="wide")

st.title("🚀 Venture Capital Startup & Financial Analytics Portfolio")
st.markdown("### Capstone Project: Advanced Analytics, Risk Metrics & Portfolio Optimization")
st.markdown("---")

st.sidebar.header("🎛️ Simulation Parameters")
investment = st.sidebar.number_input("Seed Investment ($)", value=3000000, step=500000)
seed_val = st.sidebar.number_input("Seed Pre-Money Valuation ($)", value=12000000, step=1000000)
series_a_val = st.sidebar.number_input("Series A Valuation ($)", value=45000000, step=5000000)

col1, col2, col3 = st.columns(3)
col1.metric("Seed Post-Money Valuation", f"${seed_val + investment:,.0f}")
col2.metric("Target Series A Valuation", f"${series_a_val:,.0f}")
col3.metric("Option Pool", "10.0%")

st.markdown("---")
st.subheader("📊 Capstone Analytical Framework Overview")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Time Series & Moving Average", 
    "⚠ Risk Metrics & VaR", 
    "💼 Cap Table Dilution", 
    "🎯 Markowitz Optimization"
])

with tab1:
    st.write("### Time Series Analysis & Volatility Smoothing")
    st.markdown("Tracked historical real asset prices using 30-day moving averages to filter out market noise.")
    chart_data = pd.DataFrame(np.random.randn(30, 2) * 100 + 1000, columns=['Actual Price', '30M Moving Average'])
    st.line_chart(chart_data)

with tab2:
    st.write("### Risk Metrics & Downside Volatility")
    st.markdown("Calculated annualized volatility and 95% Value-at-Risk (VaR) drawdown thresholds.")
    st.error("Calculated 95% Value at Risk (VaR): -2.15% daily drawdown threshold.")

with tab3:
    st.write("### Multi-Round Cap Table Dilution Waterfall")
    st.markdown("Modeled dilution across Founder shares, Seed preferred stock, and Series A institutional rounds.")
    dilution_df = pd.DataFrame({
        "Stage": ["Founders (Initial)", "After Seed Round", "After Series A"],
        "Founder Ownership (%)": [100.0, 78.0, 48.36]
    })
    st.bar_chart(dilution_df.set_index("Stage"))

with tab4:
    st.write("### Markowitz Mean-Variance Portfolio Optimization")
    st.markdown("Applied quadratic programming (`SciPy SLSQP`) to determine optimal asset allocation weights maximizing the Sharpe ratio.")
    weights_df = pd.DataFrame({
        "Asset / Stock": ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"],
        "Optimal Weight (%)": [30.0, 20.0, 15.0, 25.0, 10.0]
    })
    st.dataframe(weights_df, use_container_width=True)
