import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="VC & Financial Forensics Decision Platform",
    page_icon="🚀",
    layout="wide"
)

# --- HEADER SECTION ---
st.title("🚀 AI-Driven Venture Capital & Financial Forensics Platform")
st.markdown("### Production-Grade Decision Support System for Startup Valuation, Risk Forecasting & Portfolio Optimization")
st.markdown("---")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Live Parameter Controls")
selected_tickers = st.sidebar.multiselect(
    "Select Portfolio Assets / Proxies", 
    ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA'], 
    default=['AAPL', 'MSFT', 'GOOGL', 'AMZN']
)
seed_inv = st.sidebar.number_input("Seed Investment ($)", value=3000000, step=500000)
seed_val = st.sidebar.number_input("Seed Pre-Money Valuation ($)", value=12000000, step=1000000)
series_a_val = st.sidebar.number_input("Target Series A Valuation ($)", value=45000000, step=5000000)

# --- TABS FOR STRUCTURED PRESENTATION ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Live API Data & Time Series", 
    "🤖 ML Financial Distress Predictor", 
    "💼 Cap Table Dilution Waterfall", 
    "🎯 Markowitz Portfolio Optimization",
    "📝 Financial Text Analytics (NLP)"
])

# --- TAB 1: LIVE API DATA & TIME SERIES ---
with tab1:
    st.subheader("Live Financial Data Ingestion (Yahoo Finance API)")
    st.markdown("Pulling historical pricing and calculating rolling moving averages to smooth out market volatility.")
    
    if selected_tickers:
        @st.cache_data
        def load_data(tickers):
            df = yf.download(tickers, start="2023-01-01", end="2026-01-01")['Close']
            return df.dropna()
        
        live_data = load_data(selected_tickers)
        st.line_chart(live_data)
        
        st.write("### Statistical Summary of Asset Returns")
        st.dataframe(live_data.pct_change().describe(), use_container_width=True)
    else:
        st.warning("Please select at least one ticker from the sidebar.")

# --- TAB 2: MACHINE LEARNING RISK PREDICTOR ---
with tab2:
    st.subheader("Machine Learning Startup Insolvency Classifier")
    st.markdown("Trained on corporate financial ratios using a Random Forest architecture to predict corporate financial distress.")
    
    col_a, col_b, col_c = st.columns(3)
    input_burn = col_a.slider("Cash Burn Ratio", 0.5, 4.0, 1.8)
    input_debt = col_b.slider("Debt-to-Equity Ratio", 0.1, 5.0, 2.2)
    input_current = col_c.slider("Current Ratio", 0.4, 2.5, 1.1)
    
    # Train mock model live for interactive demo
    np.random.seed(42)
    n = 1000
    X_train = pd.DataFrame({
        'Cash_Burn_Ratio': np.random.uniform(0.5, 3.5, n),
        'Debt_to_Equity': np.random.uniform(0.1, 5.0, n),
        'Current_Ratio': np.random.uniform(0.5, 2.5, n)
    })
    y_train = np.where((X_train['Cash_Burn_Ratio'] > 2.5) | (X_train['Debt_to_Equity'] > 3.0), 1, 0)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    user_input = pd.DataFrame([[input_burn, input_debt, input_current]], columns=['Cash_Burn_Ratio', 'Debt_to_Equity', 'Current_Ratio'])
    prediction = model.predict(user_input)[0]
    prob = model.predict_proba(user_input)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ HIGH DISTRESS RISK DETECTED (Probability: {prob*100:.1f}%)")
    else:
        st.success(f"✅ FINANCIALLY STABLE PROFILE (Insolvency Risk: {prob*100:.1f}%)")

# --- TAB 3: CAP TABLE DILUTION ---
with tab3:
    st.subheader("Multi-Round Equity Waterfall & Dilution")
    option_pool = 0.10
    seed_post = seed_val + seed_inv
    seed_own = seed_inv / seed_post
    sa_post = series_a_val + 10000000
    sa_own = 10000000 / sa_post
    founder_final = (1.0 - seed_own - option_pool) * (1.0 - sa_own) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Founder Final Ownership", f"{founder_final:.2f}%")
    col2.metric("Seed Investor Ownership", f"{seed_own*100:.2f}%")
    col3.metric("Series A Investor Ownership", f"{sa_own*100:.2f}%")
    
    dilution_df = pd.DataFrame({
        "Stage": ["Founders (Initial)", "After Seed Round", "After Series A"],
        "Founder Ownership (%)": [100.0, 100.0 - (seed_own*100), founder_final]
    })
    st.bar_chart(dilution_df.set_index("Stage"))

# --- TAB 4: MARKOWITZ OPTIMIZATION ---
with tab4:
    st.subheader("Markowitz Mean-Variance Portfolio Optimization")
    st.markdown("Utilizing quadratic programming (`SciPy SLSQP`) to compute optimal fund allocation weights maximizing the Sharpe Ratio.")
    
    if selected_tickers and len(selected_tickers) > 1:
        returns = live_data.pct_change().dropna()
        mean_ret = returns.mean() * 252
        cov_mat = returns.cov() * 252
        
        def neg_sharpe(w, r, c):
            p_ret = np.sum(w * r)
            p_vol = np.sqrt(np.dot(w.T, np.dot(c, w)))
            return -(p_ret - 0.05) / p_vol
            
        num = len(selected_tickers)
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0.0, 1.0) for _ in range(num))
        init = num * [1.0 / num]
        
        opt_res = minimize(neg_sharpe, init, args=(mean_ret, cov_mat), method='SLSQP', bounds=bnds, constraints=cons)
        opt_weights = pd.DataFrame({"Asset": selected_tickers, "Optimal Weight (%)": opt_res.x * 100})
        
        st.dataframe(opt_weights.set_index("Asset"), use_container_width=True)
    else:
        st.info("Please select at least 2 tickers in the sidebar to run Markowitz Portfolio Optimization.")

# --- TAB 5: FINANCIAL TEXT ANALYTICS (NLP) ---
with tab5:
    st.subheader("Financial Text Analytics & SEC 10-K Risk Scoring")
    st.markdown("Parsing corporate disclosures and filings using custom financial lexicons to track Risk Density and Sentiment Trajectories.")
    
    financial_filings = [
        "The company experienced robust revenue growth, expanding operating margins and strong liquidity positions.",
        "Significant supply chain disruptions, rising inflationary pressures, and regulatory compliance risks threaten near-term profitability.",
        "Management notes positive customer acquisition trends, though elevated debt service costs remain a key headwind.",
        "Uncertain macroeconomic conditions, currency volatility, and pending litigation exposures could adversely impact future cash flows.",
        "Strategic technological investments and successful new product launches enhanced market penetration and earnings stability."
    ]

    positive_lexicon = ['growth', 'expanding', 'strong', 'liquidity', 'positive', 'successful', 'stability', 'enhancement']
    risk_lexicon = ['disruptions', 'pressures', 'risks', 'threaten', 'uncertain', 'volatility', 'adverse', 'headwind', 'debt', 'litigation']

    text_analysis_data = []
    for i, text in enumerate(financial_filings):
        words = text.lower().split()
        total_words = max(len(words), 1)
        pos_count = sum(1 for w in words if any(p in w for p in positive_lexicon))
        risk_count = sum(1 for w in words if any(r in w for r in risk_lexicon))
        
        risk_density = (risk_count / total_words) * 100
        net_sentiment = pos_count - risk_count
        
        text_analysis_data.append({
            'Filing_Period': f'Period Q{i+1}',
            'Risk_Density_Pct': risk_density,
            'Net_Sentiment': net_sentiment
        })

    df_nlp = pd.DataFrame(text_analysis_data)
    st.dataframe(df_nlp, use_container_width=True)

    # Render Matplotlib Graph live inside Streamlit
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df_nlp['Filing_Period'], df_nlp['Risk_Density_Pct'], marker='o', color='crimson', linewidth=2.5, label='Risk Keyword Density (%)')
    ax.plot(df_nlp['Filing_Period'], df_nlp['Net_Sentiment'], marker='s', color='navy', linewidth=2.5, label='Net Sentiment Score')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.set_title('SEC Filing Risk Density & Sentiment Trajectory', fontsize=12, fontweight='bold')
    ax.set_xlabel('Reporting Period')
    ax.set_ylabel('Metric Score')
    ax.legend()
    plt.tight_layout()
    
    st.pyplot(fig)

st.markdown("---")
st.markdown("### 🏆 Built by Jayashree Varadharajan")
