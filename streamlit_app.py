import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import requests

st.set_page_config(page_title="🔮 Crypto + News/Trends Predictor", layout="centered")

# ---- Top 10 cryptos, Yahoo tickers ----
TOP_CRYPTO_YF = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Tether (USDT)": "USDT-USD",
    "Solana (SOL)": "SOL-USD",
    "BNB (BNB)": "BNB-USD",
    "XRP (XRP)": "XRP-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Cardano (ADA)": "ADA-USD",
    "Toncoin (TON)": "TON11419-USD",  # Try this or remove if it fails
    "USDC (USDC)": "USDC-USD",
}

def fetch_yf_price_history(symbol="BTC-USD", days=365):
    try:
        df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
        if df is None or df.empty or len(df) < 30:
            return None
        df = df.rename(columns={"Close": "price"})
        df = df[["price"]].copy()
        df["returns"] = df["price"].pct_change().fillna(0)
        # RSI
        delta = df["price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        RS = gain / loss
        df["RSI"] = 100 - (100 / (1 + RS))
        exp12 = df["price"].ewm(span=12, adjust=False).mean()
        exp26 = df["price"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp12 - exp26
        return df.dropna()
    except Exception as e:
        st.error(f"Failed to fetch data from Yahoo Finance: {e}")
        return None

# ---- Google Trends ----
def fetch_google_trends_score(keyword):
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([keyword], cat=0, timeframe='today 3-m')
        data = pytrends.interest_over_time()
        if not data.empty:
            return float(data[keyword].iloc[-1])
    except Exception as e:
        st.warning(f"Google Trends not available: {e}")
    return None

# ---- News Sentiment (Simple headlines polarity with requests + TextBlob fallback) ----
def fetch_news_sentiment(coin):
    try:
        from newspaper import Article
        from textblob import TextBlob
    except ImportError:
        st.warning("Sentiment analysis packages not installed.")
        return None
    try:
        url = f"https://news.google.com/rss/search?q={coin}+crypto"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return None
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        titles = [item.find('title').text for item in root.findall(".//item")]
        sentiments = []
        for title in titles:
            blob = TextBlob(title)
            sentiments.append(blob.sentiment.polarity)
        if sentiments:
            return float(np.mean(sentiments))
    except Exception as e:
        st.warning(f"News sentiment not available: {e}")
    return None

def make_features(df, lookback=14, trend=None, sentiment=None):
    df_feat = df.copy()
    df_feat["ma"] = df_feat["price"].rolling(lookback).mean().bfill()
    df_feat["vol"] = df_feat["returns"].rolling(lookback).std().bfill()
    df_feat["momentum"] = df_feat["price"] / df_feat["price"].shift(lookback) - 1
    df_feat["RSI"] = df_feat["RSI"].bfill().fillna(50)
    df_feat["MACD"] = df_feat["MACD"].bfill().fillna(0)
    # Optionally add trends/sentiment
    if trend is not None:
        df_feat["trend"] = trend
    if sentiment is not None:
        df_feat["sentiment"] = sentiment
    df_feat = df_feat.dropna()
    return df_feat

def train_xgb(df_feat, forecast_horizon=1, extra_features=[]):
    features = ["price", "ma", "vol", "momentum", "RSI", "MACD"] + extra_features
    X, y = [], []
    for i in range(len(df_feat) - forecast_horizon):
        X.append(df_feat.iloc[i][features].values)
        y.append(df_feat.iloc[i + forecast_horizon]["price"])
    if not X or not y:
        return None, None, None, None
    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    split = int(0.8 * len(X))
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled[:split], y[:split])
    y_pred = model.predict(X_scaled)
    return y, y_pred, df_feat.index[forecast_horizon:], scaler

# --- UI ---
st.title("🔮 Crypto Forecast: Yahoo Finance + News/Trends (XGBoost)")
st.markdown("""
Top 10 coins, Yahoo Finance data. XGBoost modeling with MACD, RSI, momentum features. Optionally includes news sentiment & Google Trends for extra alpha.
""")

coin_name = st.selectbox("Choose Crypto Asset", list(TOP_CRYPTO_YF.keys()), index=0)
coin_symbol = TOP_CRYPTO_YF[coin_name]

col1, col2, col3 = st.columns(3)
with col1:
    lookback = st.slider("Lookback Window", min_value=5, max_value=60, value=14, step=1)
with col2:
    forecast_horizon = st.slider("Forecast Days Ahead", min_value=1, max_value=7, value=1, step=1)
with col3:
    days = st.slider("Days of Data", min_value=60, max_value=730, value=365, step=1)

use_trend = st.checkbox("Include Google Trends score", value=True)
use_sentiment = st.checkbox("Include News Sentiment", value=True)

st.info(f"Pulling last {days} days of {coin_name} prices from Yahoo Finance...")

df = fetch_yf_price_history(symbol=coin_symbol, days=days)
if df is None or len(df) < lookback + forecast_horizon + 1:
    st.error("Failed to fetch data from Yahoo Finance or not enough data for modeling.")
    st.stop()
st.success(f"Loaded {len(df)} daily prices.")

trend, sentiment = None, None
extra_features = []
if use_trend:
    with st.spinner("Fetching Google Trends..."):
        trend = fetch_google_trends_score(coin_name.split()[0])
        if trend is not None:
            st.write(f"Google Trends score: {trend:.2f}")
            extra_features.append("trend")
if use_sentiment:
    with st.spinner("Fetching news sentiment..."):
        sentiment = fetch_news_sentiment(coin_name.split()[0])
        if sentiment is not None:
            st.write(f"News sentiment score: {sentiment:.2f}")
            extra_features.append("sentiment")

df_feat = make_features(df, lookback=lookback, trend=trend, sentiment=sentiment)
if len(df_feat) < lookback + forecast_horizon + 7:
    st.error("Not enough feature rows for this window. Try smaller lookback or more days.")
    st.stop()

y_true, y_pred_xgb, idx, scaler = train_xgb(df_feat, forecast_horizon=forecast_horizon, extra_features=extra_features)
if y_true is None or y_pred_xgb is None:
    st.error("Prediction could not be computed. Try other settings.")
    st.stop()

chart_df = pd.DataFrame({
    "Actual Price": y_true,
    "Predicted Next Price": y_pred_xgb
}, index=idx)

st.subheader("Actual vs. Predicted (Next Price)")
st.line_chart(chart_df)

st.caption("MAE (mean absolute error): {:.2f}".format(np.mean(np.abs(y_true - y_pred_xgb))))
st.info("Quantum-ready: Swap in your QML pipeline as needed! Open-source, cloud-ready.")

st.caption("Add more features? Want news/trends for forex? Just ask!")

