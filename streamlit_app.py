import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests

st.set_page_config(page_title="Crypto Predictor with Sentiment & Trends", layout="centered")

# --------- Helper Functions ---------

@st.cache_data(show_spinner=False)
def fetch_crypto_price_history(coin_id="bitcoin", days=365):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "1d"
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return None
    prices = r.json().get("prices", [])
    df = pd.DataFrame(prices, columns=["time", "price"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time")
    df["returns"] = df["price"].pct_change()
    return df

def get_trend_score(keyword):
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([keyword], timeframe='now 7-d')
        trends = pytrends.interest_over_time()
        if trends.empty:
            return 0
        return trends[keyword].iloc[-1]  # Use last value
    except Exception as e:
        st.info(f"(Trends unavailable: {e})")
        return 0

def get_twitter_sentiment(keyword):
    try:
        # To actually use Twitter/X, you need your API key and tweepy.
        # Here's a dummy placeholder; Twitter API now requires paid access.
        # For real sentiment, you could use StockTwits or other public APIs.
        raise NotImplementedError("Twitter API now requires paid access.")
    except Exception as e:
        st.info(f"(Sentiment unavailable: {e})")
        return 0

def compute_features(df, coin_name):
    df = df.copy()
    lookback = 14
    df["vol"] = df["returns"].rolling(lookback).std().fillna(0)
    df["ma"] = df["price"].rolling(lookback).mean().fillna(method='bfill')
    df["trend_score"] = get_trend_score(coin_name)
    df["sentiment"] = get_twitter_sentiment(coin_name)
    return df

# --------- UI ---------

st.title("🚀 Crypto Price Predictor (Sentiment & Trend Enhanced)")
coin_id = st.selectbox("Choose a Crypto", [
    "bitcoin", "ethereum", "solana", "ripple", "cardano", "dogecoin", 
    "tron", "avalanche-2", "litecoin", "chainlink"
])
days = st.slider("Days of historical data", 30, 365, 365)
show_features = st.checkbox("Show features used in model", value=True)

# --------- Data Fetch ---------

with st.spinner("Fetching price history..."):
    df = fetch_crypto_price_history(coin_id, days)

if df is None or df.empty:
    st.error("Failed to fetch data from CoinGecko.")
    st.stop()

with st.spinner("Computing features..."):
    df_feat = compute_features(df, coin_id)

# --------- Prediction Logic (Dummy Example) ---------
# Here, you would replace with your XGBoost/LSTM/QML or other logic.
df_feat["predicted"] = df_feat["price"].shift(-1)  # Naive next-day prediction

# --------- Charts ---------

st.subheader(f"Price Prediction for {coin_id.title()}")
chart = pd.DataFrame({
    "Actual": df_feat["price"],
    "Predicted": df_feat["predicted"]
})
st.line_chart(chart)

if show_features:
    st.subheader("Model Features (Last 10 Rows)")
    st.dataframe(df_feat.tail(10))

st.caption("Sentiment and trend features are used if available. If unavailable, neutral values are used.")

