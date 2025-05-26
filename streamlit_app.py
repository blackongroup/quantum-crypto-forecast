import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests

st.set_page_config(page_title="Crypto Predictor", layout="centered")

# --- Robust Feature Fetchers ---
def fetch_crypto_price_history(coin_id="bitcoin", days=365):
    try:
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
        df["returns"] = df["price"].pct_change().fillna(0)
        return df
    except Exception as e:
        st.warning(f"Could not fetch price data: {e}")
        return None

def get_trend_score(keyword):
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([keyword], timeframe='now 7-d')
        trends = pytrends.interest_over_time()
        if trends.empty:
            return 0
        return trends[keyword].iloc[-1]
    except Exception:
        return 0

def get_twitter_sentiment(keyword):
    try:
        # Placeholder: No Twitter API
        return 0
    except Exception:
        return 0

def compute_features(df, coin_name):
    df = df.copy()
    lookback = 14
    df["vol"] = df["returns"].rolling(lookback).std().fillna(0)
    df["ma"] = df["price"].rolling(lookback).mean().bfill()
    df["trend_score"] = get_trend_score(coin_name)
    df["sentiment"] = get_twitter_sentiment(coin_name)
    return df

# --- Streamlit UI ---

st.title("🚀 Crypto Price Predictor (Robust Chart)")
coin_id = st.selectbox("Choose a Crypto", [
    "bitcoin", "ethereum", "solana", "ripple", "cardano", "dogecoin", 
    "tron", "avalanche-2", "litecoin", "chainlink"
])
days = st.slider("Days of historical data", 30, 365, 365)
show_features = st.checkbox("Show features used in model", value=True)

with st.spinner("Fetching price history..."):
    df = fetch_crypto_price_history(coin_id, days)

if df is None or df.empty:
    st.error("Failed to fetch data from CoinGecko.")
    st.stop()

with st.spinner("Computing features..."):
    df_feat = compute_features(df, coin_id)

# Dummy prediction: Shift price by -1 (as placeholder)
df_feat["predicted"] = df_feat["price"].shift(-1)
df_feat = df_feat.dropna(subset=["price", "predicted"])

# --- Robust Chart Drawing ---
if not df_feat.empty and df_feat[["price", "predicted"]].notna().sum().min() > 0:
    st.subheader(f"Price Prediction for {coin_id.title()}")
    chart = pd.DataFrame({
        "Actual": df_feat["price"],
        "Predicted": df_feat["predicted"]
    })
    st.line_chart(chart)
else:
    st.warning("Not enough data for chart. Try a different coin, lower days, or wait for data to populate.")

if show_features:
    st.subheader("Model Features (Last 10 Rows)")
    st.dataframe(df_feat.tail(10))

st.caption("Sentiment and trend features are used if available. If unavailable, neutral values are used.")
