import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from xgboost import XGBRegressor
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto Forecast (XGBoost)", layout="centered")

# ---- Config ----
TOP_COINS = {
    "Bitcoin (BTC)": "bitcoin",
    "Ethereum (ETH)": "ethereum",
    "Tether (USDT)": "tether",
    "Solana (SOL)": "solana",
    "USDC (USDC)": "usd-coin",
    "BNB (BNB)": "binancecoin",
    "XRP (XRP)": "ripple",
    "Dogecoin (DOGE)": "dogecoin",
    "Toncoin (TON)": "the-open-network",
    "Cardano (ADA)": "cardano",
}

# ---- Data Fetch ----
@st.cache_data(ttl=900, show_spinner=True)
def fetch_crypto_price_history(coin_id="bitcoin", days=365):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 200:
                prices = r.json().get("prices", [])
                if not prices or len(prices) < 30:
                    return None
                df = pd.DataFrame(prices, columns=["time", "price"])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                df = df.set_index("time")
                df["returns"] = df["price"].pct_change().fillna(0)
                return df
            elif r.status_code == 429:
                time.sleep(3)
            else:
                time.sleep(2)
        except Exception as e:
            time.sleep(2)
    return None

# ---- Feature Engineering ----
def make_features(df, lookback=14):
    df_feat = df.copy()
    df_feat["ma"] = df_feat["price"].rolling(lookback).mean().fillna(method="bfill")
    df_feat["vol"] = df_feat["returns"].rolling(lookback).std().fillna(method="bfill")
    df_feat["momentum"] = df_feat["price"] / df_feat["price"].shift(lookback) - 1
    df_feat["momentum"] = df_feat["momentum"].fillna(0)
    df_feat = df_feat.dropna()
    return df_feat

def train_xgb_and_predict(df_feat, forecast_horizon=1):
    # Prepare dataset
    X, y = [], []
    for i in range(len(df_feat) - forecast_horizon):
        X.append(df_feat.iloc[i][["price", "ma", "vol", "momentum"]].values)
        y.append(df_feat.iloc[i + forecast_horizon]["price"])
    if not X or not y:
        return None, None, None
    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X[:split], y[:split])
    y_pred = model.predict(X)
    return y, y_pred, df_feat.index[forecast_horizon:]

# ---- UI ----
st.title("🔮 Quantum-Ready Crypto Forecast Dashboard")
st.markdown("Minimal XGBoost, CoinGecko data, top 10 assets. No local files, all cloud-based.")

coin_name = st.selectbox("Choose Crypto Asset", list(TOP_COINS.keys()), index=0)
coin_id = TOP_COINS[coin_name]

col1, col2, col3 = st.columns(3)
with col1:
    lookback = st.slider("Lookback Window", min_value=5, max_value=60, value=14, step=1)
with col2:
    forecast_horizon = st.slider("Forecast Days Ahead", min_value=1, max_value=7, value=1, step=1)
with col3:
    days = st.slider("Days of Data", min_value=60, max_value=730, value=365, step=1)

st.info(f"Pulling last {days} days of {coin_name} prices from CoinGecko...")

df = fetch_crypto_price_history(coin_id=coin_id, days=days)
if df is None or len(df) < lookback + forecast_horizon + 1:
    st.error("Failed to fetch data from CoinGecko or not enough data for modeling.")
    st.stop()

st.success(f"Loaded {len(df)} daily prices.")

df_feat = make_features(df, lookback=lookback)
if len(df_feat) < lookback + forecast_horizon + 1:
    st.error("Not enough feature rows for this window. Try smaller lookback or more days.")
    st.stop()

y_true, y_pred, idx = train_xgb_and_predict(df_feat, forecast_horizon=forecast_horizon)
if y_true is None or y_pred is None:
    st.error("Prediction could not be computed. Try other settings.")
    st.stop()

chart_df = pd.DataFrame({
    "Actual Price": y_true,
    "Predicted Next Price": y_pred
}, index=idx)

st.subheader("Actual vs. Predicted (Next Price)")
st.line_chart(chart_df)

st.caption("MAE (mean absolute error): {:.2f}".format(np.mean(np.abs(y_true - y_pred))))
