import streamlit as st
import pandas as pd
import numpy as np
import requests
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# --- 1. CONFIG ---

st.set_page_config("XGBoost Crypto Predictor", layout="wide")

COIN_OPTIONS = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Solana": "solana",
    "Dogecoin": "dogecoin",
    "Litecoin": "litecoin",
    "Cardano": "cardano",
    "Avalanche": "avalanche-2",
    "Chainlink": "chainlink",
    "Polygon": "polygon",
    "XRP": "ripple",
}
DEFAULT_COIN = "bitcoin"

# --- 2. FETCH DATA ---

@st.cache_data(show_spinner=False)
def fetch_coingecko_price(coin_id, days=365):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return None
    prices = resp.json().get("prices", [])
    if not prices:
        return None
    df = pd.DataFrame(prices, columns=["time", "close"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time").resample("D").last().ffill().reset_index()
    return df

# --- 3. FEATURES ---

def make_features(df, lookback=14):
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["ma"] = df["close"].rolling(lookback).mean()
    df["vol"] = df["returns"].rolling(lookback).std()
    df["momentum"] = df["close"] - df["close"].shift(lookback)
    df = df.dropna()
    return df

# --- 4. DASHBOARD CONTROLS ---

st.title("🚀 Crypto Next-Price Predictor (XGBoost)")
with st.sidebar:
    coin_name = st.selectbox("Crypto Coin", list(COIN_OPTIONS.keys()), index=0)
    lookback = st.slider("Lookback Window (days)", min_value=5, max_value=60, value=14, step=1)
    forecast_days = st.slider("Forecast Horizon (days ahead)", min_value=1, max_value=7, value=1, step=1)
    days = st.slider("Days of Data to Fetch", min_value=120, max_value=1095, value=365, step=1)
    show_pred_chart = st.checkbox("Show Prediction Chart", value=True)

# --- 5. LOAD & PROCESS DATA ---

coin_id = COIN_OPTIONS[coin_name]
df = fetch_coingecko_price(coin_id, days=days)

if df is None or len(df) < (lookback + forecast_days + 10):
    st.warning("Not enough data to model. Choose a different coin or reduce lookback/horizon.")
    st.stop()

df_feat = make_features(df, lookback=lookback)
# Prepare supervised learning (X = features, y = next close)
df_feat["target"] = df_feat["close"].shift(-forecast_days)
df_feat = df_feat.dropna().reset_index(drop=True)

if len(df_feat) < 40:
    st.warning("Not enough samples after feature engineering. Try smaller lookback/horizon or a different coin.")
    st.stop()

FEATURES = ["close", "ma", "vol", "momentum"]

X = df_feat[FEATURES].values
y = df_feat["target"].values

# --- 6. TRAIN/VALIDATE MODEL ---

split_idx = int(len(df_feat)*0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

df_feat["pred"] = model.predict(X)

mae = mean_absolute_error(y_test, model.predict(X_test))

# --- 7. SHOW RESULTS ---

st.subheader(f"{coin_name} — MAE: ${mae:,.2f}")

if show_pred_chart:
    plot_df = df_feat[["time", "close", "pred"]].copy()
    plot_df = plot_df.set_index("time")
    # Make the predicted line red, actual blue
    st.line_chart(plot_df.rename(columns={"close": "Actual Close", "pred": "Predicted Next Close"}))

# --- 8. FUTURE PRICE FORECAST ---

st.markdown("### Forecast Next Price(s)")
with st.expander("Show forecast for next N days", expanded=True):
    future_window = df_feat.iloc[-1:][FEATURES].values
    preds = []
    last_vals = df_feat.iloc[-1].copy()
    for step in range(forecast_days):
        next_pred = model.predict(future_window)[0]
        preds.append(next_pred)
        # Roll window forward: simulate the next close
        last_vals["close"] = next_pred
        last_vals["ma"] = (last_vals["ma"]*(lookback-1) + next_pred)/lookback
        last_vals["momentum"] = next_pred - last_vals["close"]
        # vol is not advanced for simplicity
        future_window = last_vals[FEATURES].values.reshape(1, -1)
    st.write(f"**Forecast for next {forecast_days} day(s):**")
    st.write([f"${p:,.2f}" for p in preds])

st.caption("100% cloud/streamlit. No files or local dependencies.")

