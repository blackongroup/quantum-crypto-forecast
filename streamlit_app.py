import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ----- UI Controls -----
st.title("🔮 Crypto Price Forecast: XGBoost Only (Quantum Ready)")
st.markdown(
    "Top 10 Coinbase cryptos. XGBoost-only for fast Streamlit Cloud. "
    "**NO files, NO local save, 365 days, forecast N days ahead.**"
)

# Controls
forecast_days = st.slider("Forecast horizon (days ahead)", 1, 7, 1)
lookback = st.slider("Lookback window (days)", 14, 90, 30)
coin = st.selectbox(
    "Crypto Pair (USD)", [
        "bitcoin",        # BTC-USD
        "ethereum",       # ETH-USD
        "solana",         # SOL-USD
        "dogecoin",       # DOGE-USD
        "avalanche-2",    # AVAX-USD
        "litecoin",       # LTC-USD
        "chainlink",      # LINK-USD
        "uniswap",        # UNI-USD
        "pepe",           # PEPE-USD
        "arbitrum"        # ARB-USD
    ],
    format_func=lambda x: x.replace("-", " ").upper()
)
st.caption("Powered by CoinGecko API (public rate limits apply).")

# ----- Data Fetch -----
@st.cache_data(show_spinner=True)
def fetch_coingecko_price(coin, days=365):
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
        f"?vs_currency=usd&days={days}&interval=daily"
    )
    resp = requests.get(url)
    if resp.status_code != 200:
        st.error("Failed to fetch price data.")
        return None
    prices = resp.json().get("prices", [])
    df = pd.DataFrame(prices, columns=["time", "close"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time")
    df["close"] = df["close"].astype(float)
    return df

df = fetch_coingecko_price(coin)
if df is None or len(df) < lookback + forecast_days + 5:
    st.warning("Not enough data to model. Choose a different coin or reduce lookback/horizon.")
    st.stop()

# ----- Feature Engineering -----
def make_features(df, lookback=30):
    df_feat = pd.DataFrame(index=df.index)
    df_feat["close"] = df["close"]
    df_feat["returns"] = np.log(df["close"] / df["close"].shift(1))
    df_feat["momentum"] = df["close"].pct_change(periods=lookback).fillna(0)
    df_feat["vol"] = df_feat["returns"].rolling(lookback).std().fillna(0)
    # RSI
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(lookback).mean()
    avg_loss = pd.Series(loss).rolling(lookback).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    df_feat["rsi"] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df_feat["macd"] = macd
    df_feat["macd_signal"] = signal
    df_feat = df_feat.dropna()
    return df_feat

df_feat = make_features(df, lookback=lookback)

# ----- XGBoost Model -----
def train_predict_xgb(df_feat, forecast_horizon=1):
    # Use all features except "close" as predictors
    X, y = [], []
    feat_cols = [c for c in df_feat.columns if c != "close"]
    data = df_feat.copy()
    data = data.dropna()
    for i in range(len(data) - forecast_horizon):
        X.append(data[feat_cols].iloc[i].values)
        y.append(data["close"].iloc[i + forecast_horizon])
    X, y = np.array(X), np.array(y)
    if len(X) < 30:
        return np.full(len(df_feat), np.nan)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_scaled, y)
    # For plotting, predict for all (align predictions forward)
    X_pred = scaler.transform(data[feat_cols].values)
    preds = model.predict(X_pred)
    preds_full = np.full(len(df_feat), np.nan)
    preds_full[forecast_horizon:len(preds)+forecast_horizon] = preds
    return preds_full

preds_xgb = train_predict_xgb(df_feat, forecast_horizon=forecast_days)

# ----- Plot -----
chart_df = pd.DataFrame({
    "Actual": df_feat["close"],
    f"Pred_XGB_{forecast_days}d": preds_xgb
}, index=df_feat.index)

st.line_chart(chart_df)

# Download Data Option
csv = chart_df.to_csv(index=True)
st.download_button(
    "Download Data as CSV", csv, "crypto_forecast.csv", "text/csv"
)
