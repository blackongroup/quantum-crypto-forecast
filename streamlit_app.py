import streamlit as st
import pandas as pd
import numpy as np
import requests
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Quantum-Ready Crypto Forecaster", layout="wide")

# ----- SETTINGS -----
DEFAULT_TOP10 = [
    "bitcoin", "ethereum", "solana", "ripple", "dogecoin",
    "cardano", "avalanche-2", "tron", "polkadot", "chainlink"
]
COINGECKO_SYMBOLS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "ripple": "XRP",
    "dogecoin": "DOGE",
    "cardano": "ADA",
    "avalanche-2": "AVAX",
    "tron": "TRX",
    "polkadot": "DOT",
    "chainlink": "LINK",
}

# ----- HELPERS -----
@st.cache_data(show_spinner=False)
def fetch_price_history(coin_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    r = requests.get(url, params=params)
    data = r.json()
    if "prices" not in data:
        return pd.DataFrame()
    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "close"])
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("time", inplace=True)
    df = df.resample("1D").last()
    df["close"] = df["close"].ffill()
    return df[["close"]]

def make_features(df, lookback=14):
    df_feat = pd.DataFrame(index=df.index)
    df_feat["close"] = df["close"]
    df_feat["returns"] = df["close"].pct_change().fillna(0)
    df_feat["logret"] = np.log(df["close"]).diff().fillna(0)
    df_feat["sma"] = df["close"].rolling(lookback).mean().fillna(method="bfill")
    df_feat["vol"] = df["returns"].rolling(lookback).std().fillna(0)
    df_feat["momentum"] = df["close"].diff(lookback).fillna(0)
    # RSI
    delta = df["close"].diff().fillna(0)
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(lookback).mean()
    roll_down = down.rolling(lookback).mean()
    rs = roll_up / (roll_down + 1e-8)
    df_feat["rsi"] = 100 - 100 / (1 + rs)
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df_feat["macd"] = ema12 - ema26
    df_feat = df_feat.dropna()
    return df_feat

def train_predict_xgb(df_feat, forecast_horizon=1):
    X, y = [], []
    for i in range(len(df_feat) - forecast_horizon):
        X.append(df_feat.iloc[i].values)
        y.append(df_feat["close"].iloc[i + forecast_horizon])
    X, y = np.array(X), np.array(y)
    if len(X) < 20:
        return np.zeros_like(df_feat["close"]), np.zeros_like(df_feat["close"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X_scaled, y)
    X_pred = scaler.transform(df_feat.values)
    preds = model.predict(X_pred)
    # Pad so alignment is correct
    preds_full = np.full(len(df_feat), np.nan)
    preds_full[:-forecast_horizon] = preds[forecast_horizon:]
    return preds_full, y

# ----- UI -----
st.title("🚀 Quantum-Ready Crypto Prediction Dashboard")
st.caption("Top 10 most traded cryptos, adjustable lookback, risk, and forecast horizon. Model: XGBoost, QML-ready. Data: CoinGecko.")

with st.sidebar:
    coin = st.selectbox("Select Crypto", DEFAULT_TOP10, format_func=lambda x: f"{COINGECKO_SYMBOLS[x]} ({x})")
    days = st.slider("Days of Data", 60, 365, 365, step=5)
    lookback = st.slider("Feature Lookback Window", 7, 60, 14)
    risk = st.slider("Risk Multiplier (affects model)", 0.1, 3.0, 1.0, 0.05)
    forecast = st.slider("Forecast Horizon (days ahead)", 1, 7, 1)

st.write(f"Showing **{COINGECKO_SYMBOLS[coin]}** for last **{days} days** (CoinGecko, daily).")

df = fetch_price_history(coin, days)
if df.empty or len(df) < lookback + 10:
    st.error("Insufficient data to model. Try a different coin or fewer days.")
    st.stop()

df_feat = make_features(df, lookback=lookback)
preds, actuals = train_predict_xgb(df_feat * risk, forecast_horizon=forecast)

plot_df = pd.DataFrame({
    "Actual": df_feat["close"],
    "Predicted (Next Close)": preds,
})
plot_df.index.name = "Date"

st.line_chart(plot_df, height=420, use_container_width=True)
st.dataframe(plot_df.tail(20), use_container_width=True)

st.markdown(
    """
    **How it works:**  
    - Model: XGBoost (can swap to QML!)  
    - Features: Momentum, Vol, RSI, MACD, SMA, Log-Returns  
    - Forecast: Next N closes, with dashboard controls  
    - Data: CoinGecko, auto cloud fetch (no local files)
    """
)

st.caption("Want quantum or ensemble? Let me know when you have a quantum backend or want ensemble mode (LSTM, QML, etc).")
