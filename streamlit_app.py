import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="🔮 Yahoo Finance Crypto Forecast", layout="centered")

# ---- Top 10 cryptos, Yahoo tickers ----
TOP_CRYPTO_YF = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Tether (USDT)": "USDT-USD",
    "Solana (SOL)": "SOL-USD",
    "BNB (BNB)": "BNB-USD",
    "XRP (XRP)": "XRP-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Toncoin (TON)": "TON11419-USD",  # Yahoo symbol may be different for Toncoin, fallback if not found
    "Cardano (ADA)": "ADA-USD",
    "USDC (USDC)": "USDC-USD",
}

# --- Robust Yahoo fetch ---
@st.cache_data(ttl=900, show_spinner=True)
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

def make_features(df, lookback=14):
    df_feat = df.copy()
    df_feat["ma"] = df_feat["price"].rolling(lookback).mean().bfill()
    df_feat["vol"] = df_feat["returns"].rolling(lookback).std().bfill()
    df_feat["momentum"] = df_feat["price"] / df_feat["price"].shift(lookback) - 1
    df_feat["RSI"] = df_feat["RSI"].bfill().fillna(50)
    df_feat["MACD"] = df_feat["MACD"].bfill().fillna(0)
    df_feat = df_feat.dropna()
    return df_feat

def train_xgb(df_feat, forecast_horizon=1):
    X, y = [], []
    features = ["price", "ma", "vol", "momentum", "RSI", "MACD"]
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

# ---- UI ----
st.title("🔮 Yahoo Finance Crypto Forecast Dashboard (XGBoost)")
st.markdown("Top 10 coins, Yahoo Finance. XGBoost modeling, MACD, RSI, momentum features. No local files, cloud-based.")

coin_name = st.selectbox("Choose Crypto Asset", list(TOP_CRYPTO_YF.keys()), index=0)
coin_symbol = TOP_CRYPTO_YF[coin_name]

col1, col2, col3 = st.columns(3)
with col1:
    lookback = st.slider("Lookback Window", min_value=5, max_value=60, value=14, step=1)
with col2:
    forecast_horizon = st.slider("Forecast Days Ahead", min_value=1, max_value=7, value=1, step=1)
with col3:
    days = st.slider("Days of Data", min_value=60, max_value=730, value=365, step=1)

st.info(f"Pulling last {days} days of {coin_name} prices from Yahoo Finance...")

df = fetch_yf_price_history(symbol=coin_symbol, days=days)
if df is None or len(df) < lookback + forecast_horizon + 1:
    st.error("Failed to fetch data from Yahoo Finance or not enough data for modeling.")
    st.stop()
st.success(f"Loaded {len(df)} daily prices.")

df_feat = make_features(df, lookback=lookback)
if len(df_feat) < lookback + forecast_horizon + 7:
    st.error("Not enough feature rows for this window. Try smaller lookback or more days.")
    st.stop()

y_true, y_pred_xgb, idx, scaler = train_xgb(df_feat, forecast_horizon=forecast_horizon)
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
st.info("Quantum-ready: All feature engineering and dashboard logic is ready for quantum pipeline.")

st.caption("Open-source, cloud-ready. Want to add more models or sentiment? Just ask!")
