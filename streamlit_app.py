import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- CONFIG ---
TOP10 = [
    "bitcoin", "ethereum", "solana", "ripple", "dogecoin",
    "cardano", "avalanche-2", "tron", "polkadot", "chainlink"
]
SYMBOLS = {
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
    returns = df["close"].pct_change().fillna(0)
    logret = np.log(df["close"]).diff().fillna(0)
    df_feat["returns"] = returns
    df_feat["logret"] = logret
    df_feat["sma"] = df["close"].rolling(lookback).mean().bfill()
    df_feat["vol"] = returns.rolling(lookback).std().fillna(0)
    df_feat["momentum"] = df["close"].diff(lookback).fillna(0)
    delta = df["close"].diff().fillna(0)
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(lookback).mean()
    roll_down = down.rolling(lookback).mean()
    rs = roll_up / (roll_down + 1e-8)
    df_feat["rsi"] = 100 - 100 / (1 + rs)
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
    if len(X) < 30:
        return np.full(len(df_feat), np.nan)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X_scaled, y)
    X_pred = scaler.transform(df_feat.values)
    preds = model.predict(X_pred)
    preds_full = np.full(len(df_feat), np.nan)
    preds_full[:-forecast_horizon] = preds[forecast_horizon:]
    return preds_full

def train_predict_lstm(df_feat, lookback=14, forecast_horizon=1):
    # Only use close for LSTM
    close = df_feat["close"].values.reshape(-1, 1)
    scaler = StandardScaler()
    close_scaled = scaler.fit_transform(close)
    X, y = [], []
    for i in range(len(close_scaled) - lookback - forecast_horizon):
        X.append(close_scaled[i:i + lookback, 0])
        y.append(close_scaled[i + lookback + forecast_horizon - 1, 0])
    X, y = np.array(X), np.array(y)
    if len(X) < 30:
        return np.full(len(df_feat), np.nan)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(32, input_shape=(lookback, 1), return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=20, batch_size=8, verbose=0)
    preds = model.predict(np.array([
        close_scaled[i - lookback + 1:i + 1, 0] if i - lookback + 1 >= 0 else np.zeros(lookback)
        for i in range(lookback - 1, len(close_scaled))
    ]).reshape(-1, lookback, 1), verbose=0)
    preds_rescaled = scaler.inverse_transform(np.concatenate([preds, np.zeros((preds.shape[0], 0))], axis=1))[:, 0]
    preds_full = np.full(len(df_feat), np.nan)
    preds_full[lookback-1:] = preds_rescaled
    return preds_full

st.title("Quantum-Ready Crypto Ensemble Predictor")
st.caption("Top traded coins, ensemble XGBoost+LSTM (tunable blend), CoinGecko data. No files, fully cloud. QML ready.")

with st.sidebar:
    coin = st.selectbox("Crypto", TOP10, format_func=lambda x: f"{SYMBOLS[x]} ({x})")
    days = st.slider("Days of Data", 90, 365, 365, step=7)
    lookback = st.slider("Feature Lookback Window", 7, 60, 14)
    forecast = st.slider("Forecast Horizon (days)", 1, 7, 1)
    blend = st.slider("Ensemble Blend (0 = XGB, 1 = LSTM)", 0.0, 1.0, 0.5, 0.05)

st.write(f"**{SYMBOLS[coin]}**, {days} days, blend={blend:.2f}")

df = fetch_price_history(coin, days)
if df.empty or len(df) < lookback + 30:
    st.error("Insufficient data. Try different coin or parameters.")
    st.stop()
df_feat = make_features(df, lookback=lookback)
xgb_pred = train_predict_xgb(df_feat, forecast_horizon=forecast)
lstm_pred = train_predict_lstm(df_feat, lookback=lookback, forecast_horizon=forecast)

ensemble_pred = (1 - blend) * xgb_pred + blend * lstm_pred

plot_df = pd.DataFrame({
    "Actual": df_feat["close"],
    "Pred_XGB": xgb_pred,
    "Pred_LSTM": lstm_pred,
    "Ensemble": ensemble_pred,
})
plot_df.index.name = "Date"

st.line_chart(plot_df, height=420, use_container_width=True)
st.dataframe(plot_df.tail(20), use_container_width=True)
st.markdown(
    """
    **How it works:**
    - XGBoost (tree) and LSTM (deep) both predict next prices, then blend.
    - Adjust blend for optimal forecast (XGB: stable, LSTM: catches fast moves).
    - Features: log returns, SMA, momentum, RSI, MACD, volatility.
    - All fully cloud-based (no files).  
    - Quantum-ML drop-in: ready for PennyLane/AWS Braket integration.
    """
)
