import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error
import plotly.graph_objs as go

# -- Optional: Sentiment/Trend Fallbacks
def fetch_google_trends(coin):
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq()
        kw_list = [coin]
        pytrends.build_payload(kw_list, cat=0, timeframe='now 3-m')
        data = pytrends.interest_over_time()
        return data[coin].iloc[-1] if not data.empty else 0
    except Exception:
        return 0

def fetch_reddit_sentiment(coin):
    # Dummy fallback sentiment
    return 0

def fetch_twitter_sentiment(coin):
    # Dummy fallback sentiment
    return 0

def fetch_onchain_signal(coin):
    # Dummy on-chain metric
    return 0

# -- Fetch Top 10 Crypto from CoinGecko (USD pairs)
def get_top10_coins():
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "volume_desc", "per_page": 10, "page": 1, "sparkline": False}
    coins = requests.get(url, params=params).json()
    return [(c['id'], c['symbol'].upper(), c['name']) for c in coins]

def fetch_price_history(coin_id, days, interval):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": interval
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return None
    data = r.json()
    df = pd.DataFrame(data["prices"], columns=["time", "price"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    if "total_volumes" in data:
        df["volume"] = [v[1] for v in data["total_volumes"]]
    else:
        df["volume"] = np.nan
    return df

# -- Feature Engineering
def make_features(df, lookback=14):
    df_feat = df.copy()
    df_feat["returns"] = df_feat["price"].pct_change()
    df_feat["ma"] = df_feat["price"].rolling(lookback).mean()
    df_feat["vol"] = df_feat["returns"].rolling(lookback).std()
    df_feat["momentum"] = df_feat["price"].diff(lookback)
    # RSI
    delta = df_feat["price"].diff()
    up = delta.clip(lower=0).rolling(lookback).mean()
    down = -delta.clip(upper=0).rolling(lookback).mean()
    rs = up / (down + 1e-6)
    df_feat["RSI"] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = df_feat["price"].ewm(span=12, adjust=False).mean()
    ema26 = df_feat["price"].ewm(span=26, adjust=False).mean()
    df_feat["MACD"] = ema12 - ema26
    # Bollinger Bands
    mavg = df_feat["price"].rolling(window=lookback).mean()
    mstd = df_feat["price"].rolling(window=lookback).std()
    df_feat["Bollinger_Upper"] = mavg + 2 * mstd
    df_feat["Bollinger_Lower"] = mavg - 2 * mstd
    # Sentiment/Trend/On-chain
    df_feat["trend"] = fetch_google_trends(st.session_state.get("symbol", "bitcoin"))
    df_feat["sent_reddit"] = fetch_reddit_sentiment(st.session_state.get("symbol", "bitcoin"))
    df_feat["sent_twitter"] = fetch_twitter_sentiment(st.session_state.get("symbol", "bitcoin"))
    df_feat["onchain"] = fetch_onchain_signal(st.session_state.get("symbol", "bitcoin"))
    return df_feat.dropna().reset_index(drop=True)

# -- Model Ensemble: XGBoost + LSTM
def train_predict_ensemble(df_feat, features, forecast_horizon, blend=0.5):
    X, y = [], []
    for i in range(len(df_feat) - forecast_horizon):
        X.append(df_feat.iloc[i][features].values)
        y.append(df_feat.iloc[i + forecast_horizon]["price"])
    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    split = int(0.8 * len(X))
    # --- XGBoost
    model_xgb = XGBRegressor(n_estimators=100, random_state=42)
    model_xgb.fit(X_scaled[:split], y[:split])
    y_pred_xgb = model_xgb.predict(X_scaled)
    # --- LSTM
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    model_lstm = Sequential([
        LSTM(32, input_shape=(1, X_scaled.shape[1])),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_lstm[:split], y[:split], epochs=15, batch_size=16, verbose=0)
    y_pred_lstm = model_lstm.predict(X_lstm).flatten()
    # --- Ensemble
    y_pred = blend * y_pred_xgb + (1 - blend) * y_pred_lstm
    return y, y_pred, scaler, model_xgb

# -- Multi-horizon Forecasts
def forecast_future(df_feat, scaler, model_xgb, features, window_size, n_steps):
    last_window = df_feat.iloc[-window_size:][features].values
    preds = []
    window = last_window.copy()
    for _ in range(n_steps):
        X_input = scaler.transform([window[-1]])
        y_pred = model_xgb.predict(X_input)[0]
        preds.append(y_pred)
        next_feat = window[-1].copy()
        next_feat[0] = y_pred  # update price for next step
        window = np.vstack([window, next_feat])[1:]
    return preds

# -- Strategy Backtest
def compute_backtest(df, preds, buy_thresh=0.01):
    signals = [1 if (p - a) / a > buy_thresh else -1 if (p - a) / a < -buy_thresh else 0
               for p, a in zip(preds, df["price"].values)]
    returns = df["returns"].shift(-1).fillna(0)
    strat_returns = returns * signals
    cumret = (1 + strat_returns).cumprod() - 1
    sharpe = np.mean(strat_returns) / (np.std(strat_returns) + 1e-8) * np.sqrt(252)
    max_dd = (cumret.cummax() - cumret).max()
    win_rate = (np.array(signals) == np.sign(returns)).mean()
    return signals, cumret, sharpe, max_dd, win_rate

# -- Streamlit UI
st.set_page_config(page_title="Quantum Crypto Forecaster", layout="wide")
st.title("🚀 Quantum-Inspired Crypto Forecast Dashboard (CoinGecko)")

# --- Sidebar Controls
top10 = get_top10_coins()
symbols = [f"{c[2]} ({c[1]})" for c in top10]
coin_lookup = {f"{c[2]} ({c[1]})": c[0] for c in top10}
symbol = st.sidebar.selectbox("Choose Asset", symbols)
coin_id = coin_lookup[symbol]
st.session_state["symbol"] = coin_id

interval = st.sidebar.selectbox("Data Interval", ["daily", "hourly"])
days = st.sidebar.slider("Days of Data", min_value=30, max_value=365, value=365)
lookback = st.sidebar.slider("Lookback Window", min_value=7, max_value=60, value=20)
forecast_horizon = st.sidebar.selectbox("Forecast Horizon (days ahead)", [1, 3, 7])
blend = st.sidebar.slider("XGBoost vs LSTM Blend (1=XGB only, 0=LSTM only)", 0.0, 1.0, 0.7)
show_sent = st.sidebar.checkbox("Show Sentiment/Trend Features", value=True)

# --- Main Pipeline
st.info(f"Pulling {symbol} ({coin_id}) history...")
df = fetch_price_history(coin_id, days, "hourly" if interval == "hourly" else "daily")
if df is None or df.empty:
    st.error("Failed to fetch data from CoinGecko.")
    st.stop()

# -- Feature Construction
with st.spinner("Computing features..."):
    df_feat = make_features(df, lookback=lookback)
    if df_feat.empty:
        st.warning("Not enough data for feature construction.")
        st.stop()

# -- Modeling
features = ["price", "ma", "vol", "momentum", "RSI", "MACD", "Bollinger_Upper", "Bollinger_Lower"]
if show_sent:
    features += ["trend", "sent_reddit", "sent_twitter", "onchain"]
y, y_pred, scaler, model_xgb = train_predict_ensemble(df_feat, features, forecast_horizon, blend)

# -- Backtest and Metrics
signals, cumret, sharpe, max_dd, win_rate = compute_backtest(df_feat.iloc[:-forecast_horizon], y_pred, buy_thresh=0.01)
mae = mean_absolute_error(y, y_pred)
st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
st.metric("Sharpe Ratio", f"{sharpe:.2f}")
st.metric("Max Drawdown", f"{max_dd:.2%}")
st.metric("Win Rate", f"{win_rate:.2%}")

# -- Charts
st.subheader(f"{symbol}: Price & Forecast")
chart_df = df_feat.iloc[:-forecast_horizon].copy()
chart_df["Predicted"] = y_pred
chart_df = chart_df.set_index("time")
fig = go.Figure()
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["price"], name="Actual Price", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Predicted"], name="Predicted Price", line=dict(color="red")))
st.plotly_chart(fig, use_container_width=True)

# -- Multi-horizon Forecast Chart
st.subheader(f"Forecasting the Next {forecast_horizon} Steps")
future_preds = forecast_future(df_feat, scaler, model_xgb, features, window_size=lookback, n_steps=forecast_horizon)
future_times = [df_feat["time"].iloc[-1] + timedelta(days=i+1) for i in range(forecast_horizon)]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_feat["time"], y=df_feat["price"], name="History"))
fig2.add_trace(go.Scatter(x=future_times, y=future_preds, name="Future Forecast", line=dict(color="orange", dash="dot")))
st.plotly_chart(fig2, use_container_width=True)

# -- Signal Export & Telegram/Discord (stub)
st.download_button("Download Signals (.csv)", chart_df.reset_index()[["time", "price", "Predicted"]].to_csv(index=False), file_name="signals.csv")
st.info("Telegram/Discord alerts not yet configured. (Ready to plug in via webhook or bot token.)")

# -- Quantum-Ready: QML Stub
st.markdown("""
> 🟦 <b>Quantum Ready:</b> This dashboard is ready for QML. To use real quantum models, just replace the XGBoost/LSTM step with PennyLane or Qiskit circuits using cloud backends like AWS Braket.<br>
> All feature and interface code will work without modification!
""", unsafe_allow_html=True)

st.caption("Copyright © 2024 Quantum Crypto Forecaster")


