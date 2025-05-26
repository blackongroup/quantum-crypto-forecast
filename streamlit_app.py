import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
import plotly.graph_objs as go

# 1️⃣ Top 10 Crypto Tickers (by volume, as of 2024-05, can update this list)
TOP10_CRYPTO_YF = [
    ("BTC-USD", "Bitcoin"),
    ("ETH-USD", "Ethereum"),
    ("USDT-USD", "Tether"),
    ("SOL-USD", "Solana"),
    ("DOGE-USD", "Dogecoin"),
    ("BNB-USD", "BNB"),
    ("XRP-USD", "XRP"),
    ("TON-USD", "Toncoin"),
    ("ADA-USD", "Cardano"),
    ("SHIB-USD", "Shiba Inu"),
]

# 2️⃣ Sentiment/trend stubs (optional)
def fetch_google_trends(symbol):
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq()
        pytrends.build_payload([symbol], cat=0, timeframe='now 3-m')
        df = pytrends.interest_over_time()
        return float(df[symbol].iloc[-1]) if not df.empty else 0.0
    except Exception:
        return 0.0

def fetch_reddit_sentiment(symbol): return 0.0
def fetch_twitter_sentiment(symbol): return 0.0
def fetch_onchain_signal(symbol): return 0.0

# 3️⃣ Data loader from Yahoo Finance
@st.cache_data
def fetch_price_history_yahoo(ticker, days, interval):
    period = f"{days}d" if interval == "1d" else f"{days*24}h"
    data = yf.download(tickers=ticker, period=period, interval=interval, progress=False)
    if data.empty: return None
    df = data.reset_index()
    df = df.rename(columns={"Date": "time", "Close": "price", "Volume": "volume"})
    df = df[["time", "price", "volume"]]
    return df

# 4️⃣ Feature engineering
def make_features(df, lookback=14, symbol="BTC-USD", show_sent=True):
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
    if show_sent:
        df_feat["trend"] = fetch_google_trends(symbol)
        df_feat["sent_reddit"] = fetch_reddit_sentiment(symbol)
        df_feat["sent_twitter"] = fetch_twitter_sentiment(symbol)
        df_feat["onchain"] = fetch_onchain_signal(symbol)
    else:
        df_feat["trend"] = df_feat["sent_reddit"] = df_feat["sent_twitter"] = df_feat["onchain"] = 0.0
    return df_feat.dropna().reset_index(drop=True)

# 5️⃣ Model: XGBoost + LSTM
def train_predict_ensemble(df_feat, features, forecast_horizon, blend=0.7):
    X, y = [], []
    for i in range(len(df_feat) - forecast_horizon):
        X.append(df_feat.iloc[i][features].values)
        y.append(df_feat.iloc[i + forecast_horizon]["price"])
    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    split = int(0.8 * len(X))
    # XGBoost
    model_xgb = XGBRegressor(n_estimators=100, random_state=42)
    model_xgb.fit(X_scaled[:split], y[:split])
    y_pred_xgb = model_xgb.predict(X_scaled)
    # LSTM
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    model_lstm = Sequential([
        LSTM(32, input_shape=(1, X_scaled.shape[1])),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_lstm[:split], y[:split], epochs=15, batch_size=16, verbose=0)
    y_pred_lstm = model_lstm.predict(X_lstm).flatten()
    # Blend
    y_pred = blend * y_pred_xgb + (1 - blend) * y_pred_lstm
    return y, y_pred, scaler, model_xgb

# 6️⃣ Multi-horizon forecast
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

# 7️⃣ Backtest
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

# 8️⃣ Streamlit UI
st.set_page_config(page_title="Quantum Crypto Forecaster (Yahoo)", layout="wide")
st.title("🚀 Quantum Crypto Forecast Dashboard (Yahoo Finance)")

# Sidebar controls
symbols = [f"{name} ({ticker})" for ticker, name in TOP10_CRYPTO_YF]
symbol_choice = st.sidebar.selectbox("Choose Asset", symbols)
symbol, ticker = symbol_choice.split(" (")[0], symbol_choice.split("(")[-1][:-1]
interval = st.sidebar.selectbox("Data Interval", ["1d", "1h"])
days = st.sidebar.slider("Days of Data", min_value=30, max_value=365, value=365)
lookback = st.sidebar.slider("Lookback Window", min_value=7, max_value=60, value=20)
forecast_horizon = st.sidebar.selectbox("Forecast Horizon (days ahead)", [1, 3, 7])
blend = st.sidebar.slider("XGBoost vs LSTM Blend (1=XGB only, 0=LSTM only)", 0.0, 1.0, 0.7)
show_sent = st.sidebar.checkbox("Show Sentiment/Trend Features", value=True)

# Main data pull
st.info(f"Pulling {symbol} ({ticker}) history from Yahoo Finance...")
df = fetch_price_history_yahoo(ticker, days, interval)
if df is None or df.empty:
    st.error("Failed to fetch data from Yahoo Finance.")
    st.stop()

# Feature engineering
with st.spinner("Computing features..."):
    df_feat = make_features(df, lookback=lookback, symbol=ticker, show_sent=show_sent)
    if df_feat.empty:
        st.warning("Not enough data for feature construction.")
        st.stop()

features = ["price", "ma", "vol", "momentum", "RSI", "MACD", "Bollinger_Upper", "Bollinger_Lower"]
if show_sent:
    features += ["trend", "sent_reddit", "sent_twitter", "onchain"]
y, y_pred, scaler, model_xgb = train_predict_ensemble(df_feat, features, forecast_horizon, blend)

# Backtest/metrics
signals, cumret, sharpe, max_dd, win_rate = compute_backtest(df_feat.iloc[:-forecast_horizon], y_pred, buy_thresh=0.01)
mae = mean_absolute_error(y, y_pred)
st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
st.metric("Sharpe Ratio", f"{sharpe:.2f}")
st.metric("Max Drawdown", f"{max_dd:.2%}")
st.metric("Win Rate", f"{win_rate:.2%}")

# Charts
st.subheader(f"{symbol}: Price & Forecast")
chart_df = df_feat.iloc[:-forecast_horizon].copy()
chart_df["Predicted"] = y_pred
chart_df = chart_df.set_index("time")
fig = go.Figure()
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["price"], name="Actual Price", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Predicted"], name="Predicted Price", line=dict(color="red")))
st.plotly_chart(fig, use_container_width=True)

# Future forecast
st.subheader(f"Forecasting the Next {forecast_horizon} Steps")
future_preds = forecast_future(df_feat, scaler, model_xgb, features, window_size=lookback, n_steps=forecast_horizon)
future_times = [df_feat["time"].iloc[-1] + timedelta(days=i+1) for i in range(forecast_horizon)]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_feat["time"], y=df_feat["price"], name="History"))
fig2.add_trace(go.Scatter(x=future_times, y=future_preds, name="Future Forecast", line=dict(color="orange", dash="dot")))
st.plotly_chart(fig2, use_container_width=True)

# Download/export
st.download_button("Download Signals (.csv)", chart_df.reset_index()[["time", "price", "Predicted"]].to_csv(index=False), file_name="signals.csv")

# Quantum-ready & notes
st.markdown("""
> 🟦 <b>Quantum Ready:</b> Plug in PennyLane/Qiskit quantum circuits to replace XGBoost/LSTM!<br>
> No files saved, no local storage, 100% in the cloud!
""", unsafe_allow_html=True)
st.caption("© 2025 Quantum Crypto Forecaster (Yahoo Finance)")


