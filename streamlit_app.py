import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error

# ---- CONFIG ----
DEFAULT_SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD",
    "AVAX-USD", "LINK-USD", "MATIC-USD", "WIF-USD", "SHIB-USD"
]
LOOKBACK_RANGE = (5, 120)
FORECAST_RANGE = (1, 14)

# ---- SIDEBAR UI ----
st.sidebar.title("Crypto Prediction Dashboard")
symbol = st.sidebar.selectbox("Select Coin", DEFAULT_SYMBOLS)
lookback = st.sidebar.slider("Lookback Window (days)", *LOOKBACK_RANGE, value=30)
forecast_horizon = st.sidebar.slider("Forecast Horizon (days ahead)", *FORECAST_RANGE, value=1)
days = st.sidebar.slider("History (days)", 90, 730, 365)
show_mae = st.sidebar.checkbox("Show MAE (Model Error)", True)

# ---- FETCH DATA ----
@st.cache_data(ttl=3600)
def fetch_data(symbol, days):
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        return None
    df = df[["Close"]].rename(columns={"Close": "close"})
    df = df.dropna()
    df["returns"] = df["close"].pct_change().fillna(0)
    return df

df = fetch_data(symbol, days)
if df is None or len(df) < lookback + forecast_horizon + 5:
    st.error("Not enough data for this coin. Try a different one, or reduce lookback/horizon.")
    st.stop()

# ---- FEATURE ENGINEERING ----
def make_features(df, lookback):
    X, y = [], []
    for i in range(lookback, len(df) - forecast_horizon):
        window = df.iloc[i - lookback:i]
        target_idx = i + forecast_horizon - 1
        if target_idx >= len(df):
            break
        feats = [
            window["close"].iloc[-1],                       # Last price
            window["close"].mean(),                         # Mean price
            window["close"].std(),                          # Volatility
            window["returns"].iloc[-1],                     # Last return
            window["returns"].mean(),                       # Mean return
            window["returns"].std(),                        # Vol of returns
        ]
        X.append(feats)
        y.append(df["close"].iloc[target_idx])
    return np.array(X), np.array(y)

X, y = make_features(df, lookback)
if len(X) == 0:
    st.warning("Prediction could not be computed (too little data or all NaN).")
    st.stop()

# ---- TRAIN/TEST SPLIT ----
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---- MODEL ----
model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---- PREDICT NEXT PRICES ----
# Take last `lookback` days for next N-step forecast
def predict_next_n(df, lookback, N):
    X_last = []
    base_idx = len(df) - lookback
    for k in range(1, N+1):
        window = df.iloc[base_idx:base_idx+lookback]
        feats = [
            window["close"].iloc[-1],
            window["close"].mean(),
            window["close"].std(),
            window["returns"].iloc[-1],
            window["returns"].mean(),
            window["returns"].std(),
        ]
        X_last.append(feats)
    return model.predict(np.array(X_last))

future_prices = predict_next_n(df, lookback, forecast_horizon)
future_dates = [df.index[-1] + timedelta(days=i+1) for i in range(forecast_horizon)]

# ---- UI OUTPUT ----
st.title("🔮 Quantum-Ready Crypto Price Predictor (XGBoost Only)")

# Main chart
chart_df = pd.DataFrame({
    "Actual Price": df["close"].iloc[-len(y_test):].values,
    "Predicted Price": y_pred
}, index=df.index[-len(y_test):])
chart_df_future = pd.DataFrame({
    "Forecasted Price": future_prices
}, index=future_dates)

st.line_chart(chart_df)
st.line_chart(chart_df_future)

if show_mae:
    mae = mean_absolute_error(y_test, y_pred)
    st.info(f"Model MAE (last test period): {mae:,.2f}")

st.caption("Model: XGBoost regressor with simple rolling features. Edit and extend this for more sophistication!")
