import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pytrends.request import TrendReq
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ---- CONFIG ----
COINS = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Solana": "solana",
    "XRP": "ripple",
    "Dogecoin": "dogecoin",
    "Cardano": "cardano",
    "Avalanche": "avalanche-2",
    "Shiba Inu": "shiba-inu",
    "Litecoin": "litecoin",
    "Chainlink": "chainlink"
}
VS_CURRENCY = "usd"
NEWS_API_KEY = "YOUR_NEWSAPI_KEY"   # Replace with your NewsAPI key

# ---- FUNCTIONS ----
def fetch_price_history(coin_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": VS_CURRENCY, "days": days}
    r = requests.get(url, params=params)
    data = r.json()
    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    prices["date"] = pd.to_datetime(prices["timestamp"], unit="ms").dt.date
    prices = prices.groupby("date").last().reset_index()
    return prices[["date", "price"]]

def fetch_google_trends(keyword, days):
    pytrends = TrendReq(hl='en-US', tz=360)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    timeframe = f"{start_date} {end_date}"
    pytrends.build_payload([keyword], timeframe=timeframe)
    data = pytrends.interest_over_time().reset_index()
    data["date"] = data["date"].dt.date
    data = data.rename(columns={keyword: "trend"})
    return data[["date", "trend"]]

def fetch_news_sentiment(keyword, days):
    analyzer = SentimentIntensityAnalyzer()
    base_url = "https://newsapi.org/v2/everything"
    today = datetime.now()
    from_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "q": keyword,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWS_API_KEY,
        "pageSize": 100,
    }
    r = requests.get(base_url, params=params)
    articles = r.json().get("articles", [])
    if not articles:
        return pd.DataFrame({"date": [], "sentiment": []})
    df = pd.DataFrame([{
        "date": pd.to_datetime(a["publishedAt"]).date(),
        "title": a["title"]
    } for a in articles])
    df["sentiment"] = df["title"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    sentiment_daily = df.groupby("date")["sentiment"].mean().reset_index()
    return sentiment_daily

def make_features(df, lookback):
    df["returns"] = np.log(df["price"] / df["price"].shift(1))
    df["vol"] = df["returns"].rolling(lookback).std()
    df["ma"] = df["price"].rolling(lookback).mean()
    df["momentum"] = df["price"] - df["ma"]
    df = df.fillna(0)
    return df

# ---- APP ----
st.title("Quantum-Ready Crypto Predictor (with Sentiment & Trends)")
coin = st.selectbox("Choose a coin", list(COINS.keys()))
days = st.slider("Days of data", 60, 365, 365)
lookback = st.slider("Feature lookback window (days)", 2, 30, 7)
forecast_horizon = st.slider("Forecast steps ahead (days)", 1, 7, 1)

st.info("Fetching price data, news sentiment, and trends...")

df = fetch_price_history(COINS[coin], days)
trend_df = fetch_google_trends(coin, days)
sent_df = fetch_news_sentiment(coin, days)

# Merge features
df = df.merge(trend_df, on="date", how="left")
df = df.merge(sent_df, on="date", how="left")
df = make_features(df, lookback)
df = df.fillna(0)

# ML modeling
if len(df) < lookback + forecast_horizon + 1:
    st.error("Not enough data to fit the model. Try lowering lookback/horizon.")
else:
    # Feature/target setup
    features = ["price", "vol", "momentum", "trend", "sentiment"]
    X, y = [], []
    for i in range(len(df) - lookback - forecast_horizon):
        X.append(df[features].iloc[i:i+lookback].values.flatten())
        y.append(df["price"].iloc[i+lookback+forecast_horizon-1])
    X = np.array(X)
    y = np.array(y)

    # Train/test split
    if len(y) > 7:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.success(f"MAE (test): {mae:.2f} {VS_CURRENCY.upper()}")

    # Next-day prediction (rolling forward)
    recent_feat = df[features].iloc[-lookback:].values.flatten().reshape(1, -1)
    pred_next = model.predict(recent_feat)[0]
    st.write(f"**Predicted price for {coin} ({forecast_horizon} day(s) ahead):** {pred_next:,.2f} {VS_CURRENCY.upper()}")

    # Plot
    chart_df = df[["date", "price"]].copy()
    chart_df["pred"] = np.nan
    chart_df.loc[chart_df.index[-1], "pred"] = pred_next
    st.line_chart(chart_df.set_index("date"))

    with st.expander("Show full DataFrame"):
        st.dataframe(df)

    st.caption("Features: price, vol (volatility), momentum, Google Trend, news sentiment. Models retrain on each run for latest predictions.")

# --- END APP ---
