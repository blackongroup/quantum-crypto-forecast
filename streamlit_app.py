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
import ta

# Optional: Twitter sentiment
import tweepy

# --------------- CONFIG ---------------
COINS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "SHIB": "shiba-inu",
    "LTC": "litecoin",
    "LINK": "chainlink"
}
VS_CURRENCY = "usd"
NEWS_API_KEY = "YOUR_NEWSAPI_KEY" # <-- enter your NewsAPI key here

# (Optional) Twitter API keys (free dev acc required)
TWITTER_KEYS = {
    "api_key": "YOUR_API_KEY",
    "api_secret": "YOUR_API_SECRET",
    "access_token": "YOUR_ACCESS_TOKEN",
    "access_secret": "YOUR_ACCESS_SECRET"
}

# ------------- DATA FUNCTIONS -------------
def fetch_price_history(coin_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": VS_CURRENCY, "days": days}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        st.error(f"CoinGecko error: {r.text}")
        return pd.DataFrame()
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
    try:
        pytrends.build_payload([keyword], timeframe=timeframe)
        data = pytrends.interest_over_time().reset_index()
        data["date"] = data["date"].dt.date
        data = data.rename(columns={keyword: "trend"})
        return data[["date", "trend"]]
    except Exception as e:
        st.warning("Google Trends error: " + str(e))
        return pd.DataFrame({"date": [], "trend": []})

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
    try:
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
    except Exception as e:
        st.warning("NewsAPI error: " + str(e))
        return pd.DataFrame({"date": [], "sentiment": []})

def fetch_twitter_sentiment(keyword, days):
    try:
        auth = tweepy.OAuth1UserHandler(
            TWITTER_KEYS["api_key"], TWITTER_KEYS["api_secret"],
            TWITTER_KEYS["access_token"], TWITTER_KEYS["access_secret"])
        api = tweepy.API(auth)
        analyzer = SentimentIntensityAnalyzer()
        since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang='en', since=since, tweet_mode='extended').items(200)
        data = []
        for tweet in tweets:
            dt = tweet.created_at.date()
            score = analyzer.polarity_scores(tweet.full_text)["compound"]
            data.append({"date": dt, "sentiment": score})
        if data:
            df = pd.DataFrame(data)
            return df.groupby("date")["sentiment"].mean().reset_index()
        else:
            return pd.DataFrame({"date": [], "sentiment": []})
    except Exception as e:
        st.warning("Twitter API error: " + str(e))
        return pd.DataFrame({"date": [], "sentiment": []})

def add_ta_features(df, lookback):
    df = df.copy()
    df["returns"] = np.log(df["price"] / df["price"].shift(1))
    df["vol"] = df["returns"].rolling(lookback).std()
    df["ma"] = df["price"].rolling(lookback).mean()
    df["momentum"] = df["price"] - df["ma"]
    df["rsi"] = ta.momentum.RSIIndicator(df["price"], window=lookback).rsi()
    macd = ta.trend.MACD(df["price"])
    df["macd"] = macd.macd()
    df = df.fillna(0)
    return df

# --------------- STREAMLIT UI -----------------
st.title("Quantum-Ready Crypto Predictor (All-in-One Alpha Dashboard)")
coin = st.selectbox("Choose a coin", list(COINS.keys()))
days = st.slider("Days of data", 60, 365, 365)
lookback = st.slider("Feature lookback window (days)", 2, 30, 7)
forecast_horizon = st.slider("Forecast steps ahead (days)", 1, 7, 1)
st.caption("Features: price, vol, momentum, RSI, MACD, Google Trends, News & Twitter Sentiment.")

st.info("Fetching price data, news, Twitter, and Google Trends...")

df = fetch_price_history(COINS[coin], days)
if df.empty:
    st.error("Failed to fetch price data. Try again.")
    st.stop()

trend_df = fetch_google_trends(coin, days)
news_df = fetch_news_sentiment(coin, days)
# Optional: Twitter sentiment
twitter_df = fetch_twitter_sentiment(coin, days)

# Merge features
df = df.merge(trend_df, on="date", how="left")
df = df.merge(news_df, on="date", how="left", suffixes=("", "_news"))
df = df.merge(twitter_df, on="date", how="left", suffixes=("", "_twitter"))
df = add_ta_features(df, lookback)
df = df.fillna(0)

# ML features
features = ["price", "vol", "momentum", "rsi", "macd", "trend", "sentiment", "sentiment_twitter"]

# ML modeling
if len(df) < lookback + forecast_horizon + 10:
    st.error("Not enough data to fit the model. Try lowering lookback/horizon.")
else:
    # Feature/target setup
    X, y = [], []
    for i in range(len(df) - lookback - forecast_horizon):
        row = []
        for feat in features:
            # For missing features (e.g., trend/sentiment), use fill value 0
            v = df[feat].iloc[i:i+lookback].values if feat in df.columns else np.zeros(lookback)
            row.extend(v)
        X.append(row)
        y.append(df["price"].iloc[i+lookback+forecast_horizon-1])
    X = np.array(X)
    y = np.array(y)

    # Train/test split
    if len(y) > 7:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    model = XGBRegressor(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.success(f"MAE (test): {mae:.2f} {VS_CURRENCY.upper()}")

    # Next forecast
    recent_feat = []
    for feat in features:
        v = df[feat].iloc[-lookback:].values if feat in df.columns else np.zeros(lookback)
        recent_feat.extend(v)
    pred_next = model.predict(np.array(recent_feat).reshape(1, -1))[0]
    st.write(f"**Predicted price for {coin} ({forecast_horizon} day(s) ahead):** {pred_next:,.2f} {VS_CURRENCY.upper()}")

    # Plot actual & predicted
    chart_df = df[["date", "price"]].copy()
    chart_df["pred"] = np.nan
    chart_df.loc[chart_df.index[-1], "pred"] = pred_next
    st.line_chart(chart_df.set_index("date"))

    # Feature importance
    st.subheader("Feature importance (mean gain):")
    imp_df = pd.DataFrame({
        "feature": [f"{feat}_{i+1}" for feat in features for i in range(lookback)],
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    st.dataframe(imp_df.head(20))

    with st.expander("Show full DataFrame"):
        st.dataframe(df)

    st.caption("Features: price, vol (volatility), momentum, RSI, MACD, Google Trend, News and Twitter Sentiment.")

# --- END APP ---
