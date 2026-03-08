import yfinance as yf
import pandas as pd
import streamlit as st


# -------------------------------
# FX DATA CACHE (USD → INR)
# -------------------------------
@st.cache_data(ttl=86400)
def get_fx_data(start="2015-01-01", end=None):
    fx = yf.download("USDINR=X", start=start, end=end)

    if isinstance(fx.columns, pd.MultiIndex):
        fx.columns = fx.columns.get_level_values(0)

    return fx["Close"]


# -------------------------------
# STOCK DATA LOADER
# -------------------------------
@st.cache_data(ttl=3600)
def load_data(ticker, start="2015-01-01", end=None, convert_to_inr=False):

    df = yf.download(ticker, start=start, end=end)

    # Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)

    if convert_to_inr:

        stock = yf.Ticker(ticker)

        try:
            currency = stock.fast_info["currency"]
        except Exception:
            currency = "USD"

        if currency == "USD":

            fx = get_fx_data(start, end)

            fx = fx.reindex(df.index).ffill()

            for col in ["Open", "High", "Low", "Close"]:
                if col in df.columns:
                    df[col] = df[col] * fx

    return df


# -------------------------------
# STOCK INFO FETCHER
# -------------------------------
@st.cache_data(ttl=3600)
def get_stock_info(ticker):

    try:
        stock = yf.Ticker(ticker)

        info = stock.info

        return {
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "N/A"),
            "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
            "avg_volume": info.get("averageVolume", "N/A"),
        }

    except Exception:

        return {
            "name": ticker,
            "sector": "N/A",
            "industry": "N/A",
            "market_cap": "N/A",
            "currency": "USD",
            "exchange": "N/A",
            "52w_high": "N/A",
            "52w_low": "N/A",
            "avg_volume": "N/A",
        }