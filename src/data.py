import yfinance as yf
import pandas as pd
import streamlit as st


# -----------------------------------
# FX DATA CACHE (USD → INR)
# -----------------------------------
@st.cache_data(ttl=86400)
def get_fx_data(start="2015-01-01", end=None):

    fx = yf.download("USDINR=X", start=start, end=end, progress=False)

    if fx is None or fx.empty:
        return pd.Series()

    # Flatten MultiIndex if present
    if isinstance(fx.columns, pd.MultiIndex):
        fx.columns = fx.columns.get_level_values(0)

    return fx["Close"]


# -----------------------------------
# STOCK DATA LOADER
# -----------------------------------
@st.cache_data(ttl=3600)
def load_data(ticker, start="2015-01-01", end=None, convert_to_inr=False):

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, threads=False)
    except Exception:
        st.error("Failed to download stock data.")
        return pd.DataFrame()

    # Check if data exists
    if df is None or df.empty:
        st.warning(f"No data found for ticker: {ticker}")
        return pd.DataFrame()

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)

    if df.empty:
        st.warning("Data became empty after cleaning.")
        return pd.DataFrame()

    # -------------------------------
    # Convert USD stocks to INR
    # -------------------------------
    if convert_to_inr:

        stock = yf.Ticker(ticker)

        try:
            currency = stock.fast_info.get("currency", "USD")
        except Exception:
            currency = "USD"

        if currency == "USD":

            fx = get_fx_data(start, end)

            if not fx.empty:

                fx = fx.reindex(df.index).ffill()

                for col in ["Open", "High", "Low", "Close"]:
                    if col in df.columns:
                        df[col] = df[col] * fx

    return df


# -----------------------------------
# STOCK INFO FETCHER
# -----------------------------------
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