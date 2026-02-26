import yfinance as yf
import pandas as pd


def load_data(ticker, start="2015-01-01", end=None, convert_to_inr=False):
    """
    Download stock data.
    Optionally convert USD stocks to INR using historical USDINR rate.
    """

    df = yf.download(ticker, start=start, end=end)

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)

    if convert_to_inr:
        stock = yf.Ticker(ticker)
        currency = stock.info.get("currency", "USD")

        # Convert only if stock is in USD
        if currency == "USD":

            fx = yf.download("USDINR=X", start=start, end=end)

            if isinstance(fx.columns, pd.MultiIndex):
                fx.columns = fx.columns.get_level_values(0)

            fx = fx["Close"].reindex(df.index).ffill()

            for col in ["Open", "High", "Low", "Close"]:
                df[col] = df[col] * fx

            print("Converted USD prices to INR")

    return df


def get_stock_info(ticker):
    """Fetch basic stock information."""
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
        return {"name": ticker}