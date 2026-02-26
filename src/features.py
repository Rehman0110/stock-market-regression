import pandas as pd
import numpy as np


def create_features(df):
    """Create technical indicator features for the model."""
    df = df.copy()

    # Price-based features
    df["Return"] = df["Close"].pct_change()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # Volatility
    df["Volatility"] = df["Return"].rolling(10).std()

    # RSI (Relative Strength Index)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands width
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Width"] = (2 * bb_std) / bb_mid

    # Price momentum
    df["Momentum_5"] = df["Close"] / df["Close"].shift(5) - 1
    df["Momentum_10"] = df["Close"] / df["Close"].shift(10) - 1

    # Target: next day close
    df["Target"] = df["Close"].shift(-1)

    df.dropna(inplace=True)

    feature_cols = [
        "Open", "High", "Low", "Volume",
        "MA10", "MA20", "MA50", "Volatility",
        "RSI", "MACD", "MACD_Signal", "BB_Width",
        "Momentum_5", "Momentum_10"
    ]

    X = df[feature_cols]
    y = df["Target"]

    return X, y, df