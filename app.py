import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.data import load_data, get_stock_info
from src.features import create_features
from src.train import train_pipeline

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-card h3 {
        color: #aaa;
        font-size: 0.85rem;
        margin-bottom: 5px;
    }
    .metric-card p {
        font-size: 1.6rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-green p { color: #2ca02c; }
    .metric-blue p { color: #1f77b4; }
    .metric-orange p { color: #ff7f0e; }
    .metric-red p { color: #d62728; }

    /* Info cards */
    .info-card {
        background: #1e1e2f;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 10px;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #1a1a2e 100%);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">📈 Stock Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered stock price prediction using Random Forest with technical indicators</p>', unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # Stock Selection
    st.markdown("### 💱 Currency")
    currency_option = st.selectbox("Display currency", ["INR (₹)", "USD ($)"], index=0)
    convert_to_inr = currency_option.startswith("INR")
    currency_symbol = "₹" if convert_to_inr else "$"
    st.session_state["currency_symbol"] = currency_symbol
    st.session_state["convert_to_inr"] = convert_to_inr

    st.markdown("---")

    st.markdown("### 🏢 Stock Selection")
    popular_stocks = {
        "Apple (AAPL)": "AAPL",
        "Google (GOOGL)": "GOOGL",
        "Microsoft (MSFT)": "MSFT",
        "Amazon (AMZN)": "AMZN",
        "Tesla (TSLA)": "TSLA",
        "Meta (META)": "META",
        "NVIDIA (NVDA)": "NVDA",
        "Netflix (NFLX)": "NFLX",
        "Custom ticker...": "CUSTOM",
    }

    selected = st.selectbox("Choose a stock", list(popular_stocks.keys()), index=0)

    if popular_stocks[selected] == "CUSTOM":
        ticker = st.text_input("Enter custom ticker symbol", "AAPL").upper()
    else:
        ticker = popular_stocks[selected]

    st.markdown("---")

    # Date Range
    st.markdown("### 📅 Date Range")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input(
            "Start",
            value=datetime(2018, 1, 1),
            min_value=datetime(2000, 1, 1),
            max_value=datetime.today() - timedelta(days=365),
        )
    with col_d2:
        end_date = st.date_input(
            "End",
            value=datetime.today(),
            min_value=datetime(2001, 1, 1),
            max_value=datetime.today(),
        )

    st.markdown("---")

    # Actions
    st.markdown("### 🚀 Actions")
    btn_load = st.button("📥  Load Data", use_container_width=True)
    btn_train = st.button("🧠  Train Model", use_container_width=True)
    btn_predict = st.button("🔮  Predict Next Day", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#555; font-size:0.8rem;'>"
        "Built with Streamlit & Scikit-learn</div>",
        unsafe_allow_html=True,
    )


# ─── Helpers ────────────────────────────────────────────────────────────────
def format_large_number(n, sym=None):
    if sym is None:
        sym = st.session_state.get("currency_symbol", "₹")
    if n == "N/A":
        return "N/A"
    try:
        n = float(n)
    except (ValueError, TypeError):
        return str(n)
    if n >= 1e12:
        return f"{sym}{n/1e12:.2f}T"
    if n >= 1e9:
        return f"{sym}{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{sym}{n/1e6:.2f}M"
    return f"{sym}{n:,.0f}"


MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")


# ─── LOAD DATA ──────────────────────────────────────────────────────────────
if btn_load or "df" not in st.session_state:
    if btn_load:
        with st.spinner(f"Fetching data for **{ticker}**..."):
            try:
                df = load_data(ticker, start=str(start_date), end=str(end_date), convert_to_inr=convert_to_inr)
                info = get_stock_info(ticker)
                st.session_state["df"] = df
                st.session_state["ticker"] = ticker
                st.session_state["info"] = info
                st.session_state["start_date"] = str(start_date)
                st.session_state["end_date"] = str(end_date)
                st.toast(f"✅ Loaded {len(df)} rows for {ticker}", icon="📥")
            except Exception as e:
                st.error(f"Failed to load data: {e}")

# ─── Main Content ───────────────────────────────────────────────────────────
if "df" in st.session_state:
    df = st.session_state["df"]
    ticker_display = st.session_state.get("ticker", ticker)
    info = st.session_state.get("info", {})
    currency_symbol = st.session_state.get("currency_symbol", "₹")
    convert_to_inr = st.session_state.get("convert_to_inr", True)

    # ── Stock Info Banner ────────────────────────────────────────────────────
    stock_name = info.get("name", ticker_display)

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        current_price = df["Close"].iloc[-1]
        prev_price = df["Close"].iloc[-2] if len(df) > 1 else current_price
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
        color = "#2ca02c" if change >= 0 else "#d62728"
        arrow = "▲" if change >= 0 else "▼"
        st.markdown(
            f"""<div class="info-card">
                <b>{stock_name}</b> ({ticker_display})<br>
                <span style="font-size:1.8rem; font-weight:bold;">{currency_symbol}{current_price:.2f}</span><br>
                <span style="color:{color}; font-size:1rem;">{arrow} {change:+.2f} ({change_pct:+.2f}%)</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_info2:
        st.markdown(
            f"""<div class="info-card">
                <b>Market Cap</b><br>
                <span style="font-size:1.3rem;">{format_large_number(info.get("market_cap", "N/A"))}</span><br>
                <span style="color:#888;">Sector: {info.get("sector", "N/A")}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_info3:
        st.markdown(
            f"""<div class="info-card">
                <b>52-Week Range</b><br>
                <span style="color:#d62728;">Low: {currency_symbol}{info.get("52w_low", "N/A")}</span><br>
                <span style="color:#2ca02c;">High: {currency_symbol}{info.get("52w_high", "N/A")}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_info4:
        avg_vol = info.get("avg_volume", "N/A")
        st.markdown(
            f"""<div class="info-card">
                <b>Avg Volume</b><br>
                <span style="font-size:1.3rem;">{format_large_number(avg_vol)}</span><br>
                <span style="color:#888;">Exchange: {info.get("exchange", "N/A")}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_chart, tab_analysis, tab_train, tab_predict = st.tabs(
        ["📊 Price Chart", "🔍 Technical Analysis", "🧠 Train Model", "🔮 Predict"]
    )

    # ──── Tab 1: Price Chart ─────────────────────────────────────────────────
    with tab_chart:
        st.subheader("Interactive Price Chart")

        chart_type = st.radio(
            "Chart type", ["Candlestick", "Line", "Area"],
            horizontal=True, label_visibility="collapsed",
        )

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
            subplot_titles=("", "Volume"),
        )

        if chart_type == "Candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"], name="OHLC",
                ),
                row=1, col=1,
            )
        elif chart_type == "Line":
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["Close"], name="Close",
                    line=dict(color="#1f77b4", width=2),
                ),
                row=1, col=1,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df["Close"], name="Close",
                    fill="tozeroy", fillcolor="rgba(31,119,180,0.15)",
                    line=dict(color="#1f77b4", width=2),
                ),
                row=1, col=1,
            )

        # Volume bars
        colors = ["#2ca02c" if c >= o else "#d62728" for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume", showlegend=False),
            row=2, col=1,
        )

        fig.update_layout(
            height=600,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Quick Stats
        st.markdown("#### Quick Statistics")
        qs1, qs2, qs3, qs4, qs5 = st.columns(5)
        qs1.metric("Open", f"{currency_symbol}{df['Open'].iloc[-1]:.2f}")
        qs2.metric("High", f"{currency_symbol}{df['High'].iloc[-1]:.2f}")
        qs3.metric("Low", f"{currency_symbol}{df['Low'].iloc[-1]:.2f}")
        qs4.metric("Close", f"{currency_symbol}{df['Close'].iloc[-1]:.2f}")
        qs5.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")

    # ──── Tab 2: Technical Analysis ──────────────────────────────────────────
    with tab_analysis:
        st.subheader("Technical Indicators")

        # Moving averages chart
        ma_fig = go.Figure()
        ma_fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#1f77b4", width=2)))

        ma10 = df["Close"].rolling(10).mean()
        ma20 = df["Close"].rolling(20).mean()
        ma50 = df["Close"].rolling(50).mean()

        ma_fig.add_trace(go.Scatter(x=df.index, y=ma10, name="MA 10", line=dict(color="#ff7f0e", dash="dot")))
        ma_fig.add_trace(go.Scatter(x=df.index, y=ma20, name="MA 20", line=dict(color="#2ca02c", dash="dot")))
        ma_fig.add_trace(go.Scatter(x=df.index, y=ma50, name="MA 50", line=dict(color="#d62728", dash="dash")))

        ma_fig.update_layout(
            title="Moving Averages", height=400, template="plotly_dark",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(ma_fig, use_container_width=True)

        # RSI + MACD
        col_rsi, col_macd = st.columns(2)

        with col_rsi:
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI", line=dict(color="#9467bd", width=2)))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            rsi_fig.update_layout(
                title="RSI (14)", height=300, template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(range=[0, 100]),
            )
            st.plotly_chart(rsi_fig, use_container_width=True)

            # RSI interpretation
            current_rsi = rsi.dropna().iloc[-1] if not rsi.dropna().empty else 50
            if current_rsi > 70:
                st.warning(f"RSI = {current_rsi:.1f} — **Overbought** territory. Potential pullback.")
            elif current_rsi < 30:
                st.success(f"RSI = {current_rsi:.1f} — **Oversold** territory. Potential bounce.")
            else:
                st.info(f"RSI = {current_rsi:.1f} — Neutral zone.")

        with col_macd:
            ema12 = df["Close"].ewm(span=12, adjust=False).mean()
            ema26 = df["Close"].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line

            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD", line=dict(color="#1f77b4", width=2)))
            macd_fig.add_trace(go.Scatter(x=df.index, y=signal_line, name="Signal", line=dict(color="#ff7f0e", width=2)))
            hist_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in histogram]
            macd_fig.add_trace(go.Bar(x=df.index, y=histogram, name="Histogram", marker_color=hist_colors))
            macd_fig.update_layout(
                title="MACD", height=300, template="plotly_dark",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(macd_fig, use_container_width=True)

            last_macd = macd_line.dropna().iloc[-1]
            last_signal = signal_line.dropna().iloc[-1]
            if last_macd > last_signal:
                st.success(f"MACD ({last_macd:.2f}) above Signal ({last_signal:.2f}) — **Bullish**")
            else:
                st.warning(f"MACD ({last_macd:.2f}) below Signal ({last_signal:.2f}) — **Bearish**")

        # Bollinger Bands
        st.markdown("#### Bollinger Bands")
        bb_mid = df["Close"].rolling(20).mean()
        bb_std = df["Close"].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        bb_fig = go.Figure()
        bb_fig.add_trace(go.Scatter(x=df.index, y=bb_upper, name="Upper Band", line=dict(color="#d62728", dash="dash")))
        bb_fig.add_trace(go.Scatter(
            x=df.index, y=bb_lower, name="Lower Band",
            line=dict(color="#2ca02c", dash="dash"),
            fill="tonexty", fillcolor="rgba(44,160,44,0.08)",
        ))
        bb_fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#1f77b4", width=2)))
        bb_fig.add_trace(go.Scatter(x=df.index, y=bb_mid, name="SMA 20", line=dict(color="#ff7f0e", dash="dot")))
        bb_fig.update_layout(
            height=400, template="plotly_dark",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(bb_fig, use_container_width=True)

    # ──── Tab 3: Train Model ────────────────────────────────────────────────
    with tab_train:
        st.subheader("Train Prediction Model")
        st.markdown(
            "This trains a **Random Forest Regressor** with hyperparameter tuning "
            "(RandomizedSearchCV) on technical indicators."
        )

        if btn_train:
            progress_bar = st.progress(0, text="Starting training...")
            status_text = st.empty()

            def update_progress(frac, msg):
                progress_bar.progress(frac, text=msg)
                status_text.text(msg)

            try:
                result = train_pipeline(
                    ticker_display,
                    start_date=st.session_state.get("start_date", "2018-01-01"),
                    end_date=st.session_state.get("end_date", None),
                    progress_callback=update_progress,
                )
                st.session_state["train_result"] = result
                progress_bar.progress(1.0, text="✅ Training complete!")
                st.toast("Model trained successfully!", icon="🧠")
            except Exception as e:
                st.error(f"Training failed: {e}")

        if "train_result" in st.session_state:
            result = st.session_state["train_result"]
            metrics = result["metrics"]

            # Metrics row
            st.markdown("#### Model Performance")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.markdown(
                f'<div class="metric-card metric-green"><h3>R² Score</h3><p>{metrics["R2"]:.4f}</p></div>',
                unsafe_allow_html=True,
            )
            mc2.markdown(
                f'<div class="metric-card metric-blue"><h3>MAE</h3><p>{metrics["MAE"]:.4f}</p></div>',
                unsafe_allow_html=True,
            )
            mc3.markdown(
                f'<div class="metric-card metric-orange"><h3>RMSE</h3><p>{metrics["RMSE"]:.4f}</p></div>',
                unsafe_allow_html=True,
            )
            mc4.markdown(
                f'<div class="metric-card metric-red"><h3>MSE</h3><p>{metrics["MSE"]:.4f}</p></div>',
                unsafe_allow_html=True,
            )

            st.markdown("")

            # Actual vs Predicted
            col_ap, col_fi = st.columns([3, 2])

            with col_ap:
                st.markdown("#### Actual vs Predicted")
                ap_fig = go.Figure()
                y_test = result["y_test"]
                y_pred = result["y_pred"]
                ap_fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test.values, name="Actual", line=dict(color="#1f77b4", width=2)))
                ap_fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, name="Predicted", line=dict(color="#ff7f0e", width=2, dash="dot")))
                ap_fig.update_layout(
                    height=400, template="plotly_dark",
                    xaxis_title="Test Samples", yaxis_title=f"Price ({currency_symbol})",
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(ap_fig, use_container_width=True)

            with col_fi:
                st.markdown("#### Feature Importance")
                importances = result["feature_importances"]
                sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
                fi_fig = go.Figure(go.Bar(
                    x=list(sorted_imp.values()),
                    y=list(sorted_imp.keys()),
                    orientation="h",
                    marker_color="#1f77b4",
                ))
                fi_fig.update_layout(
                    height=400, template="plotly_dark",
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fi_fig, use_container_width=True)

            # Best hyperparameters
            with st.expander("🔧 Best Hyperparameters"):
                for k, v in result["best_params"].items():
                    st.write(f"**{k}**: {v}")

            # Prediction error distribution
            with st.expander("📊 Prediction Error Distribution"):
                errors = y_test.values - y_pred
                err_fig = go.Figure(go.Histogram(x=errors, nbinsx=40, marker_color="#9467bd"))
                err_fig.update_layout(
                    title="Prediction Error Distribution",
                    xaxis_title=f"Error ({currency_symbol})", yaxis_title="Count",
                    height=300, template="plotly_dark",
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(err_fig, use_container_width=True)
        else:
            st.info("👈 Click **Train Model** in the sidebar to start training.")

    # ──── Tab 4: Predict ─────────────────────────────────────────────────────
    with tab_predict:
        st.subheader("Predict Next Day Closing Price")

        model_exists = os.path.exists(MODEL_PATH)

        if not model_exists and "train_result" not in st.session_state:
            st.warning("⚠️ No trained model found. Please train a model first in the **Train Model** tab.")
        else:
            if btn_predict:
                with st.spinner("Generating prediction..."):
                    try:
                        if "train_result" in st.session_state:
                            model = st.session_state["train_result"]["model"]
                        else:
                            model = joblib.load(MODEL_PATH)

                        df_fresh = load_data(
                            ticker_display,
                            start=st.session_state.get("start_date", "2018-01-01"),
                            end=st.session_state.get("end_date", None),
                            convert_to_inr=convert_to_inr,
                        )
                        X, y, df_feat = create_features(df_fresh)

                        last_row = X.iloc[-1:]
                        prediction = model.predict(last_row)[0]
                        last_close = df_fresh["Close"].iloc[-1]

                        change = prediction - last_close
                        change_pct = (change / last_close) * 100

                        st.session_state["prediction"] = {
                            "value": prediction,
                            "last_close": last_close,
                            "change": change,
                            "change_pct": change_pct,
                            "df_feat": df_feat,
                        }
                        st.toast("Prediction generated!", icon="🔮")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

            if "prediction" in st.session_state:
                pred = st.session_state["prediction"]
                prediction = pred["value"]
                last_close = pred["last_close"]
                change = pred["change"]
                change_pct = pred["change_pct"]

                # Prediction banner
                pred_color = "#2ca02c" if change >= 0 else "#d62728"
                arrow = "▲" if change >= 0 else "▼"
                direction = "UP" if change >= 0 else "DOWN"

                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #1e1e2f, #2d2d44);
                                border-radius: 16px; padding: 30px; text-align: center;
                                border: 2px solid {pred_color}; margin-bottom: 20px;">
                        <p style="color: #888; font-size: 1rem; margin-bottom: 5px;">
                            Predicted Next Day Close for <b>{ticker_display}</b>
                        </p>
                        <p style="font-size: 3rem; font-weight: bold; color: {pred_color}; margin: 5px 0;">
                            {currency_symbol}{prediction:.2f}
                        </p>
                        <p style="font-size: 1.2rem; color: {pred_color};">
                            {arrow} {direction} {currency_symbol}{abs(change):.2f} ({change_pct:+.2f}%)
                        </p>
                        <p style="color: #666; font-size: 0.85rem;">
                            Current Close: {currency_symbol}{last_close:.2f}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Chart with prediction point
                df_feat = pred["df_feat"]
                recent = df_feat.tail(60)

                pred_fig = go.Figure()
                pred_fig.add_trace(go.Scatter(
                    x=recent.index, y=recent["Close"], name="Historical Close",
                    line=dict(color="#1f77b4", width=2),
                ))

                # Add prediction point as a future date
                last_date = recent.index[-1]
                next_date = last_date + pd.Timedelta(days=1)
                pred_fig.add_trace(go.Scatter(
                    x=[last_date, next_date],
                    y=[last_close, prediction],
                    name="Prediction",
                    line=dict(color=pred_color, width=3, dash="dash"),
                    mode="lines+markers",
                    marker=dict(size=12, symbol="star"),
                ))

                pred_fig.update_layout(
                    title="Recent Price + Prediction",
                    height=400, template="plotly_dark",
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis_title=f"Price ({currency_symbol})",
                )
                st.plotly_chart(pred_fig, use_container_width=True)

                st.info(
                    "⚠️ **Disclaimer**: This prediction is based on historical patterns and "
                    "technical indicators. It should NOT be used as financial advice. "
                    "Stock markets are inherently unpredictable."
                )
            else:
                st.info("👈 Click **Predict Next Day** in the sidebar to generate a prediction.")

else:
    # No data loaded yet — show welcome
    st.markdown("---")

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        st.markdown(
            """### 📊 Visualize
            Interactive candlestick charts, volume analysis, and price trends with zoom & pan.
            """
        )
    with col_w2:
        st.markdown(
            """### 🔍 Analyze
            Technical indicators — RSI, MACD, Bollinger Bands, Moving Averages — all computed live.
            """
        )
    with col_w3:
        st.markdown(
            """### 🔮 Predict
            Train a Random Forest model on historical data and predict the next day's closing price.
            """
        )

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#888;">👈 Select a stock and click '
        '<b>Load Data</b> in the sidebar to get started.</p>',
        unsafe_allow_html=True,
    )