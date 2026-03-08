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


st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #1f77b4, #2ca02c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-header {color:#888;font-size:1rem;margin-bottom:30px;}

.metric-card {
    background: linear-gradient(135deg,#1e1e2f,#2d2d44);
    border-radius:12px;padding:20px;text-align:center;border:1px solid #333;
}
.metric-card h3{color:#aaa;font-size:.85rem;margin-bottom:5px;}
.metric-card p{font-size:1.6rem;font-weight:bold;margin:0;}

.info-card{
background:#1e1e2f;border-radius:10px;padding:16px;border-left:4px solid #1f77b4;margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<p class="main-header">📈 Stock Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered stock price prediction using Random Forest</p>', unsafe_allow_html=True)


with st.sidebar:

    st.markdown("## ⚙️ Configuration")

    currency_option = st.selectbox("Currency",["INR (₹)","USD ($)"])
    convert_to_inr = currency_option.startswith("INR")
    currency_symbol = "₹" if convert_to_inr else "$"

    st.session_state["currency_symbol"]=currency_symbol
    st.session_state["convert_to_inr"]=convert_to_inr

    popular_stocks={
        "Apple (AAPL)":"AAPL",
        "Google (GOOGL)":"GOOGL",
        "Microsoft (MSFT)":"MSFT",
        "Amazon (AMZN)":"AMZN",
        "Tesla (TSLA)":"TSLA",
        "Meta (META)":"META",
        "NVIDIA (NVDA)":"NVDA",
        "Netflix (NFLX)":"NFLX",
        "Custom ticker...":"CUSTOM"
    }

    selected=st.selectbox("Choose stock",list(popular_stocks.keys()))

    if popular_stocks[selected]=="CUSTOM":
        ticker=st.text_input("Ticker","AAPL").upper()
    else:
        ticker=popular_stocks[selected]

    col1,col2=st.columns(2)

    with col1:
        start_date=st.date_input("Start",datetime(2018,1,1))

    with col2:
        end_date=st.date_input("End",datetime.today())

    btn_load=st.button("📥 Load Data",use_container_width=True)
    btn_train=st.button("🧠 Train Model",use_container_width=True)
    btn_predict=st.button("🔮 Predict Next Day",use_container_width=True)


MODEL_PATH=os.path.join(os.path.dirname(__file__),"best_model.pkl")


# LOAD DATA

if btn_load or "df" not in st.session_state:

    if btn_load:

        with st.spinner("Fetching data..."):

            try:

                df=load_data(
                    ticker,
                    start=str(start_date),
                    end=str(end_date),
                    convert_to_inr=convert_to_inr
                )

                # FIX 1: prevent empty dataframe crash
                if df is None or df.empty:
                    st.error("No data returned for this ticker.")
                    st.stop()

                info=get_stock_info(ticker)

                st.session_state["df"]=df
                st.session_state["ticker"]=ticker
                st.session_state["info"]=info

            except Exception as e:
                st.error(f"Failed to load data: {e}")


# MAIN CONTENT

if "df" in st.session_state:

    df=st.session_state["df"]
    ticker_display=st.session_state["ticker"]
    info=st.session_state.get("info",{})
    currency_symbol=st.session_state["currency_symbol"]


    # FIX 2: prevent iloc crash
    if "Close" not in df.columns or df["Close"].empty:
        st.error("Price data unavailable.")
        st.stop()


    current_price=df["Close"].iloc[-1]
    prev_price=df["Close"].iloc[-2] if len(df)>1 else current_price

    change=current_price-prev_price
    change_pct=(change/prev_price)*100


    col1,col2,col3,col4=st.columns(4)

    with col1:

        st.markdown(f"""
        <div class="info-card">
        <b>{ticker_display}</b><br>
        <span style="font-size:1.8rem">{currency_symbol}{current_price:.2f}</span><br>
        {change:+.2f} ({change_pct:+.2f}%)
        </div>
        """,unsafe_allow_html=True)


    tab1,tab2,tab3,tab4=st.tabs(
        ["📊 Price Chart","🔍 Technical","🧠 Train","🔮 Predict"]
    )


    # PRICE CHART

    with tab1:

        fig=make_subplots(rows=2,cols=1,shared_xaxes=True)

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"]
            ),
            row=1,col=1
        )

        colors=["#2ca02c" if c>=o else "#d62728"
                for c,o in zip(df["Close"],df["Open"])]

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                marker_color=colors
            ),
            row=2,col=1
        )

        fig.update_layout(height=600,template="plotly_dark")

        st.plotly_chart(fig,use_container_width=True)


    # TRAIN MODEL

    with tab3:

        if btn_train:

            try:

                result=train_pipeline(
                    ticker_display,
                    start_date=str(start_date),
                    end_date=str(end_date)
                )

                st.session_state["train_result"]=result
                st.success("Model trained")

            except Exception as e:

                st.error(f"Training failed: {e}")


    # PREDICT

    with tab4:

        if btn_predict:

            try:

                if "train_result" in st.session_state:
                    model=st.session_state["train_result"]["model"]
                else:
                    model=joblib.load(MODEL_PATH)


                df_fresh=load_data(
                    ticker_display,
                    start=str(start_date),
                    end=str(end_date),
                    convert_to_inr=convert_to_inr
                )


                # FIX 3: prevent fresh data crash
                if df_fresh is None or df_fresh.empty:
                    st.error("No fresh data available.")
                    st.stop()


                X,y,df_feat=create_features(df_fresh)


                # FIX 4: prevent feature empty crash
                if X.empty:
                    st.error("Feature data not available.")
                    st.stop()


                last_row=X.iloc[-1:]

                prediction=model.predict(last_row)[0]

                last_close=df_fresh["Close"].iloc[-1]

                change=prediction-last_close
                change_pct=(change/last_close)*100


                st.success(f"Predicted next close: {currency_symbol}{prediction:.2f}")

                st.write(f"Change: {change:+.2f} ({change_pct:+.2f}%)")

            except Exception as e:

                st.error(f"Prediction failed: {e}")

else:

    st.info("👈 Select a stock and click Load Data to start.")