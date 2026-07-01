import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib

# ---------------- LOAD MODEL ----------------

model = joblib.load('models/alphatrade_rf.pkl')

# Features used during training
features = [
    'Close',
    'Volume',
    'SMA_20',
    'SMA_50',
    'EMA_20',
    'EMA_50',
    'Daily_Return',
    'Volatility',
    'RSI',
    'MACD',
    'Signal_Line'
]


# ---------------- FEATURE ENGINEERING ----------------

def create_features(df):

    # Moving averages
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()

    # Exponential moving averages
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()

    # Daily returns
    df['Daily_Return'] = df['Close'].pct_change()

    # Volatility
    df['Volatility'] = df['Daily_Return'].rolling(20).std()

    # RSI
    delta = df['Close'].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()

    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()

    return df


# ---------------- STREAMLIT APP ----------------

st.set_page_config(
    page_title="AlphaTrade",
    layout="wide"
)

st.title("AlphaTrade")
st.subheader("AI Powered Multi-Market Stock Prediction System")

ticker = st.text_input(
    "Enter Stock Symbol"
)

st.caption(
    "Examples: AAPL, MSFT, NVDA, RELIANCE, TCS, INFY, JIOFIN"
)

if st.button("Predict"):

    try:

        original_ticker = ticker.strip().upper()

        # ---------------- AUTO MARKET DETECTION ----------------

        # Try original symbol
        df = yf.download(
            original_ticker,
            start="2020-01-01",
            progress=False
        )

        detected_symbol = original_ticker

        # Try NSE
        if df.empty:
            temp = yf.download(
                original_ticker + ".NS",
                start="2020-01-01",
                progress=False
            )

            if not temp.empty:
                df = temp
                detected_symbol = original_ticker + ".NS"

        # Try BSE
        if df.empty:
            temp = yf.download(
                original_ticker + ".BO",
                start="2020-01-01",
                progress=False
            )

            if not temp.empty:
                df = temp
                detected_symbol = original_ticker + ".BO"

        # Final validation
        if df.empty:
            st.error("Invalid stock symbol.")
            st.stop()

        # ---------------- STOCK INFO ----------------

        stock = yf.Ticker(detected_symbol)

        try:
            info = stock.fast_info
            currency = info.get('currency', 'USD')
        except:
            currency = 'USD'

        # Currency symbols
        currency_symbols = {
            'USD': '$',
            'INR': '₹',
            'GBP': '£',
            'EUR': '€',
            'JPY': '¥'
        }

        currency_symbol = currency_symbols.get(
            currency,
            currency + " "
        )

        # Fix MultiIndex issue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # ---------------- FEATURE ENGINEERING ----------------

        df = create_features(df)
        df = df.dropna()

        # ---------------- CURRENT INFO ----------------

        current_price = float(df['Close'].iloc[-1])
        latest_date = df.index[-1]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Current Price",
                f"{currency_symbol}{current_price:.2f}"
            )

        with col2:
            st.metric(
                "Currency",
                currency
            )

        with col3:
            st.metric(
                "Detected Symbol",
                detected_symbol
            )

        with col4:
            st.metric(
                "Last Updated",
                str(latest_date.date())
            )

        # ---------------- PREDICTION ----------------

        latest_data = df[features].iloc[-1:]

        prediction = model.predict(
            latest_data
        )[0]

        st.subheader("AlphaTrade Prediction")

        if prediction == 1:
            st.success(
                "BUY — Market expected to move UP"
            )
        else:
            st.error(
                "SELL — Market expected to move DOWN"
            )

        # ---------------- STOCK DATA ----------------

        st.subheader("Recent Stock Data")

        st.dataframe(
            df.tail(),
            width="stretch"
        )

    except Exception as e:
        st.error(f"Error: {e}")