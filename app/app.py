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
    page_icon="📈",
    layout="wide"
)

st.title("📈 AlphaTrade")
st.subheader("AI Powered Stock Prediction System")

ticker = st.text_input(
    "Enter Stock Symbol",
    "AAPL"
)

if st.button("Predict"):

    try:

        # Download stock data
        df = yf.download(
            ticker,
            start="2020-01-01",
            progress=False
        )

        if df.empty:
            st.error("Invalid stock symbol.")
            st.stop()

        # Fix MultiIndex issue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Generate features
        df = create_features(df)

        # Remove NaN values
        df = df.dropna()

        # Current stock information
        current_price = float(df['Close'].iloc[-1])
        latest_date = df.index[-1]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Current Price",
                f"${current_price:.2f}"
            )

        with col2:
            st.metric(
                "Last Updated",
                str(latest_date.date())
            )

        # Prediction
        latest_data = df[features].iloc[-1:]

        prediction = model.predict(
            latest_data
        )[0]

        st.subheader("📊 AlphaTrade Prediction")

        if prediction == 1:
            st.success(
                "🟢 BUY — Market expected to move UP"
            )
        else:
            st.error(
                "🔴 SELL — Market expected to move DOWN"
            )

        # Show recent stock data
        st.subheader("📋 Recent Stock Data")

        st.dataframe(
            df.tail(),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error: {e}")