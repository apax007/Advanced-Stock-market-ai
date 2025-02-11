import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from prophet import Prophet
from arch import arch_model

# Cache stock data for 1 hour
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, period="5y"):
    try:
        data = yf.Ticker(symbol).history(period=period)
        if data.empty:
            return None
        data.fillna(method='ffill', inplace=True)  # Fill missing values
        return data
    except:
        return None

# Calculate technical indicators
def calculate_indicators(data, period):
    lookback = {"1mo": 14, "3mo": 20, "6mo": 50, "1y": 100, "5y": 200}.get(period, 50)

    data['SMA'] = ta.trend.sma_indicator(data['Close'], lookback)
    data['EMA'] = ta.trend.ema_indicator(data['Close'], lookback)
    data['RSI'] = ta.momentum.rsi(data['Close'], lookback)
    data['MACD'] = ta.trend.macd(data['Close'])
    data['VWAP'] = ta.volume.volume_weighted_average_price(data['High'], data['Low'], data['Close'], data['Volume'])
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'], lookback)

    return data.dropna()

# Prophet Forecast Model
def prophet_forecast(data):
    df = data[['Close', 'Volume', 'RSI', 'MACD']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.add_regressor('Volume')
    model.add_regressor('RSI')
    model.add_regressor('MACD')
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    future['Volume'] = np.mean(df['Volume'])
    future['RSI'] = np.mean(df['RSI'])
    future['MACD'] = np.mean(df['MACD'])
    return model.predict(future)

# LSTM Model for Stock Price Prediction
def train_lstm_model(data):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'SMA', 'EMA', 'VWAP', 'ADX']
    data = data[features].dropna()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(50, len(scaled_data) - 1):
        X.append(scaled_data[i-50:i])
        y.append(scaled_data[i, 3])  # Predict 'Close' price

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        GRU(100, return_sequences=False),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    return model, scaler

# Risk Analysis using GARCH Model
def risk_analysis(data):
    returns = data['Close'].pct_change().dropna()
    garch = arch_model(returns * 100, vol='Garch', p=1, q=1)
    garch_fit = garch.fit(disp="off")
    volatility = garch_fit.conditional_volatility[-1]

    max_drawdown = (1 - (data['Close'] / data['Close'].cummax())).max() * 100
    risk_free_rate = 0.06
    sharpe_ratio = (returns.mean() - risk_free_rate / 252) / returns.std() * np.sqrt(252)

    var_95 = np.percentile(returns, 5) * 100
    cvar_95 = np.mean([x for x in returns if x <= var_95 / 100]) * 100

    return {
        "Volatility": volatility,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
        "VaR (95%)": var_95,
        "CVaR (95%)": cvar_95
    }

# Streamlit UI
def main():
    st.set_page_config(layout="wide", page_title="AI Stock Analysis")
    st.title("📈 AI-Powered Stock Market Analysis")

    with st.sidebar:
        symbol = st.text_input("Stock Symbol (e.g., RELIANCE.NS):", "RELIANCE.NS")
        timeframe = st.selectbox("Select Timeframe", ["1mo", "3mo", "6mo", "1y", "5y"])
        if st.button("Analyze"):
            st.session_state.run_analysis = True

    if "run_analysis" in st.session_state:
        with st.spinner("Fetching and analyzing data..."):
            data = fetch_stock_data(symbol, timeframe)
            if data is None or data.empty:
                st.error("Failed to fetch data. Try another symbol.")
                return

            data = calculate_indicators(data, timeframe)

            tab1, tab2, tab3, tab4 = st.tabs(["📊 Stock Chart", "📈 Indicators", "🔮 AI Predictions", "📉 Risk Analysis"])

            with tab1:
                st.subheader(f"📊 {symbol} Price Chart")
                st.line_chart(data['Close'])

            with tab3:
                st.subheader("🔮 AI Price Prediction")
                model, scaler = train_lstm_model(data)
                last_50_days = scaler.transform(data.iloc[-50:][['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'SMA', 'EMA', 'VWAP', 'ADX']].values.reshape(1, 50, -1))
                predicted_price = model.predict(last_50_days)
                predicted_price = scaler.inverse_transform(np.array([0, 0, 0, predicted_price[0][0], 0, 0, 0, 0, 0, 0, 0]).reshape(1, -1))[0][3]
                st.metric("Predicted Close Price", f"₹{predicted_price:.2f}")

            with tab4:
                st.subheader("📉 Risk Analysis")
                risk = risk_analysis(data)
                st.json(risk)

if __name__ == "__main__":
    main()
