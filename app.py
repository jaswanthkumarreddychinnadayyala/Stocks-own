import streamlit as st
import joblib
import pandas as pd
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Custom background
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1506744038136-46273834b3fb");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """, unsafe_allow_html=True
)

# App Title
st.markdown("<h1 style='color: #0072B5;'>📈 Tata Steel Stock Price Prediction App</h1>", unsafe_allow_html=True)
st.write("Enter the input feature values below:")

# Load model
model = joblib.load('xgb_model.joblib')

# Inputs in sidebar
st.sidebar.header("Feature Inputs 🎯")
MA7 = st.sidebar.number_input('MA7', value=50.0)
Daily_Return = st.sidebar.number_input('Daily Return', value=0.0, format="%.6f")
MA30 = st.sidebar.number_input('MA30', value=50.0)
RSI = st.sidebar.number_input('RSI', value=50.0)
MACD = st.sidebar.number_input('MACD', value=0.0, format="%.6f")
MACD_signal = st.sidebar.number_input('MACD Signal', value=0.0, format="%.6f")
Bollinger_High = st.sidebar.number_input('Bollinger High', value=60.0)
Bollinger_Low = st.sidebar.number_input('Bollinger Low', value=40.0)
EMA_12 = st.sidebar.number_input('EMA 12', value=50.0)
EMA_26 = st.sidebar.number_input('EMA 26', value=50.0)
ATR = st.sidebar.number_input('ATR', value=1.0)

input_data = pd.DataFrame([[
    MA7, Daily_Return, MA30, RSI, MACD, MACD_signal,
    Bollinger_High, Bollinger_Low, EMA_12, EMA_26, ATR
]], columns=[
    'MA7', 'Daily Return', 'MA30', 'RSI', 'MACD', 'MACD_signal',
    'Bollinger_High', 'Bollinger_Low', 'EMA_12', 'EMA_26', 'ATR'
])

# Initialize prediction history if not yet created
if 'history' not in st.session_state:
    st.session_state['history'] = []

if st.button('Predict Closing Price'):
    with st.spinner('Calculating prediction...'):
        time.sleep(1.5)
        prediction = model.predict(input_data)[0]
        st.success(f'Predicted Closing Price: ₹{prediction:.2f}')
        st.metric("Prediction", f"{prediction:.2f} ₹", delta="Up" if prediction > MA7 else "Down")

        # Save the input and prediction in session state history
        st.session_state['history'].append({
            'MA7': MA7,
            'Daily Return': Daily_Return,
            'MA30': MA30,
            'RSI': RSI,
            'MACD': MACD,
            'MACD_signal': MACD_signal,
            'Bollinger_High': Bollinger_High,
            'Bollinger_Low': Bollinger_Low,
            'EMA_12': EMA_12,
            'EMA_26': EMA_26,
            'ATR': ATR,
            'Prediction': prediction
        })

# If there is prediction history, show it and plot
if st.session_state['history']:
    hist_df = pd.DataFrame(st.session_state['history'])
    st.subheader("Prediction History")
    st.dataframe(hist_df)

    st.subheader("Predicted Closing Price Over Time")
    st.line_chart(hist_df['Prediction'])

    st.subheader("MA7 vs Predicted Closing Price")
    fig, ax = plt.subplots()
    ax.plot(hist_df['MA7'], hist_df['Prediction'], marker='o', linestyle='-')
    ax.set_xlabel('MA7')
    ax.set_ylabel('Predicted Price')
    st.pyplot(fig)

# Load historical stock data for candlestick chart
data = pd.read_csv('tatasteel_stock_data.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Candlestick chart
fig_candle = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
)])

fig_candle.update_layout(
    title='Tata Steel Stock Price Candlestick Chart',
    xaxis_title='Date',
    yaxis_title='Price (₹)'
)

st.subheader("Stock Price Candlestick Chart")
st.plotly_chart(fig_candle)
