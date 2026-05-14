import streamlit as st
import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

st.title("📈 NIFTY Index Prediction using LSTM")

class PriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        return self.linear(lstm_out[:, -1, :])

@st.cache_resource
def load_and_train():
    df = yf.download("^NSEI", period="5y", interval="1d")
    data = df[['Close']].values.astype(float)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(data)

    seq_len = 30
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len])

    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y))

    model = PriceLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for _ in range(20):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

    return model, scaler, X

st.write("Training model... please wait ⏳")
model, scaler, X = load_and_train()

with torch.no_grad():
    last_seq = X[-1:].view(1, 30, 1)
    pred = model(last_seq)
    price = scaler.inverse_transform(pred.numpy())

st.success(f"📊 Predicted Next-Day NIFTY Value: points : {price[0][0]:.2f}")
