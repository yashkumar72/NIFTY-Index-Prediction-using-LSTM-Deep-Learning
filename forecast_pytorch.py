import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class PriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions


def prepare_data(ticker="^NSEI"):
    df = yf.download(ticker, period="5y", interval="1d")
    data = df[['Close']].values.astype(float)
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    
    
    seq_length = 30
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length])
        
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)), scaler


X, y, scaler = prepare_data()
model = PriceLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("--- Starting Deep Learning Training (PyTorch) ---")
for epoch in range(25):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f'Epoch {epoch} | Loss: {loss.item():.6f}')


model.eval()
with torch.no_grad():
    last_seq = torch.FloatTensor(X[-1:]).view(1, 30, 1)
    prediction = model(last_seq)
    predicted_price = scaler.inverse_transform(prediction.numpy())

print(f"\n✅ Next-Day Inflation Proxy Prediction: points :{predicted_price[0][0]:.2f}")