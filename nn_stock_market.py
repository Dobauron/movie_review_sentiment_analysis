import yfinance as yf
import pandas as pd
import numpy as np

# Pobranie danych historycznych (ceny akcji)
ticker = "O"  # Przykładowa spółka (Apple)
stock_data = yf.Ticker(ticker)
history = stock_data.history(period="5y", interval="1d")  # 5 lat danych dziennych

# Pobranie danych fundamentalnych (kwartalne)
financials = stock_data.quarterly_financials
balance_sheet = stock_data.quarterly_balance_sheet
cashflow = stock_data.quarterly_cashflow
print("Financials columns:", financials.index.tolist())
print("Balance Sheet columns:", balance_sheet.index.tolist())
print("Cashflow columns:", cashflow.index.tolist())
# Scalanie danych w jeden DataFrame
fundamentals = pd.concat([financials, balance_sheet, cashflow], axis=0)
fundamentals = fundamentals.drop_duplicates()
print(fundamentals.columns)

# Wybieramy kluczowe wskaźniki (możesz dostosować listę)
selected_metrics = [
    'Total Revenue', 'Gross Profit', 'EBITDA',
    'Diluted EPS', 'Operating Income', 'Interest Expense'
]

# Filtrujemy tylko wybrane wskaźniki
fundamentals_filtered = fundamentals.loc[selected_metrics]

# Transponujemy, aby daty były wierszami
fundamentals_filtered = fundamentals_filtered.T
fundamentals_filtered.index = pd.to_datetime(fundamentals_filtered.index)

# Łączymy z danymi cenowymi
merged_data = history.join(fundamentals_filtered, how='left')

# Wypełniamy braki danych (interpolacja)
merged_data = merged_data.ffill().bfill()


from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

# Dane wejściowe (X) i wyjściowe (y)
features = merged_data[['Open', 'High', 'Low', 'Close', 'Volume'] + selected_metrics].values
target = merged_data['Close'].values.reshape(-1, 1)

# Skalowanie danych (0-1)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_y.fit_transform(target)

# Podział na treningowy i testowy
train_size = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

# Konwersja do tensorów PyTorch
def create_sequences(X, y, seq_length=30):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return torch.FloatTensor(X_seq), torch.FloatTensor(y_seq)

seq_length = 30  # Okno czasowe (30 dni)
X_train_torch, y_train_torch = create_sequences(X_train, y_train, seq_length)
X_test_torch, y_test_torch = create_sequences(X_test, y_test, seq_length)

# DataLoader (do batchowania danych)
train_dataset = TensorDataset(X_train_torch, y_train_torch)
test_dataset = TensorDataset(X_test_torch, y_test_torch)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Bierzemy ostatni krok czasowy
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Parametry modelu
input_size = X_train_torch.shape[2]  # Liczba cech
hidden_size = 50
num_layers = 2
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
print(model)

import torch.optim as optim

# Funkcja straty i optymalizator
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trenowanie
num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Walidacja
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            test_loss += criterion(outputs, batch_y).item()
        test_loss /= len(test_loader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}')


model.eval()
with torch.no_grad():
    y_pred_torch = model(X_test_torch.to(device)).cpu().numpy()

# Odwrócenie skalowania
y_pred_actual = scaler_y.inverse_transform(y_pred_torch)
y_test_actual = scaler_y.inverse_transform(y_test_torch.numpy())

# Wykres
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Rzeczywiste ceny')
plt.plot(y_pred_actual, label='Przewidywane ceny (PyTorch LSTM)')
plt.legend()
plt.show()


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")