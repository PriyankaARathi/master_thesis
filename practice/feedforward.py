import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load train dataset
data = pd.read_csv("../data/APPL.csv")
data = data.sort_values('Date')

# prepare data
price = data[['Close']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(price['Close'].values.reshape(-1, 1))

x = []
y = []
input_sz = 60

# predicting 1 value after 60 days
for i in range(input_sz, len(scaled_data)):
    x.append(scaled_data[i - input_sz:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(x)
Y = np.array(y)

n_features = X.shape[1]

# print(X.shape)
train_size = int(np.round(X.shape[0] * 0.8))
x_train = X[:train_size]
# print(x_train.shape)
y_train = Y[:train_size]
x_test = X[train_size:]
y_test = Y[train_size:]
# print(x_test.shape)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


class NNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NNetwork(input_sz, 100).to(device)

# step 3: loss & optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# step 4: training
num_epochs = 10000

for epoch in range(num_epochs):
    # forward pass and loss
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    # backward
    loss.backward()

    # updates
    optimizer.step()
    # zero gradient
    optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        print(f'epoch: {epoch + 1}, loss= {loss.item():.4f}')

with torch.no_grad():
    y_pred = model(x_test)
    acc = y_pred.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')




