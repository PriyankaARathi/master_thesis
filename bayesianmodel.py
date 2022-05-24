import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameter
# Number of days data to predict one day in future
input_sz = 30
h1_size = 500
lr = 0.01
num_epochs = 1000
test_ens = 20

#Step 1: Preparing dataset
# load train dataset
train_data = pd.read_csv("data/AAPL_train.csv")
train_data = train_data.sort_values('Date')

# prepare train data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))

x_train = []
y_train = []

# predicting 1 value after 60 days
for i in range(input_sz, len(scaled_data)):
    x_train.append(scaled_data[i - input_sz:i, 0])
    y_train.append(scaled_data[i, 0])

# converting to numpy array
x_train = np.array(x_train)
y_train = np.array(y_train)

# converting to tensors
x_train = torch.from_numpy(x_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)

# load test dataset
test_data = pd.read_csv("data/AAPL_test.csv")
test_data = test_data.sort_values('Date')

# prepare test data
actual_price = np.array(test_data[['Close']])
actual_price = actual_price.reshape(-1, 1)
actual_price = scaler.transform(actual_price)
y_actual = torch.from_numpy(actual_price.astype(np.float32))

total_dataset = pd.concat((train_data['Close'], test_data['Close']), axis=0)

test_model_ip = total_dataset[len(total_dataset)-len(test_data) - input_sz:].values
test_model_ip = test_model_ip.reshape(-1, 1)
test_model_ip = scaler.transform(test_model_ip)

x_test = []

for i in range(input_sz, len(test_model_ip)):
    x_test.append(test_model_ip[i - input_sz:i, 0])

# converting to numpy array
x_test = np.array(x_test)

# converting to tensors
x_test = torch.from_numpy(x_test.astype(np.float32))


# Step 2: Building Model class
@variational_estimator
class BNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BNNetwork, self).__init__()
        self.bl1 = BayesianLinear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.bl2 = BayesianLinear(hidden_size, 1)

    def forward(self, x):
        out = self.bl1(x)
        out = self.relu(out)
        out = self.bl2(out)
        return out


model = BNNetwork(input_sz, h1_size).to(device)

# step 3: loss & optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# step 4: training
for epoch in range(num_epochs):
    # forward pass and loss
    #y_pred = model(x_train)
    loss = model.sample_elbo(inputs=x_train.to(device),
                                 labels=y_train.to(device),
                                 criterion=criterion,
                                 sample_nbr=3,
                                 complexity_cost_weight=1 / x_train.shape[0])

    # backward
    loss.backward()

    # updates
    optimizer.step()
    # zero gradient
    optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        print(f'epoch: {epoch + 1}, loss= {loss.item():.4f}')

with torch.no_grad():
    print('testing the results')
    y_pred = model(x_test)
    MSE_err = ((y_actual-y_pred) ** 2 ).mean()
    print(f'MSE error = {MSE_err:.4f}')

#Plot test
plt.plot(y_actual, color='blue', label=f"Actual Apple Price")
plt.plot(y_pred, color='red', label=f"Predicted Apple Price")
plt.xlabel('Days')
plt.ylabel("share price")
plt.legend()
plt.show()
