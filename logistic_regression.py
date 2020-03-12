import torch
import torch.nn as nn
import numpy as np
import sklearn.datasets as datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Prepare Data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1212)

# Sacle features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# Reshape y_train
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# Model
# f = wx +b, sigmoid function at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1) # 30 input features, 1 output features
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)

# Loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training Loop
n_epochs = 100

for epoch in range(n_epochs):
    # Forward pass & loss calculation
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

    # Empty gradient
    optimizer.zero_grad()

    if (epoch%10) == 0:
        print(f'Epoch: {epoch}, loss: {loss.item():.4f}')

# Evaluate model
with torch.no_grad():
    y_predicted = model(X_test)
    # convert to 0 or 1 if >=0.5 = 1, <0.5 = 0
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum().item() / float(y_test.shape[0]) # Y predicted class equalling to y_test
    print(f'accuracy = {acc:.4f}')