import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Prepare dataset
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1) # This is double type

# Below is of 1D
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

# COnverting to 2D
y = y.view(y.shape[0], 1) # view reshape tensor with 1 column

n_samples, n_features = X.shape

# Design Model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# Loss & Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # Backward pass
    loss.backward()

    # Update
    optimizer.step()

    # Empty our gradients. Because on backward, it will sum up the gradients from .grad function
    optimizer.zero_grad()

    if (epoch+1)%10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# Plot
predicted = model(X).detach() # required_grad=False on predicted.
predicted = predicted.numpy()

plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()