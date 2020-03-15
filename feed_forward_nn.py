# MNIST
# DataLoader, Transformation
# Multilayer Neural Net, activation function
# Loss and Optimizer
# Training Loop (Batch Training)
# Model Evaluation
# GPU support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyper parameters
input_size = 784 # Image size 28x28. on flatten it will be 784(28*28)
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST data
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# Create dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

'''
    examples = iter(train_loader)
    samples, labels = examples.next()

    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(samples[i][0], cmap='gray')
    plt.show()

    print(samples.shape, labels.shape)
        Output torch.Size([100, 1, 28, 28]) torch.Size([100])
        100 => Batch Size, 1 => Color Channel, 28, 28 => Image Size
'''

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out

model = NeuralNet(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # shape: 100, 1, 28, 28, input_size = 784, it should be 100, 784 (1*28*28)
        # reshape
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        # loss calculate
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%100==0:
            print(f'epoch: {epoch+1} / {num_epochs}, step: {i+1}/{n_total_steps}, loss: {loss.item():.4f}')

# Testing and evaluation
with torch.no_grad():
    n_correct_pred = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        # Caclculate the prediction
        outputs = model(images)

        _, prediction = torch.max(outputs, 1) # Return value and index
        n_samples += labels.shape[0]
        n_correct_pred = (prediction == labels).sum().item()

    acc = 100.0 * (n_correct_pred / n_samples)
    print(f'Accuracy: {acc}')