'''
    Activation function apply a non-linear transformation and decide whether a neuron should be activated or not.
    without activation functions our network is basically just a stacked linea regression model. 
    With non-linear transformations our network can learn better and perform more complex tasks. 

    Activation functions:
    - Step function:
        f(x) = 1 if x>=0 else 0
    - Sigmoid:
        Value range from 0-1. Typically use in last layer of binary classification problem
    - TanH:
        Value range from -1 to 1. Actually use for hidden layer
    - ReLu:
        Value 0 to infinity. 0 for negative value. It is non-linear function
    - Leaky ReLU
        f(x) = x if x>=0 else a(very small value).x
        Improved version of ReLU, trying to solve the vanishing gradient problem.
    - Softmax
        Good in last layer in multi-class classification problems. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# Option 1
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)

        return out

# Option2
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))

        return out