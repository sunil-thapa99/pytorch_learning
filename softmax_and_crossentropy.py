'''
### Numpy version of Softmax function
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

### Tensorflow version of Softmax function
    def softmax_tensorflow(scores):
        return tf.exp(scores)/tf.reducesum(tf.exp(scores), 1) 
'''

import torch
import torch.nn as nn
import numpy as np

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)

print(outputs)

# Cross entropy loss
'''
    D(Y_pred, Y) = -1/N*Σ(Y(i)*log(Y_pred(i)))
    Y => One-hot encoded class labels
    Y_pred => Probabilities (Softmax)

    def cross_entropy(actual, predicted):
        loss = -np.sum(actual * np.log(predicted))
        return loss
'''

'''
    CrossEntropyLoss already applies nn.LogSoftmax + nn.NLLLoss (Negative log likelihood loss)
    No Softmax in last layer
    Y has class labels, not One-hot encoded
    Y_pred has raw scores, no softmax scores (probabilities)
'''

loss = nn.CrossEntropyLoss()

'''
Y = torch.tensor([0]) 
# has shape of 1x3. nsample * nclasses
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f'Good loss: {l1.item()}, Bad loss: {l2.item()}')

_, prediction1 = torch.max(Y_pred_good, 1) # Along 1 dimension
_, prediction2 = torch.max(Y_pred_bad, 1) # Along 1 dimension
print(f'Good prediction: {prediction1}, Bad prediction: {prediction2}')
'''

# 3 samples
Y = torch.tensor([2, 0, 1]) # class labels 2, 0, 1
# nsample * nclasses = 3x3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 2.0, 3.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f'Good loss: {l1.item()}, Bad loss: {l2.item()}')

_, prediction1 = torch.max(Y_pred_good, 1) # Along 1 dimension
_, prediction2 = torch.max(Y_pred_bad, 1) # Along 1 dimension
print(f'Good prediction: {prediction1}, Bad prediction: {prediction2}')

'''
# Multiclass problem
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        # No softmax at the end
        return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()

# Binary problem
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        y_pred = torch.sigmoid(out)

        return y_pred

model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.CrossEntropyLoss()
'''