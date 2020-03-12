'''
    - epoch = 1 forward and backward pass of ALL training sample
    - batch_size  number of training samples in one forward & backward pass
    - number of iterations = number of passes, each pass using [batch_size] number of samples
    e.g. 100 samples, batch_size=20 --> 100/20 = 5 iterations for 1 epoch
'''

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples = 1
        self.n_samples = xy.shape[0]
        

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples

dataset = WineDataset()
# Batch_size = 4 provides 4 data each time. 
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=2)

# dataiter = iter(dataloader)
# data = dataiter.next()
# features, labels = data
# print(features, labels)

# training loop
learning_rate = 0.01
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
# print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for index, (inputs, labels) in enumerate(dataloader):
        # forward, backward pass and update weights
        if (index%5) == 0:
            print(f'epoch: {epoch}/{num_epochs}, step {index}/{n_iterations}, inputs {inputs.shape}')

# torchvision.datasets.MNIST()
# Pytorch datasets: fashion-mnist, cifar, coco 