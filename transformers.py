'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of Dataset.

Complete list of built-in transforms:
https://pytorch.org/docs/stable/torchvision/transforms.html

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndarray
ToTensor: from numpy.ndarray or PILImage

Generic
-------
Use Lambda

Custom
------
Write own class

COmpose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256), RandomCrop(224)])

torchvision.transforms.ReScale(256)
torchvision.transforms.ToTensor()
'''

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np

class WineDataset(Dataset):
    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]

        self.transform = transform        

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples

# Custom transform class
class ToTensor:
    def __call__(self, sample):
        inputs, labels = sample
        return torch.from_numpy(inputs), torch.from_numpy(labels)

class MulTransform:
    # Multiplication transform
    def __init__(self,factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, labels = sample
        inputs *= self.factor
        return inputs, labels

dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels), features)

# Using compose transform
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels), features)
