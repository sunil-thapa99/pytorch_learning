import torch
import numpy as np

# x = torch.empty(3)
x = torch.empty(3, dtype=torch.int) # torch.double, torch.float16
y = torch.empty(2, 3, 3) # 3-D tensor
# torch.zeros(), torch.ones()
print(x, y.dtype, x.size())

# Construct tensor from data
x = torch.tensor([2.5, 0.1])
print(x)

x = torch.rand(2, 2)
y = torch.rand(2, 2)

z = x + y # torch.add(x, y)
# y.add_(x) => y = x+y

x[1, 1].item() # Gives value of index [1, 1]

# Reshape tensor
x = torch.rand(4, 4)
y = x.view(16) # Reshaping to 1D
y = x.view(-1, 8) # resizing to 2D with size [2, 8]

# Converting numpy to tensor and vice versa
a = torch.ones(5)
b = a.numpy()
print(b)
a.add_(1) # Both a & b point to same reference
print(a, b)

a = np.ones(5)
b =torch.from_numpy(a)
print(b)

# This express, at some time this tensor need to calculate gradient for optimization
x = torch.ones(5, requires_grad=True)
