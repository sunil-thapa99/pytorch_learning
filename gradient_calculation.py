'''
    Gradient Calculation with Autograd
'''
import torch

x = torch.randn(3, requires_grad=True) # requires_grad=False

y = x+2
print(y) # tensor([3.0433, 1.5157, 1.1591], grad_fn=<AddBackward0>) Add Backward Gradiation function

z = y*y*2
print(z) # tensor([18.5228,  4.5950,  2.6870], grad_fn=<MulBackward0>) Multiplication Backward Gradiation function

z = z.mean() # z is scalar vector here
print(z) # tensor(8.6016, grad_fn=<MeanBackward1>) Mean Backward Gradiation function

# Calculate gradient
z.backward() # dz/dx
print(x.grad)

# If mean wasn't applied
# Vector Jacobian
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32) 
z.backward(v) # Multiplying z with vector v
print(x.grad)

'''
    # Creates new tensor that doesn't require gradient
    x.detach()
    with torch.no_grad():
        y = x + 2
        print(y)
'''

weights = torch.ones(4, requires_grad=True)
for epoch in range(2):
    model_output = (weights*3).sum()
    # Calculate gradient
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()

# Builtin optimizer
optimizer = torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad() # Clears gradient of optimized tensor
