# Examples using PyTorch autograd to calculate loss gradient

import torch.nn.functional as F
import torch
from torch.autograd import grad

# Logistic regression forward pass (prediction step)
# Simple logistic regression classifier = single-layer Neural Network

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = x1 * w1 + b
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print ("Manual loss gradient for w1 is:")
print (grad_L_w1)

print ("Manual loss gradient for b is:")
print (grad_L_b)

print ("Backward function outputs")
loss.backward()

print ("w1.grad is: ")
print (w1.grad)

print ("b.grad is: ")
print (b.grad)

 

