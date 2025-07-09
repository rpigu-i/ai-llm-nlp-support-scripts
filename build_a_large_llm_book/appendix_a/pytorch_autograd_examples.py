# Examples using PyTorch automatic differentiation engine (autograd)

import torch.nn.functional as F
import torch

# Logistic regression forward pass (prediction step)
# Simple logistic regression classifier = single-layer Neural Network

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2])
b = torch.tensor([0.0])
z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

print ("Loss is:")
print (loss)

print ("Computational graph:")
print ("w1 (x) x1 -> [u = w1 (x) x1] -> (+ b) -> [z = u + b] -> [a = sigma(z)] -> [loss = L(a,y)]") 
