import torch

"""
Followed tutorial from:
https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
# requires_grad makes autograd record operations on the tensor in the computational DAG for backprop
print(a)

b = torch.sin(a)
plt.plot(a.detach(), b.detach())
print(b)

c = 2 * b
print(c)

d = c + 1
print(d)

out = d.sum()
print(out)

print('d:')
print(d.grad_fn) # tracks history of operations on the tensor
print(d.grad_fn.next_functions) # contains a memory address and function
print(d.grad_fn.next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
print('\nc:')
print(c.grad_fn)
print('\nb:')
print(b.grad_fn)
print('\na:')
print(a.grad_fn) # no operations were done on a

out.backward()
print(a.grad)

plt.plot(a.detach(), a.grad.detach())

BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10

class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()

        self.layer1 = torch.nn.Linear(DIM_IN, HIDDEN_SIZE)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(HIDDEN_SIZE, DIM_OUT)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

model = TinyModel()

print(model.layer2.weight[0][0:10]) # just a small slice
print(model.layer2.weight.grad)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

prediction = model(some_input)

loss = (ideal_output - prediction).pow(2).sum() # square of euclidean
print(loss)

loss.backward() # computes gradients
print(model.layer2.weight[0][0:10]) # weights unchanged
print(model.layer2.weight.grad[0][0:10]) # gradients computed

optimizer.step()
print(model.layer2.weight[0][0:10]) # weights changed
print(model.layer2.weight.grad[0][0:10]) 

for i in range(0, 5): # this will cause gradients to blow up, unless gradients reset
    prediction = model(some_input)
    loss = (ideal_output - prediction).pow(2).sum()
    loss.backward()

print(model.layer2.weight.grad[0][0:10]) 

optimizer.zero_grad(set_to_none=False)

print(model.layer2.weight.grad[0][0:10]) 