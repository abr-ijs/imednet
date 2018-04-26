from DMP_layer import  DMP_integrator
import torch
import torch.nn as nn
from torch.autograd import Variable

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.func =DMP_integrator(20,3,0.01,2)
        self.inputLayer = torch.nn.Linear(1, 1)


    def forward(self, x):
        #x =  nn.functional.relu(x)
        #x = self.inputLayer(x)
        x = self.func(x)
        return x


net = Net()
print(net)
print(list(net.parameters()))

start = Variable(torch.ones(1,44), requires_grad=True)

results = net(start)

print(results.data)

results.backward(torch.ones(1,1))

print(start.grad)