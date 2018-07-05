import torch
import torch.nn as nn
from torch.autograd import Variable

from deep_encoder_decoder_network.utils.dmp_layer import DMPIntegrator


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.func = DMPIntegrator(20, 3, 0.01, 2)
        self.input_layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        # x =  nn.functional.relu(x)
        # x = self.input_layer(x)
        x = self.func(x)
        return x


net = Net()
print(net)
print(list(net.parameters()))

start = Variable(torch.ones(1, 44), requires_grad=True)

results = net(start)

print(results.data)

results.backward(torch.ones(1, 1))

print(start.grad)
