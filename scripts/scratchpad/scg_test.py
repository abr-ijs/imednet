import torch
from torch.autograd import Variable
import numpy as np

from network import Network
import correct_adam


layer_sizes = [1] + [3] + [1]

model = Network(layer_sizes, False)

for p in list(model.parameters()):
    torch.nn.init.constant(p.data, 1)

inputs = np.array([[0],[2],[6],[3],[9]])
outputs = np.array([[0],[9],[4],[7],[1]])

input_data_train = Variable(torch.from_numpy(inputs)).double()
output_data_train = Variable(torch.from_numpy(outputs), requires_grad=False).double()
model = model.double()
model = model.cuda()
input_data_train = input_data_train.cuda()
output_data_train = output_data_train.cuda()

y_pred = model(input_data_train[0,0])

# For calculating loss (mean squared error)
criterion = torch.nn.MSELoss(size_average=True) 

optimizer = correct_adam.SCG(model.parameters())


def wrap():
    optimizer.zero_grad()
    y_pred = model(input_data_train)
    loss = criterion(y_pred, output_data_train)
    loss.backward()
    return loss


'''
y_pred = model(x) # output from the network
loss = criterion(y_pred,y) #loss
optimizer.zero_grad()# setting gradients to zero
loss.backward()# calculating gradients for every layer


optimizer.step()#updating weights'''

for i in range(0, 200):
    loss = optimizer.step(wrap)
