import DMP_layer
import torch
from matLoader import matLoader
import numpy as np
from torch.autograd import Variable
from trainer import Trainer
import matplotlib.pyplot as plt
import torch.nn as nn



class Net(nn.Module):

    def __init__(self,scale):
        super(Net, self).__init__()
        self.inputLayer = torch.nn.Linear(54, 54)
        self.func =DMP_layer.DMP_integrator()
        self.DMPparam = DMP_layer.createDMPparam(25, 3, 0.01, 2, scale)




    def forward(self, x):
        ctx=2
        #x =  nn.functional.relu(x)
        x = self.inputLayer(x)
        r = 8
        x = self.func(ctx,x, r )
        return x

'''self.DMPparam.data'''








dateset_name = 'slike_780.4251'

images, outputs, scale, original_trj = matLoader.loadData(dateset_name, load_original_trajectories=True)
net = Net(scale)

print(net)
n=0
for p in list(net.parameters()):
    if p.data.ndimension() == 1:
        torch.nn.init.constant(p, 0)
    else:

        p.data=torch.eye(54)


test_output = Variable(torch.from_numpy(np.array(outputs[40+n])), requires_grad=True).float()

input_image = Variable(torch.from_numpy(np.array(images[40+n]))).float()

optimizer = torch.optim.Adam(net.parameters(), eps=0.001)
optimizer.zero_grad()


trainer = Trainer()

dmp = trainer.createDMP(test_output, scale, 0.01, 25, cuda=True)
dmp.joint()


DMP = DMP_layer.DMP_integrator(25, 3, 0.01, 2, scale)
criterion = torch.nn.MSELoss(size_average=True)
criterion2 = torch.nn.MSELoss()



net = net.cuda()
test_output = test_output.cuda()

input_image = input_image.cuda()
#trajektorija = DMP.forward(test_output)
'''trajektorija = net(test_output)
gradients = DMP.backward(torch.ones((300, 2)))
to = Variable(torch.from_numpy(dmp.Y)).float()
loss_vector = criterion2(trajektorija, to)
loss_vector.backward()'''



loss_graph = np.zeros((100, 3))
n_w = 10
start_w = test_output.data[n_w+1]
for j in range(0,100):
    optimizer.zero_grad()
    test_output.data[n_w+1] = start_w+(j-50)/50
    trajektorija = net(test_output[1:55])




    to = Variable(torch.from_numpy(dmp.Y)).float().cuda()
    loss = criterion(trajektorija, to)
    loss_vector = criterion2(trajektorija, to)
    loss_vector.backward()
    loss_graph[j, 0] = test_output.data[n_w+1]
    loss_graph[j, 1] = loss_vector

    loss_graph[j, 2] = net.inputLayer.weight.grad.data[n_w, n_w]/test_output.data[n_w+1]
    test= net.inputLayer.weight.grad.data/ test_output.data[1:55]



gradient= np.gradient(loss_graph[:,1])/np.gradient(loss_graph[:,0])
plt.plot(loss_graph[:,0],loss_graph[:,1],label="test1")
plt.plot(loss_graph[:,0],gradient,label="test2")

plt.plot(loss_graph[:,0],loss_graph[:,2],label="test3")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

back_gradienti = net.inputLayer.weight.grad.data[:, 0] / (-1)#test_output.data
#mat = trainer.show_dmp(input_image.data.numpy(), trajektorija.data.numpy(), dmp, plot=True)
