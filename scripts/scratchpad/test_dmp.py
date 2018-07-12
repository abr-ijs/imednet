import sys
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn

from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

from imednet.utils.dmp_layer import DMPIntegrator, DMPParameters
from imednet.trainers.encoder_decoder_trainer import Trainer
from imednet.data.mat_loader import MatLoader


class Net(nn.Module):
    def __init__(self, scale):
        super(Net, self).__init__()
        self.input_layer = torch.nn.Linear(54, 54)
        self.func = DMPIntegrator()
        self.DMPparam = DMPParameters(25, 3, 0.01, 2, scale)

        self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('scale', self.DMPparam.scale_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)

    def forward(self, x):
        # x =  nn.functional.relu(x)
        x = self.input_layer(x)

        x = self.func.apply(x, self.DMPp,self.param_grad,self.scale)
        return x


'''self.DMPparam.data'''


grads = {}

def printgradnorm(grad):
    grads['t']=grad


class ct:

    def __init__(self,net,division):
        self.param = net.DMPp

        self.grad = net.param_grad
        self.scale = net.scale[0:division]


dataset_name = '/home/rpahic/imednet/slike_780.4251'

images, outputs, scale, original_trj = MatLoader.load_data(dataset_name, load_original_trajectories=True)
net = Net(scale)
original_trj_e = []
for i in range(0,images.shape[0]):
    c,c1,c2 = zip(*original_trj[i])
    original_trj_e.append(c)
    original_trj_e.append(c1)



print(net)
n=0

for p in list(net.parameters()):
    if p.data.ndimension() == 1:
        torch.nn.init.constant_(p, 0)
    else:

        p.data=torch.eye(54)

d= 9000
test_output = Variable(torch.from_numpy(np.array(outputs[40+n:40+d+n])), requires_grad=True).float()

input_image = Variable(torch.from_numpy(np.array(images[40+n:40+d+n]))).float()
traj=Variable(torch.from_numpy(np.array(original_trj_e[40*2+n:40*2+2*d+n]))).float()

optimizer = torch.optim.Adam(net.parameters(), eps=0.001)
optimizer.zero_grad()


trainer = Trainer()

dmp = trainer.create_dmp(test_output[0], scale, 0.01, 25, cuda=True)
dmp2 = trainer.create_dmp(test_output[1], scale, 0.01, 25, cuda=True)
dmp.joint()
dmp2.joint()


DMP = DMPIntegrator(25, 3, 0.01, 2, scale)
criterion = torch.nn.MSELoss(size_average=True)
criterion2 = torch.nn.MSELoss()



net = net.cuda()


#test_output1=torch.cat((test_output[:,0:5].transpose(1,0),test_output[:,5:].contiguous().view(50,-1)))
#test_output1=test_output1.transpose(1,0)

test_output = test_output.cuda()
traj=traj.cuda()
input_image = input_image.cuda()
#trajektorija = DMP.forward(test_output)
'''trajektorija = net(test_output)
gradients = DMP.backward(torch.ones((300, 2)))
to = Variable(torch.from_numpy(dmp.Y)).float()
loss_vector = criterion2(trajektorija, to)
loss_vector.backward()'''


points = 10
loss_graph = np.zeros((points, 3))
n_w = 6
start_w = test_output[0, n_w+1].item()
print(start_w)






for j in range(0,points):
    optimizer.zero_grad()
    test_output.data[100, n_w+1] = start_w+(j-(points/2))/(points/2)

    trajektorija = net(test_output[:, 1:55])





    to = torch.from_numpy(dmp.Y).float().transpose(1,0).cuda()

    loss = criterion(trajektorija[:, :], traj)

    to = torch.cat((to,torch.from_numpy(dmp2.Y).float().transpose(1,0).cuda()))

    loss_vector = criterion2(trajektorija[:, :], traj)
    '''f=torch.ones(2, 54).cuda()
    f.requires_grad=True
    r=DMP.apply(f, net.DMPp,net.param_grad,net.scale)
    r.backward(torch.ones((300, 4)))
    DMP.backward(DMP.ctx,torch.ones((300, 4)))
    DMP.backward(torch.ones((300, 4)))'''
    trajektorija.register_hook(printgradnorm)
    #trajektorija.register_hook(print)

    loss_vector.backward()
    division = 2 * (int(net.DMPp[1].item()) + 2)
    CT = ct(net, division)
    #print(grads['t'].data)

    gradi = DMPIntegrator.backward(CT, grads['t'].data)

    loss_graph[j, 0] = test_output[100, n_w+1].item()
    loss_graph[j, 1] = loss_vector.item()
    print(loss_vector.item())

    #loss_graph[j, 2] = net.param_grad[n_w, n_w].item()/test_output[0,n_w+1].item()
    #print(gradi[0][0,n_w+1])
    #print(gradi[0])
    loss_graph[j, 2] = net.input_layer.weight.grad.data[n_w, n_w] / test_output.data[0,n_w + 1]


#print(loss_vector.grad_fn.next_functions[0][0])

#plt.plot(trajektorija[1, :].cpu().detach().numpy())
#plt.plot(to[1, :].cpu().detach().numpy())
#plt.show

gradient = np.gradient(loss_graph[:, 1])/np.gradient(loss_graph[:, 0])
plt.plot(loss_graph[:, 0], loss_graph[:, 1], label="test1")
plt.plot(loss_graph[:, 0], gradient, label="test2")

plt.plot(loss_graph[:, 0], loss_graph[:, 2], label="test3")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
#gradient/loss_graph[:,2]
back_gradienti = net.input_layer.weight.grad.data[:, 0] / (-1)#test_output.data
#mat = trainer.show_dmp(input_image.data.numpy(), trajektorija.data.numpy(), dmp, plot=True)