import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from copy import  deepcopy












class DMP_integrator(Function):

    @staticmethod
    def forward(ctx, inputs, parameters, param_gradients, scaling):

        ctx.param = parameters
        ctx.grad = param_gradients


        division=2*(int(parameters[1].item())+2)
        inputs_np = scaling[0:division]* (inputs - scaling[-1]) + scaling[division:division*2]
        ctx.scale = scaling[0:division]



        w = torch.cat((inputs_np[:,range(2*int(parameters[0].item()),(2*int(parameters[0].item()) + int(parameters[1].item())*int(parameters[0].item()))-1,2)],
                       inputs_np[:,range(1+2*int(parameters[0].item()),(2*int(parameters[0].item()) + int(parameters[1].item())*int(parameters[0].item())),2)]),1).view(-1,25)


        X = integrate(parameters,w, inputs_np[:,range(0,int(parameters[0].item()))].view(int(parameters[0].item())*inputs.shape[0],), torch.zeros(inputs.shape[0]*int(parameters[0].item())).cuda(),
                      inputs_np[:,range(int(parameters[0].item()),int(parameters[0].item())*2)].view(int(parameters[0].item())*inputs.shape[0],), 3)


        return inputs.new(X)

    @staticmethod
    def backward(ctx, grad_outputs):

        parameters = ctx.param

        grad = ctx.grad
        scale = ctx.scale


        point_grads = torch.mm(grad_outputs,grad).view(-1,2,27).transpose(2,1).contiguous().view(1,-1,54).squeeze()


        point_grads = 10*point_grads*scale*parameters[3].item()
        #import pdb;
        #pdb.set_trace()

        return grad_outputs.new(point_grads),None,None,None


def integrate(data,w,y0,dy0,goal,tau):


    y = y0

    z = dy0 * tau

    x = 1
    if w.is_cuda==True:
        #Y = torch.zeros((w.shape[0],int(data[2].item()))).cuda()
        Y = torch.cuda.FloatTensor(w.shape[0], int(data[2].item())).fill_(0)
    else:
        Y = torch.zeros((w.shape[0],int(data[2].item())))


    for i in range(0, int(data[2].item())):


        psi = torch.exp(-0.5 * torch.pow((x - data[6:(6+int(data[1].item()))]),2) / data[(6+int(data[1].item())):(6+int(data[1].item())*2)])

        #fx = torch.sum((w[3,:] * x) * psi) / torch.sum(psi)
        fx = torch.mv(w*x,psi) / torch.sum(psi)

        dx = (-data[4].item() * x) / tau
        dz = data[5].item() * (data[5].item() / 4 * (goal - y) - z) + fx
        dy = z

        dz = dz / tau
        dy = dy / tau

        x = x + dx * data[3].item()
        y = y + dy * data[3].item()
        z = z + dz * data[3].item()

        Y[:,i]=y

    return Y


class createDMPparam():

    def __init__(self, N, tau, dt, Dof, scale):



        self.a_z = 48
        self.a_x = 2
        self.N = N
        c = np.exp(-self.a_x * np.linspace(0, 1, self.N))

        sigma2 = np.power((np.diff(c) / 2), 2)
        sigma2 = np.append(sigma2, sigma2[-1])
        self.c = torch.from_numpy(c).float()
        self.sigma2 = torch.from_numpy(sigma2).float()
        self.tau = tau
        self.dt = dt
        self.time_steps = int(np.round(self.tau / self.dt))+1
        self.y0 = [0]
        self.dy0 = np.zeros(Dof)
        self.Dof = Dof
        self.Y = torch.zeros((self.time_steps))
        # self.x_max = torch.from_numpy(scale.x_max).float().cuda()
        # self.x_min = torch.from_numpy(scale.x_min).float().cuda()
        self.y_max = scale.y_max
        self.y_min = scale.y_min
        self.x_max = torch.from_numpy(scale.x_max).float()
        self.x_min = torch.from_numpy(scale.x_min).float()

        self.K = (self.x_max[1:55] - self.x_min[1:55]) / (self.y_max - self.y_min)

        scale_tensor = torch.cat((self.K,self.x_min[1:55],self.x_max[1:55],torch.tensor([self.y_max,self.y_min]).float()),0)

        self.scale_tensor = scale_tensor
        # precomputation
        grad = torch.zeros((301, 27))

        self.data = {'time_steps':self.time_steps,'c':self.c,'sigma2':self.sigma2,'a_z':self.a_z,'a_x':self.a_x,'dt':self.dt,'Y':self.Y}
        dmp_data = torch.tensor([self.Dof,self.N,self.time_steps,self.dt,self.a_x,self.a_z])
        data_tensor = torch.cat((dmp_data,self.c,self.sigma2),0)

        data_tensor.dy0 = self.dy0
        data_tensor.tau = self.tau

        #for j in range(0, self.Dof):
            # weights

        for i in range(0, self.N):
            weights = torch.zeros((1,self.N))
            weights[0,i] = 1
            grad[:, i  + 2 ] = integrate(data_tensor, weights, 0, 0, 0, self.tau)
            #grad[:, i * self.Dof + 4 + j] = integrate(data_tensor, weights, 0, 0, 0, self.tau)

        # start_point
        weights = torch.zeros((1,self.N))
        #grad[:, j] = integrate(data_tensor, weights, 1, 0, 0, self.tau)
        grad[:, 0] = integrate(data_tensor, weights, 1, 0, 0, self.tau)

        # goal

        weights = torch.zeros((1,self.N))
        #grad[:, j + self.Dof] = integrate(data_tensor, weights, 0, 0, 1, self.tau)
        grad[:, 1] = integrate(data_tensor, weights, 0, 0, 1, self.tau)

        '''
        self.c = self.c.cuda()
        self.sigma2 = self.sigma2.cuda()
        self.grad = grad.cuda()
        self.point_grads = torch.zeros(54).cuda()
        '''
        self.data_tensor = data_tensor
        self.grad_tensor = grad


        self.point_grads = torch.zeros(54)
        self.X = np.zeros((self.time_steps, self.Dof))


















