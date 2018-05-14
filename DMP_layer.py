import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from copy import  deepcopy


class DMP_integrator(Function):
    '''def __init__(self,N, tau, dt, Dof, scale):


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
        self.time_steps = int(np.round(self.tau / self.dt))
        self.y0 = [0]
        self.dy0 = np.zeros(Dof)
        self.Dof = Dof
        self.Y = torch.zeros((self.time_steps))
        #self.x_max = torch.from_numpy(scale.x_max).float().cuda()
        #self.x_min = torch.from_numpy(scale.x_min).float().cuda()
        self.y_max = scale.y_max
        self.y_min = scale.y_min
        self.x_max = torch.from_numpy(scale.x_max).float()
        self.x_min = torch.from_numpy(scale.x_min).float()


        self.K = (self.x_max[1:55] - self.x_min[1:55]) / (self.y_max - self.y_min)
        #precomputation
        grad = torch.zeros((300,54))

        for j in range(0, self.Dof):
            # weights
            for i in range(0, self.N):
                weights = torch.zeros((self.N))
                weights[i] = 1

                grad[:,i * self.Dof + 4 + j] = self.integrate(weights, 0, 0, 0, self.tau)



            # start_point
            weights = torch.zeros((self.N))
            grad[:,j] = self.integrate(weights, 1, 0, 0, self.tau)


            # goal

            weights = torch.zeros((self.N))
            grad[:, j + self.Dof] = self.integrate(weights, 0, 0, 1, self.tau)

        ''''''self.c = self.c.cuda()
        self.sigma2 = self.sigma2.cuda()
        self.grad = grad.cuda()
        self.point_grads = torch.zeros(54).cuda()''''''

        self.grad = grad
        self.point_grads = torch.zeros(54)
        self.X = np.zeros((self.time_steps, self.Dof))'''

    @staticmethod
    def forward(ctx, inputs, parameters):

        ctx.param = parameters



        inputs_np = parameters.K* (inputs - parameters.y_min)+ parameters.x_min[1:55]

        w = inputs_np[(2*parameters.Dof):(2*parameters.Dof+ parameters.N*parameters.Dof)].view(parameters.N,parameters.Dof)

        for i in range(0, parameters.Dof):
            parameters.X[:, i] = integrate(parameters.data,w[:,i], inputs_np[i], parameters.dy0[i], inputs_np[parameters.Dof+i],parameters.tau)

        return inputs.new(parameters.X)

    @staticmethod
    def backward(ctx, grad_outputs):
        parameters=ctx.param

        for j in range(0,ctx.Dof):
            #weights
            for i in range(0, ctx.N):
                parameters.point_grads[i*parameters.Dof + 4+j] = sum(parameters.grad[:,i*parameters.Dof + 4+j]*grad_outputs[:, j])

           #start_point

            parameters.point_grads[j] = sum(parameters.grad[:,j] * grad_outputs[:, j])

            #goal

            parameters.point_grads[ j + parameters.Dof] = sum(parameters.grad[:, j + parameters.Dof] * grad_outputs[:, j])

        '''     
        '''
        parameters.point_grads = parameters.point_grads*parameters.K
        return grad_outputs.new(parameters.point_grads)


def integrate(data,w,y0,dy0,goal,tau):


    y = y0

    z = dy0 * tau

    x = 1

    for i in range(0, data['time_steps']):


        psi = torch.exp(-0.5 * torch.pow((x - data['c']), 2) / data['sigma2'])

        fx = torch.sum((w * x) * psi) / torch.sum(psi)

        dx = (-data['a_x'] * x) / tau
        dz = data['a_z']* (data['a_z']/ 4 * (goal - y) - z) + fx
        dy = z

        dz = dz / tau
        dy = dy / tau

        x = x + dx * data['dt']
        y = y + dy * data['dt']
        z = z + dz * data['dt']

        data['Y'][i]=y

    return data['Y']


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
        self.time_steps = int(np.round(self.tau / self.dt))
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
        # precomputation
        grad = torch.zeros((300, 54))
        self.data={'time_steps':self.time_steps,'c':self.c,'sigma2':self.sigma2,'a_z':self.a_z,'a_x':self.a_x,'dt':self.dt,'Y':self.Y}
        for j in range(0, self.Dof):
            # weights
            for i in range(0, self.N):
                weights = torch.zeros((self.N))
                weights[i] = 1

                grad[:, i * self.Dof + 4 + j] = integrate(self.data, weights, 0, 0, 0, self.tau)

            # start_point
            weights = torch.zeros((self.N))
            grad[:, j] = integrate(self.data, weights, 1, 0, 0, self.tau)

            # goal

            weights = torch.zeros((self.N))
            grad[:, j + self.Dof] = integrate(self.data, weights, 0, 0, 1, self.tau)

        ''''''
        self.c = self.c.cuda()
        self.sigma2 = self.sigma2.cuda()
        self.grad = grad.cuda()
        self.point_grads = torch.zeros(54).cuda()
        ''''''

        self.grad = grad
        self.point_grads = torch.zeros(54)
        self.X = np.zeros((self.time_steps, self.Dof))
















