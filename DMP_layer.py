import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from copy import  deepcopy
class DMP_integrator(Function):


    def __init__(self,N, tau, dt, Dof, scale):


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

        '''self.c = self.c.cuda()
        self.sigma2 = self.sigma2.cuda()
        self.grad = grad.cuda()
        self.point_grads = torch.zeros(54).cuda()'''

        self.grad = grad
        self.point_grads = torch.zeros(54)
        self.X = np.zeros((self.time_steps, self.Dof))



    def forward(self, inputs):





        inputs_np = self.K* (inputs - self.y_min)+ self.x_min[1:55]

        w = inputs_np[(2*self.Dof):(2*self.Dof+ self.N*self.Dof)].view(self.N,self.Dof)

        for i in range(0, self.Dof):

            self.X[:, i] = self.integrate(w[:,i], inputs_np[i], self.dy0[i], inputs_np[self.Dof+i], self.tau)

        return inputs.new(self.X)





    def backward(self, grad_outputs):


        for j in range(0,self.Dof):
            #weights
            for i in range(0, self.N):
                self.point_grads[i*self.Dof + 4+j] = sum(self.grad[:,i*self.Dof + 4+j]*grad_outputs[:, j])

           #start_point

            self.point_grads[j] = sum(self.grad[:,j] * grad_outputs[:, j])

            #goal

            self.point_grads[ j + self.Dof] = sum(self.grad[:, j + self.Dof] * grad_outputs[:, j])

        '''     
        '''
        self.point_grads = self.point_grads*self.K
        return grad_outputs.new(self.point_grads)


    def integrate(self,w,y0,dy0,goal,tau):


        y = y0

        z = dy0 * tau

        x = 1

        for i in range(0, self.time_steps):


            psi = torch.exp(-0.5 * torch.pow((x - self.c), 2) / self.sigma2)

            fx = torch.sum((w * x) * psi) / torch.sum(psi)

            dx = (-self.a_x * x) / tau
            dz = self.a_z * (self.a_z / 4 * (goal - y) - z) + fx
            dy = z

            dz = dz / tau
            dy = dy / tau

            x = x + dx * self.dt
            y = y + dy * self.dt
            z = z + dz * self.dt

            self.Y[i]=y

        return self.Y








