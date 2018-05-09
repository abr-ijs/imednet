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
        self.c = np.exp(-self.a_x * np.linspace(0, 1, self.N))

        sigma2 = np.power((np.diff(self.c) / 2), 2)
        self.sigma2 = np.append(sigma2, sigma2[-1])
        self.tau = tau
        self.dt = dt
        self.time_steps = int(np.round(self.tau / self.dt))
        self.y0 = [0]
        self.dy0 = np.zeros(Dof)
        self.Dof = Dof

        self.x_max = torch.from_numpy(scale.x_max).float().cuda()
        self.x_min = torch.from_numpy(scale.x_min).float().cuda()
        self.y_max = scale.y_max
        self.y_min = scale.y_min


        self.K = (self.x_max[1:55] - self.x_min[1:55]) / (self.y_max - self.y_min)
        #precomputation


    def forward(self, inputs):



        Y = np.zeros((self.time_steps, self.Dof))

        inputs_np = self.K* (inputs - self.y_min)+ self.x_min[1:55]

        w = inputs_np[(2*self.Dof):(2*self.Dof+ self.N*self.Dof)].view(self.N,self.Dof)

        for i in range(0, self.Dof):

            Y[:, i] = self.integrate(w[:,i], inputs_np[i], self.dy0[i], inputs_np[self.Dof+i], self.tau)

        return inputs.new(Y)





    def backward(self, grad_outputs):
        point_grads = np.zeros(54)

        for j in range(0,self.Dof):
            #weights
            for i in range(0, self.N):
                weights = np.zeros((self.N))
                weights[i] = 1

                grad = self.integrate(weights, 0 ,0, 0, self.tau)

                point_grads[i*self.Dof + 4+j] = sum(grad*grad_outputs[:, j])

           #start_point
            weights = np.zeros((self.N))
            grad= self.integrate(weights, 1, 0, 0, self.tau)
            point_grads[j] = sum(grad * grad_outputs[:, j])

            #goal

            weights = np.zeros((self.N))
            grad=self.integrate(weights, 0, 0, 1, self.tau)
            point_grads[ j + self.Dof] = sum(grad * grad_outputs[:, j])

        '''     
        '''
        point_grads=point_grads*self.K
        return grad_outputs.new(point_grads)


    def integrate(self,w,y0,dy0,goal,tau):

        Y = np.zeros((self.time_steps))
        y = y0

        z = dy0 * tau

        x = 1

        for i in range(0, self.time_steps):


            psi = np.exp(-0.5 * np.power((x - self.c), 2) / self.sigma2)

            fx = np.sum((w * x) * psi) / np.sum(psi)

            dx = (-self.a_x * x) / tau
            dz = self.a_z * (self.a_z / 4 * (goal - y) - z) + fx
            dy = z

            dz = dz / tau
            dy = dy / tau

            x = x + dx * self.dt
            y = y + dy * self.dt
            z = z + dz * self.dt

            Y[i]=y

        return Y








