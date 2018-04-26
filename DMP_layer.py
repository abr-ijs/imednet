import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np

class DMP_integrator(Function):



    def __init__(self,N, tau, dt,Dof):
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

    def forward(self, inputs):

        w = np.zeros((self.N, self.Dof))
        y0 = np.zeros((self.Dof))

        goal = np.zeros((self.Dof))

        Y = np.zeros((self.time_steps, self.Dof))

        for i in range(0,self.Dof):

            Y[:,i] = self.integrate(inputs[0, (i * self.N):((i + 1) * self.N)].numpy(), inputs[0,(i+1)*(self.N)],self.dy0[i], inputs[0,(i+1) * (self.N + 1)], self.tau)


        #N = dmp_composition.N
        #dt = dmp_composition.dt


        #result = inputs *3



        return inputs.new(Y)




    def backward(self, grad_outputs):
        point_grads = np.zeros((self.N+2)*self.Dof)

        for j in range(0,self.Dof):
            #weights
            for i in range(0,self.N):
                weights = np.zeros((self.N))
                weights[i] = 1

                grad = self.integrate(weights, 0 ,0, 0, self.tau)

                point_grads[j*(self.Dof+1)+i] = sum(grad*grad_outputs)

            #start_point
            weights = np.zeros((self.N))
            point_grads[j*(self.Dof+1) +self.Dof] = self.integrate(weights, 1, 0, 0, self.tau)

            #goal



        grad=grad_outputs*2

        return grad_outputs.new(grad)


    def integrate(self,w,y0,dy0,goal,tau):
        Y = np.zeros((self.time_steps))
        y = y0

        z = dy0 * tau

        x = 1

        for i in range(1, self.time_steps):
            # state = self.DMP_integrate(state, dt )

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








