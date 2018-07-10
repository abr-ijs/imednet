import torch
from torch.autograd import Function
import numpy as np

import pycuda.autoinit

from pycuda.compiler import SourceModule

x = torch.cuda.FloatTensor(8)

mod = SourceModule("""
#include <math.h>


__global__ void multiply_them(float *traj, float *dmp_parameters, float *c, float *sigma2, int n)
{
  const int idx_in = ((blockIdx.x * blockDim.x)+threadIdx.x)*54;
  const int idx_out =((blockIdx.x * blockDim.x)+threadIdx.x)*301*2;


  if(idx_in<n)
  {
      float x , dx;
      float fx,sum_psi, psi,a,y,z=0;

      for(int dof=0;dof<2;dof++)
      {
        x = 1.0;
        y=dmp_parameters[idx_in+dof];
        traj[idx_out+dof*301]=y;
        for(int i=0;i<300;i++)
          {
            fx = sum_psi = 0.0;

            for(int j=0;j<25;j++)
            {
                psi =exp(-0.5*((x-c[j])*(x-c[j])/sigma2[j]));
                fx = fx +(dmp_parameters[idx_in+4+j*2+dof]*psi);
                sum_psi = sum_psi + psi;

            }

            fx=(fx*x)/sum_psi;

            //a = alpha_z*(beta_z*(goal-y)-z)+fx

            a = 48*(12*(dmp_parameters[idx_in+2+dof]-y)-z)+fx;
            z = z + 0.01*a/3.0;
            y = y +0.01*z/3.0;

            //dx = -alpha_x*x/tau

            dx = -2.0*x/3.0;

            //x = x+dx*dt
            x = x+dx*0.01;

            //printf(" %d ",x);
            traj[1+i+idx_out+dof*301] = y;
        }
      }
  }
}
""")

multiply_them = mod.get_function("multiply_them")


class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()


class DMPIntegrator(Function):

    @staticmethod
    def forward(ctx, inputs, parameters, param_gradients, scaling):
        ctx.param = parameters
        ctx.grad = param_gradients

        division = 2*(int(parameters[1].item())+2)
        inputs_np = scaling[0:division] * (inputs - scaling[-1]) + scaling[division:division*2]
        ctx.scale = scaling[0:division]

        #w = torch.cat((inputs_np[:,range(2*int(parameters[0].item()),(2*int(parameters[0].item()) + int(parameters[1].item())*int(parameters[0].item()))-1,2)],
          #             inputs_np[:,range(1+2*int(parameters[0].item()),(2*int(parameters[0].item()) + int(parameters[1].item())*int(parameters[0].item())),2)]),1).view(-1,25)

        #X = integrate(parameters,w, inputs_np[:,range(0,int(parameters[0].item()))].view(int(parameters[0].item())*inputs.shape[0],), torch.zeros(inputs.shape[0]*int(parameters[0].item())).cuda(),
               #       inputs_np[:,range(int(parameters[0].item()),int(parameters[0].item())*2)].view(int(parameters[0].item())*inputs.shape[0],), 3)

        Y = torch.cuda.FloatTensor(2*inputs_np.shape[0], int(parameters[2].item())).fill_(0)

        n=inputs_np.shape[0]*inputs_np.shape[1]
        n=np.int32(n)

        k = int(1+inputs_np.shape[0]/1024)

        multiply_them(
            Holder(Y),
            Holder(inputs_np),
            Holder(parameters[6:(6+int(parameters[1].item()))]),  # c
            Holder(parameters[(6+int(parameters[1].item())):(6+int(parameters[1].item())*2)]),  # sigma_2
            n,
            block=(1024, 1, 1), grid=(k, 1))

        return inputs.new(Y)

    @staticmethod
    def backward(ctx, grad_outputs):
        parameters = ctx.param

        grad = ctx.grad
        scale = ctx.scale

        point_grads = torch.mm(grad_outputs,grad).view(-1,2,27).transpose(2,1).contiguous().view(1,-1,54).squeeze()

        # point_grads = 10*point_grads*scale*parameters[3].item()
        point_grads = point_grads * scale

        return grad_outputs.new(point_grads), None, None, None


def integrate(data, w, y0, dy0, goal, tau):
    y = y0
    z = dy0 * tau
    x = 1

    if w.is_cuda:
        # Y = torch.zeros((w.shape[0],int(data[2].item()))).cuda()
        Y = torch.cuda.FloatTensor(w.shape[0], int(data[2].item())).fill_(0)
    else:
        Y = torch.zeros((w.shape[0],int(data[2].item())))

    Y[:, 0] = y

    for i in range(0, int(data[2].item())-1):
        psi = torch.exp(-0.5 * torch.pow((x - data[6:(6+int(data[1].item()))]), 2) / data[(6+int(data[1].item())):(6+int(data[1].item())*2)])

        # fx = torch.sum((w[3,:] * x) * psi) / torch.sum(psi)
        fx = torch.mv(w*x, psi) / torch.sum(psi)

        dx = (-data[4].item() * x) / tau
        dz = data[5].item() * (data[5].item() / 4 * (goal - y) - z) + fx
        dy = z

        dz = dz / tau
        dy = dy / tau

        x = x + dx * data[3].item()
        y = y + dy * data[3].item()
        z = z + dz * data[3].item()

        Y[:, i+1] = y

    return Y


class DMPParameters():
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

        # for j in range(0, self.Dof):
            # weights

        for i in range(0, self.N):
            weights = torch.zeros((1,self.N))
            weights[0,i] = 1
            grad[:, i  + 2 ] = integrate(data_tensor, weights, 0, 0, 0, self.tau)
            # grad[:, i * self.Dof + 4 + j] = integrate(data_tensor, weights, 0, 0, 0, self.tau)

        # start_point
        weights = torch.zeros((1,self.N))
        # grad[:, j] = integrate(data_tensor, weights, 1, 0, 0, self.tau)
        grad[:, 0] = integrate(data_tensor, weights, 1, 0, 0, self.tau)

        # goal

        weights = torch.zeros((1,self.N))
        # grad[:, j + self.Dof] = integrate(data_tensor, weights, 0, 0, 1, self.tau)
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
