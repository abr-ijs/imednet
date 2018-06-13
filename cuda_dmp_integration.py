import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import torch

x = torch.cuda.FloatTensor(8)

from pycuda.compiler import SourceModule
mod = SourceModule("""
#include <math.h>


__global__ void multiply_them(float *traj, float *dmp_parameters, float *b)
{
  const int idx = threadIdx.x;
  const int qe = threadIdx.y;
  
  float x = 1.0, dx=0.0;
  float fx,sum_psi, psi;
  
  for(int i=0;i<301;i++)
  {
    fx = sum_psi = 0.0;
    for(int j=0;j<25;j++)
    {
        psi = exp(-0.5*(x-1));
        fx = fx +(psi);
        sum_psi =sum_psi+ psi;
        
    }
    
    
    
    //y[] = y +dy*dz
    
    
    //dx = -alpha_x*x/tau    
    dx = -2.0*x/3.0;
    
    //x = x+dx*dt
    x = x+dx*0.01;
    //x = 0.0;
    //printf(" %d ",x);    
    traj[i+idx*301] = x;
  }
  //traj[q] = pow(dmp_parameters[q],b[q]);
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

a = np.ones((1,25)).astype(np.float32)
b = np.ones((1,25)).astype(np.float32)

a = torch.from_numpy(a).cuda()
b = torch.from_numpy(b).cuda()
dest = torch.Tensor(2,301).cuda()

multiply_them(
        Holder(dest),
        Holder(a),
        Holder(b),
        block=(2,1,1), grid=(1,1))

torch.cuda.synchronize()

print(dest)