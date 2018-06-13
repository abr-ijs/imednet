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
  const int idx_out = threadIdx.x*301;
  const int idx_in = threadIdx.x*54;
  
  const int qe = threadIdx.y;
  
  float x = 1.0, dx=0.0;
  float fx,sum_psi, psi,a,y=dmp_parameters[idx_in],z=0;
  
  
  for(int i=0;i<301;i++)
  {
    fx = sum_psi = 0.0;
    for(int j=0;j<25;j++)
    {
        psi = exp(-0.5*(x-1));
        fx = fx +(psi);
        sum_psi = sum_psi+ psi;
        
    }
    
    
    //a = alpha_z*(beta_z*(goal-y)-z)+fx
    
    a = 48*(12*(dmp_parameters[idx_in]-y)-z)+fx;
    z = z + 0.01*a/3.0;
    y = y +0.01*z/3.0;
    
    
    //dx = -alpha_x*x/tau  
      
    dx = -2.0*x/3.0;
    
    //x = x+dx*dt
    x = x+dx*0.01;
    
    //printf(" %d ",x);    
    traj[i+idx_out] = x;
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

a = np.ones((200,50)).astype(np.float32)
b = np.ones((200,50)).astype(np.float32)

a = torch.from_numpy(a).cuda()
b = torch.from_numpy(b).cuda()
dest = torch.Tensor(400,301).cuda()

multiply_them(
        Holder(dest),
        Holder(a),

        block=(400,1,1), grid=(1,1))

torch.cuda.synchronize()

print(dest)