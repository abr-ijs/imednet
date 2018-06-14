import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import torch

x = torch.cuda.FloatTensor(8)

from pycuda.compiler import SourceModule
mod = SourceModule("""
#include <math.h>


__global__ void multiply_them(float *traj, float *dmp_parameters, float *c, float *sigma2)
{

  const int idx_in = ((blockIdx.x * blockDim.x)+threadIdx.x)*54;
  const int idx_out =((blockIdx.x * blockDim.x)+threadIdx.x)*301*2;
  

  
  const int qe = threadIdx.y;
  
  float x , dx;
  float fx,sum_psi, psi,a,y,z=0;
  
  for(int dof=0;dof<2;dof++)
  {
  
    x = 1.0;
    y=dmp_parameters[idx_in+dof];
    
    for(int i=0;i<301;i++)
      {
        fx = sum_psi = 0.0;
        for(int j=0;j<25;j++)
        {
            psi =exp(-0.5*((x-c[j])*(x-c[j])/sigma2[j]));
            fx = fx +(dmp_parameters[idx_in+4+j*2+dof]*psi);
            sum_psi = sum_psi + psi;
            
        }
        
        
        //a = alpha_z*(beta_z*(goal-y)-z)+fx
        
        a = 48*(12*(dmp_parameters[idx_in+2+dof]-y)-z)+fx;
        z = z + 0.01*a/3.0;
        y = y +0.01*z/3.0;
        
        
        
        
        //dx = -alpha_x*x/tau  
          
        dx = -2.0*x/3.0;
        
        //x = x+dx*dt
        x = x+dx*0.01;
        
        //printf(" %d ",x);    
        traj[i+idx_out+dof*301] = y;
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

a = np.ones((20000,54)).astype(np.float32)
c = np.ones((1,25)).astype(np.float32)
sigma_2 = np.ones((1,25)).astype(np.float32)

a = torch.from_numpy(a).cuda()
c = torch.from_numpy(c).cuda()
sigma_2 = torch.from_numpy(sigma_2).cuda()

dest = torch.Tensor(40000,301).cuda()

multiply_them(
        Holder(dest),
        Holder(a),
        Holder(c),
        Holder(sigma_2),
        block=(1000,1,1), grid=(20,1))

torch.cuda.synchronize()

print(dest)
print(dest.shape)