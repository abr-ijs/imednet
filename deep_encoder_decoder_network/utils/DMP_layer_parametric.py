import torch
from torch.autograd import Function
from torch.autograd import Variable

class DMP_integrator(Function):





    @staticmethod
    def forward(ctx, inputs):
        a_z = 48
        a_x = 2
        #N = dmp_composition.N
        #dt = dmp_composition.dt


        result = inputs *3

        ctx.save_for_backward(inputs)

        return inputs.new(result)





    @staticmethod
    def backward(ctx, grad_outputs):
        grad=grad_outputs.data*2

        return Variable(grad.new(grad))