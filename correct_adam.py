import math
import torch
from torch.optim import Optimizer

class Adam(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    reset = False
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0 or self.reset:
                    self.reset = False
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss



class SCG(Optimizer):


    sigma0 = 5.e-5
    lamb = 5.e-7
    lamb_ = 0




    def __init__(self, params, lr = 0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):







        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SCG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SCG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)



    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        #loss = None
        #if closure is not None:
        #    loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                w_k = p.data
                p_k_new = - p.grad.data
                r_k_new = p_k_new.clone
                success = True
                k = 0
                lamda_k = self.lamb
                I_lamda_k = self.lamb_

                while k<3:
                    p_k = p_k_new.clone
                    r_k = r_k_new.clone

                    if success:
                        success = False

                        #Calculate second order information

                        p_k_norm = torch.norm( p_k, p = 1)

                        sigma_k = self.sigma / p_k_norm

                        p.data = w_k
                        loss_wk = closure()
                        grad_wk = p.grad.data
                        p.data.add_(sigma_k,p_k)
                        closure()
                        grad_wk_sigma = p.grad.data


                        s_k = (grad_wk_sigma - grad_wk) / sigma_k
                        tau_k = p_k * s_k

                    #scale
                    tau_k = tau_k +(lamda_k-I_lamda_k)*(p_k_norm**2)




                    #Hessian matrix positive definite

                    if tau_k <= 0:
                        I_lamda_k = 2*(lamda_k-tau_k/(p_k_norm**2))

                        tau_k = -tau_k + lamda_k*(p_k_norm**2)

                        lamda_k = I_lamda_k

                    #Calculate step size
                    phi_k = p_k*r_k
                    alpha_k = phi_k/tau_k



                    #Comparison parameter
                    p.data = w_k+alpha_k*p_k
                    loss_wk_alpha = closure()

                    delta_k = 2*tau_k*(loss_wk-loss_wk_alpha)/phi_k**2

                    if delta_k >= 0:
                        # Reduction in error
                        p.data.add_(alpha_k, p_k)
                        r_k_new = -p.grad.data
                        I_lamda_k = 0
                        success = True

                        #restart every lenght of parameter iterations
                        if k % 200== 0:
                            p_k_new = r_k_new

                        else:
                            beta_k = (torch.norm(r_k_new)**2-r_k_new*r_k)/phi_k
                            p_k_new = r_k_new + beta_k*p_k

                        if delta_k>=0.75:
                            lamda_k = lamda_k/4
                    else:
                        I_lamda_k = lamda_k
                        success=False








        return loss
