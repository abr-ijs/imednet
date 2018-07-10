import mlpy
import torch
import numpy as np
from torch.autograd import Variable
from dtw import dtw

from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

from imednet.trainers.encoder_decoder_trainer import Trainer
from imednet.data.mat_loader import MatLoader

# region Import network

# Load model and parameters
directory_path = '/home/rpahic/Documents/Neural_networks/'
directory_name = 'NN ' + '2018-06-14 13:23:28.813445'

model = torch.load(directory_path + directory_name + '/model.pt')

state = torch.load(directory_path + directory_name + '/net_parameters')

model.load_state_dict(state)

# endregion


# region Import database
import os
cwd = os.getcwd()
data_path=os.path.dirname(cwd)
dataset_name = data_path + '/' + 'slike_780.4251'
images, outputs, scale, original_trj = MatLoader.load_data(dataset_name, load_original_trajectories=True)

original_trj_e = []
for i in range(0,images.shape[0]):
    c,c1,c2 = zip(*original_trj[i])
    original_trj_e.append(c)
    original_trj_e.append(c1)

test_input = Variable(torch.from_numpy(np.array(images))).float().cuda()
test_output = Variable(torch.from_numpy(np.array(original_trj_e))).float().cuda()

# endregion

model=model.cuda()
nn_output = model(test_input)

def my_custom_normmy_cust (x, y):
    #return abs(x-y)
    dif= x-y
    return (dif[0]**2 + dif[1]**2)**(0.5)

# a=mlpy.dtw_std(np.array(original_trj_e[0:2]),nn_output[0:2].detach().cpu().numpy())
sum=0
for i in range(0,200):
    print(i)
    b,b1,b2,b3=dtw(np.transpose(original_trj_e[i*2:i*2+2]),np.transpose(nn_output[i*2:i*2+2].detach().cpu().numpy()),dist=my_custom_normmy_cust )
    sum=sum+b

sum
print(sum/200)
