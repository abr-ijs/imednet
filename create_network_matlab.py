# -*- coding: utf-8 -*-
"""
Network creation

Created on Dec 14 2017

@author: Marcel Salmic

VERSION 1.0
"""
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from network import Network
from trajectory_loader import trajectory_loader as loader
from trainer import Trainer
from matLoader import matLoader

print()
## folders containing trajectories and mnist data
trajectories_folder = 'data/trajectories'
mnist_folder = 'data/mnist'
scale_file = 'scale.npy'

parameters_file = 'net_parameters'

#DMP data
N = 25
sampling_time = 0.1

#learning params
epochs = 0
learning_rate=0.001
momentum = 0
bunch = 32

load = True

cuda = False
plot = True

load_from_cuda = True


#layers size
numOfInputs = 1600
HiddenLayer = [ 600, 350, 150, 40]
conv = [10,5]
#HiddenLayer = [100]
out = 2*N + 7
#out = 2*N
layerSizes = [numOfInputs] + HiddenLayer + [out]


images, outputs, scale = matLoader.loadData('/storage/share/MNIST Drawing Project/Stevila/slike_780.4251')
input_data, output_data = matLoader(images, outputs)


#learn
print()
print('Starting learning')
print(" + Learning with parameters: ")
print("   - Samples of data", len(input_data))
print("   - Epochs: ", epochs)
print("   - Learning rate: ", learning_rate)
print("   - Momentum: ", momentum)
print("   - Bunch size: ", bunch)


model = Network(layerSizes, conv)
model.scale = scale
np.save(scale_file, scale)
#inicalizacija
if load:
    print(' + Loaded parameters from file: ', parameters_file)
    if load_from_cuda:
        model.load_state_dict(torch.load(parameters_file, map_location=lambda storage, loc: storage)) # loading parameters
    else:
        model.load_state_dict(torch.load(parameters_file)) # loading parameters

else:
    print(' + Initialized paramters randomly')
    for p in list(model.parameters()):
        torch.nn.init.normal(p,0,1e+2)

if cuda:
    model.cuda()
    input_data = input_data.cuda()
    output_data = output_data.cuda()

model.learn(input_data,output_data, bunch, epochs, learning_rate,momentum, 10, plot)
print('Learning finished\n')

parameters = list(model.parameters())

torch.save(model.state_dict(), parameters_file) # saving parameters

#Trainer.showNetworkOutput(model, 1, images, trajectories,DMPs, N, sampling_time, indexes)
if plot:
    for i in range(0,5):
        Trainer.showNetworkOutput(model, i, images, trajectories,DMPs, N, sampling_time)

    Trainer.showNetworkOutput(model, -1, test[:5], None, None, N, sampling_time)
