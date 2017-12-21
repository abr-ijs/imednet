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

print()
## folders containing trajectories and mnist data
trajectories_folder = 'data/trajectories'
mnist_folder = 'data/mnist'

parameters_file = 'net_parameters'

#DMP data
N = 25
sampling_time = 0.1

#learning params
epochs = 200
learning_rate=0.01
momentum = 0
bunch = 10
load = True
#load = False


#layers size
numOfInputs = 784
HiddenLayer = [ 600, 400, 200,  80, 50, 20, 35, 50]
#HiddenLayer = [100]
out = 2*N + 7
#out = 2*N
layerSizes = [numOfInputs] + HiddenLayer + [out]

print('Loading Mnist images')
#get mnist data
images, labels = Trainer.loadMnistData(mnist_folder)
print(' Done loading Mnist images')
#get trajectories
avaliable = loader.getAvaliableTrajectoriesNumbers(trajectories_folder)

avaliable = avaliable[:50]
print('Loading ',  len(avaliable), ' trajectories')
trajectories = Trainer.loadTrajectories(trajectories_folder, avaliable)
print(' Done loading trajectories')
# get DMPs
print('Creating DMPs')
DMPs = Trainer.createDMPs(trajectories, N, sampling_time)
print(' Done creating DMPs')
# get data to learn

input_data, output_data, scale = Trainer.getDataForNetwork(images, DMPs, avaliable)

#show data
# show = [i for i in range(lower,upper)]
# for i in show:
#     Trainer.show_dmp(images[i], trajectories[i], DMPs[i])

#learn
print()
print('Starting learning')
print(" + Learning with parameters: ")
print("   - Samples of data", len(input_data))
print("   - Epochs: ", epochs)
print("   - Learning rate: ", learning_rate)
print("   - Momentum: ", momentum)
print("   - Bunch size: ", bunch)


model = Network(layerSizes)
model.scale = scale
#inicalizacija
if load:
    print(' + Loaded parameters from file: ', parameters_file)
    model.load_state_dict(torch.load(parameters_file)) # loading parameters
else:
    print(' + Initialized paramters randomly')
    for p in list(model.parameters()):
        torch.nn.init.normal(p,0,1e+6)

model.learn(input_data,output_data, bunch, epochs, learning_rate,momentum)
print('Learning finished\n')

parameters = list(model.parameters())

torch.save(model.state_dict(), parameters_file) # saving parameters

Trainer.showNetworkOutput(model, 1, images, trajectories, avaliable,DMPs, N, sampling_time)
