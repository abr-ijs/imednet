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


## folders containing trajectories and mnist data
trajectories_folder = 'data/trajectories'
mnist_folder = 'data/mnist'


#DMP data
N = 25
sampling_time = 0.1

#learning params
epochs = 1000
learning_rate=0.01
bunch = 10

#layers size
numOfInputs = 784
HiddenLayer = [700, 500, 300,100,20,35]
#HiddenLayer = [100]
out = 2*N + 7
layerSizes = [numOfInputs] + HiddenLayer + [out]

#get mnist data
images, labels = Trainer.loadMnistData(mnist_folder)
#get trajectories
trajectories = Trainer.loadTrajectories(trajectories_folder)
# get DMPs
DMPs = Trainer.createDMPs(trajectories, N, sampling_time)
# get data to learn
input_data, output_data = Trainer.getDataForNetwork(images, DMPs,0,100)

#show data
show = [i for i in range(0)]
for i in show:
    Trainer.show_dmp(images[i], trajectories[i], DMPs[i])

#learn
print('starting learning')
model = Network(layerSizes)
#inicalizacija
for p in list(model.parameters()):
    torch.nn.init.normal(p,0,1000)
model.learn(input_data,output_data, bunch, epochs, learning_rate)
print('learning finished')

parameters = list(model.parameters())
