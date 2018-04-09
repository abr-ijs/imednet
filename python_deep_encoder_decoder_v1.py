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

from network import Network, training_parameters
from trajectory_loader import trajectory_loader as loader
from trainer import Trainer
from matLoader import matLoader


from datetime import datetime

from os import makedirs

dateset_name = 'slike_780.4251'

load = False

cuda = True
plot = False

load_from_cuda = False


#Prepare directory and description file...................................
directory_path = '/home/rpahic/Documents/Neural_networks/'
directory_name = 'NN ' + str(datetime.now())
parameters_file = directory_path + directory_name + '/net_parameters'
makedirs(directory_path+directory_name)

file = open(directory_path+directory_name+'/Network_description.txt','w')

file.write('Network created: ' + str(datetime.now()))


#Load data and scale it.....................................................

images, outputs, scale = matLoader.loadData(dateset_name)


#Create network and save model.................................................

#DMP data
N = 25
sampling_time = 0.1


#layers size
numOfInputs = 1600
HiddenLayer = [1500, 1300, 1000, 600, 200, 20, 35]
conv = None

#HiddenLayer = [100]
#out = 2*N + 4
out = 2*N

layerSizes = [numOfInputs] + HiddenLayer + [out]


file.write('\nNeurons: '+ str(layerSizes))


model = Network(layerSizes, conv, scale)


#inicalizacija
if load:
    print(' + Loaded parameters from file: ', parameters_file)
    if load_from_cuda:
        model.load_state_dict(torch.load(parameters_file, map_location=lambda storage, loc: storage)) # loading parameters
    else:
        model.load_state_dict(torch.load(parameters_file)) # loading parameters

else:
    print(' + Initialized parameters randomly')
    for p in list(model.parameters()):
        torch.nn.init.normal(p,0,1e+2)






torch.save(model, (directory_path+directory_name+'/model.pt'))

#Set learning.....................................................................

#learning params

train_param = training_parameters()
train_param.epochs = -1
learning_rate = 0.0005
momentum = 0.5
train_param.bunch = 128
train_param.training_ratio = 0.7
train_param.validation_ratio = 0.15
train_param.test_ratio = 0.15
train_param.val_fail = 30


trener = Trainer()
trener.learn(model, images, outputs, directory_path + directory_name, train_param, file, learning_rate, momentum)




parameters = list(model.parameters())

torch.save(model.state_dict(), parameters_file) # saving parameters

file.close()




'''






#Trainer.showNetworkOutput(model, 1, images, trajectories,DMPs, N, sampling_time, indexes)
if plot:
    for i in range(0,5):
        Trainer.showNetworkOutput(model, i, images, trajectories,DMPs, N, sampling_time)

    Trainer.showNetworkOutput(model, -1, test[:5], None, None, N, sampling_time)
    
    '''