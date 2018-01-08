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
scale_file = 'scale.npy'

parameters_file = 'net_parameters'

#DMP data
N = 25
sampling_time = 0.1

#learning params
epochs = 2000
learning_rate=0.01
momentum = 0.2
decay = [1e-9,1e-6]
bunch = 16
oneDigidOnly = False
data = 100
s_data = 0
artificial_samples = 0
digit = 0

load = False


cuda = True
plot = True

load_from_cuda = False


#layers size
numOfInputs = 784
HiddenLayer = [ 600, 350, 150, 40]
conv = [10,5]
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

if oneDigidOnly:
    indexes = np.where(labels==digit)
    indexes = np.intersect1d(indexes,avaliable)
else:
    avaliable = avaliable[s_data:data]
    indexes = avaliable

print('Loading ',  len(indexes), ' trajectories')
trajectories = Trainer.loadTrajectories(trajectories_folder, indexes)
print(' Done loading trajectories')
test = images[-100:]
print('Multiplying data')
images = images[indexes]
trajectories, images = Trainer.randomlyRotateData(trajectories, images, artificial_samples)
print('Done multiplying data. Now having ',  len(trajectories), ' data')

# get DMPs
print('Creating DMPs')
DMPs = Trainer.createDMPs(trajectories, N, sampling_time)
print(' Done creating DMPs')
# get data to learn


#Code to find wrong data
#wrong = []
#for i in range(500,6100):
#    DMPs[i].joint()
#    if DMPs[i].Y.max() > 28 or DMPs[i].Y.min() < 0:
#        wrong.append(indexes[i])
#        print(indexes[i])
#
#print('rm ' + " ".join(['image_' +str(i) + '.json' for i in wrong]))

#scale = np.load(scale_file)
input_data, output_data, scale = Trainer.getDataForNetwork(images, DMPs)


# for i in range(int(len(indexes))):
#     Trainer.show_dmp(images[i],trajectories[i],DMPs[i],indexes[i])


#learn
print()
print('Starting learning')
print(" + Learning with parameters: ")
print("   - Samples of data", len(input_data))
print("   - Epochs: ", epochs)
print("   - Learning rate: ", learning_rate)
print("   - Momentum: ", momentum)
print("   - Decay: ", decay)
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
        torch.nn.init.normal(p,0,0.1)

if cuda:
    model.cuda()
    input_data = input_data.cuda()
    output_data = output_data.cuda()

model.learn(input_data,output_data, bunch, epochs, learning_rate,momentum, 10, plot, decay)
print('Learning finished\n')

parameters = list(model.parameters())

torch.save(model.state_dict(), parameters_file) # saving parameters

#Trainer.showNetworkOutput(model, 1, images, trajectories,DMPs, N, sampling_time, indexes)
if plot:
    for i in np.random.rand(10)*(data-s_data):
        Trainer.showNetworkOutput(model, int(i), images, trajectories,DMPs, N, sampling_time, cuda = cuda)

    Trainer.showNetworkOutput(model, -1, test[:5], None, None, N, sampling_time, cuda = cuda)
