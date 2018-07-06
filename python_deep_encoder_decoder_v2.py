# -*- coding: utf-8 -*-
"""
Network creation

Created on Dec 14 2017

@author: Marcel Salmic

VERSION 1.0
"""
import torch
import numpy as np

from network import Network_DMP, training_parameters
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
date = str(datetime.now())
directory_name = 'NN ' + date
parameters_file = directory_path + directory_name + '/net_parameters'
makedirs(directory_path+directory_name)

file = open(directory_path+directory_name+'/Network_description.txt','w')

file.write('Network created: ' + date)
dateset_name = '/home/rpahic/deep_encoder_decoder_network/deep_encoder_decoder_data_and_networks/slike_780.4251'


#Load data and scale it.....................................................

images, outputs, scale, or_tr= matLoader.loadData(dateset_name,load_original_trajectories=True)


#Create network and save model.................................................

#DMP data
N = 25
sampling_time = 0.1


#layers size
numOfInputs = 1600
HiddenLayer = [1500, 1300, 1000, 600, 200, 20, 35]
conv = None


out = 2*N + 4
#out = 2*N

layerSizes = [numOfInputs] + HiddenLayer + [out]


file.write('\nNeurons: ' + str(layerSizes))

model_new = Network_DMP(layerSizes, conv, scale)
model_new.register_buffer('DMPp', model_new.DMPparam.data_tensor)
model_new.register_buffer('scale_t', model_new.DMPparam.scale_tensor)
model_new.register_buffer('param_grad', model_new.DMPparam.grad_tensor)


#inicalizacija
if load:
    net_id = '2018-04-25 13:36:32.095726'

    load_parameters_file = directory_path + 'NN ' + net_id
    print(' + Loaded parameters from file: ', load_parameters_file)
    model_new.load_state_dict(torch.load(load_parameters_file+'/net_parameters'))  # loading parameters
    '''if load_from_cuda:
        model.load_state_dict(torch.load(parameters_file, map_location=lambda storage, loc: storage)) # loading parameters
    else:
        model.load_state_dict(torch.load(parameters_file)) # loading parameters'''

else:
    print(' + Initialized parameters randomly')



    for p in list(model_new.parameters()):
        if p.data.ndimension() ==1:
            torch.nn.init.constant(p, 0)
        else:


            torch.nn.init.xavier_uniform(p, gain=1)











torch.save(model_new, (directory_path+directory_name+'/model.pt'))

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
train_param.val_fail = 60


trener = Trainer()
if load==True:
    #pass
    trener.indeks = np.load(directory_path + 'NN ' + net_id +'/net_indeks.npy')

original_trj_e = []
for i in range(0,images.shape[0]):
    c,c1,c2 = zip(*or_tr[i])
    original_trj_e.append(c)
    original_trj_e.append(c1)

best_nn_parameters = trener.learn_DMP(model_new, images[0:1000], original_trj_e[0:2000], directory_path + directory_name, train_param, file, learning_rate, momentum)




#parameters = list(model.parameters())

#torch.save(model.state_dict(), parameters_file) # saving parameters

np.save(directory_path+directory_name+'/net_indeks', trener.indeks)
torch.save(best_nn_parameters, parameters_file)
file.close()




'''






#Trainer.showNetworkOutput(model, 1, images, trajectories,DMPs, N, sampling_time, indexes)
if plot:
    for i in range(0,5):
        Trainer.showNetworkOutput(model, i, images, trajectories,DMPs, N, sampling_time)

    Trainer.showNetworkOutput(model, -1, test[:5], None, None, N, sampling_time)
    
    '''