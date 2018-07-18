#!/usr/bin/env python
"""
Evaluate neural network results with Dynamic time warping.

"""


import sys
import os
from os.path import dirname, realpath
from imednet.data.smnist_loader import MatLoader, Mapping
from imednet.trainers.encoder_decoder_trainer import Trainer
from imednet.models.encoder_decoder import CNNEncoderDecoderNet

import torch
import numpy as np
from dtw import dtw

sys.path.append(dirname(dirname(realpath(__file__))))

# region Import data and networks
default_data_path = os.path.join(dirname(dirname(realpath(__file__))), 'data/s-mnist/40x40-smnist-with-all-nmnist-noise.mat')

dmp_network = False

if dmp_network:

    default_network_load_path = os.path.join(dirname(dirname(dirname(realpath(__file__)))),
                                            'models/dmp_encoder_decoder',
                                            'Model 2018-07-12 11:44:21.234266')
else:
    default_network_load_path = os.path.join(dirname(dirname(realpath(__file__))),
                                             #'models/encoder_decoder',
                                             #'Model 2018-07-12 09:19:45.174311')
                                             'models/cnn_encoder_decoder',
                                             '40x40-smnist-with-all-nmnist-noise 2018-07-18 01:25:42.205735')


# Load the model
#model = torch.load(default_network_load_path + '/model.pt')

layer_sizes = np.load(os.path.join(default_network_load_path,'layer_sizes.npy')).tolist()

# Load scaling
scaling = Mapping()
scaling.x_max = np.load(os.path.join(default_network_load_path,'scale_x_max.npy'))
scaling.x_min = np.load(os.path.join(default_network_load_path,'scale_x_min.npy'))
scaling.y_max = np.load(os.path.join(default_network_load_path,'scale_y_max.npy'))
scaling.y_min = np.load(os.path.join(default_network_load_path,'scale_y_min.npy'))

default_cnn_model_load_path = os.path.join(dirname(dirname(realpath(__file__))),
                                           'models/mnist_cnn/mnist_cnn.model')
model = CNNEncoderDecoderNet(default_cnn_model_load_path, layer_sizes, scaling)

state = torch.load(default_network_load_path + '/net_parameters')

model.load_state_dict(state)

# Load data and scale it
images, outputs, scale, original_trj = MatLoader.load_data(default_data_path,
                                                           load_original_trajectories=True
                                                           #,scale = model.scale
                                                           )
trainer = Trainer()

trainer.indeks=np.load(os.path.join(default_network_load_path,'net_indeks.npy'))

original_trj_e = []
for i in range(0, images.shape[0]):
    c, c1, c2 = zip(*original_trj[i])
    original_trj_e.append(c)
    original_trj_e.append(c1)


input_data_train_b, output_data_train_b, \
input_data_test_b, output_data_test_b, \
input_data_validate_b, output_data_validate_b \
    = trainer.split_dataset(images,original_trj_e)

# endregion

##  Reshape original trajectory matrix


if dmp_network:
    test_input = torch.from_numpy(np.array(input_data_test_b)).float().cuda()
else:

    nn_output = model(input_data_test_b)
    test_output = output_data_test_b










def my_custom_norm(x, y):

    dif = x-y

    return (dif[0]**2 + dif[1]**2)**0.5


avr_error = 0
dtw_error = np.array([])
for i in range(0, nn_output.shape[0]):

    print(i)

    DMP_parameters = torch.cat((torch.tensor([-1]).float(), nn_output[i, :]), 0)
    dmp_obj = trainer.create_dmp(DMP_parameters, model.scale, 0.01, 25, True)
    dmp_obj.joint()

    b, b1, b2, b3 = dtw(np.transpose(test_output[i*2:i*2+2].numpy()), dmp_obj.Y, dist=my_custom_norm )

    #b, b1, b2, b3 = dtw(np.transpose(original_trj_e[i*2:i*2+2]), np.transpose(nn_output[i*2:i*2+2].detach().cpu().numpy()), dist=my_custom_norm )
    dtw_error = np.append(dtw_error, b)


print('Mean DTW error:')
print(np.mean(dtw_error))
max=np.max(dtw_error)
print('Maximal DTW error:')
print(max)
