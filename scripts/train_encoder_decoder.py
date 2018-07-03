#!/usr/bin/env python
"""Train an encoder-decoder network on image/trajectory data.

Loads the synthetic dataset of MNIST-esque digit images and accompanying
trajectories and trains a deep encoder-decoder network to predict DMP parameter
outputs from input images.
"""
from __future__ import print_function

import os
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

import argparse
from datetime import datetime

import torch
import numpy as np

from deep_encoder_decoder_network.models.encoder_decoder import Network, training_parameters
from deep_encoder_decoder_network.trainers.encoder_decoder_trainer import Trainer
from deep_encoder_decoder_network.data.matLoader import matLoader

# Save datetime
date = datetime.now()

# Set defaults
default_data_path = os.path.join(dirname(dirname(realpath(__file__))), 'data/slike_780.4251')
default_model_save_path = os.path.join(dirname(dirname(realpath(__file__))),
                                       'models/encoder_decoder',
                                       'Model ' + str(date))
default_model_load_path = None

# Parse arguments
description = 'Train an encoder-decoder network on image/trajectory data.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--data-path', type=str, default=default_data_path,
                    help='data path (default: "{}")'.format(str(default_data_path)))
parser.add_argument('--model-save-path', type=str, default=default_model_save_path,
                    help='model save path (default: "{}")'.format(str(default_model_save_path)))
parser.add_argument('--model-load-path', type=str, default=None,
                    help='model load path (default: "{}")'.format(str(default_model_load_path)))
args = parser.parse_args()

# Set up model save files
os.makedirs(args.model_save_path)
net_description_save_path = os.path.join(args.model_save_path, 'network_description.txt')
net_description_file = open(net_description_save_path, 'w')
net_description_file.write('Network created: ' + str(date))

# Load data and scale it
images, outputs, scale, or_tr = matLoader.loadData(args.data_path)

# Set up DMP parameters
N = 25
sampling_time = 0.1

# Define layer sizes
input_size = 1600
hidden_layer_sizes = [1500, 1300, 1000, 600, 200, 20, 35]
output_size = 2*N + 4
conv = None
layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
net_description_file.write('\nNeurons: ' + str(layer_sizes))

# Load the model
model = Network(layer_sizes, conv, scale)

# Initialize the model
if args.model_load_path:
    net_params_path = os.path.join(args.model_load_path, 'net_parameters')
    model.load_state_dict(torch.load(net_params_path))
    print(' + Loaded parameters from file: ', args.model_load_path)
else:
    for p in list(model.parameters()):
        if p.data.ndimension() == 1:
            torch.nn.init.constant(p, 0)
        else:
            torch.nn.init.xavier_uniform(p, gain=1)
    print(' + Initialized parameters randomly')

torch.save(model, (os.path.join(args.model_save_path, 'model.pt')))

# Set up trainer
train_param = training_parameters()
train_param.epochs = -1
learning_rate = 0.0005
momentum = 0.5
train_param.bunch = 128
train_param.training_ratio = 0.7
train_param.validation_ratio = 0.15
train_param.test_ratio = 0.15
train_param.val_fail = 60
trainer = Trainer()

if args.model_load_path:
    net_indeks_path = os.path.join(args.model_load_path, 'net_indeks.npy')
    trainer.indeks = np.load(net_indeks_path)

# Train
best_nn_parameters = trainer.learn(model, images, outputs,
                                   args.model_save_path,
                                   train_param,
                                   net_description_file,
                                   learning_rate, momentum)

# Save model
np.save(os.path.join(args.model_save_path, 'net_indeks'), trainer.indeks)
net_params_path = os.path.join(args.model_save_path, 'net_parameters')
torch.save(best_nn_parameters, net_params_path)
net_description_file.close()
