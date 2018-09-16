#!/usr/bin/env python
"""
Evaluate image-t
o-motion network results with dynamic time warping.
"""
from __future__ import print_function

import os
import sys
import re
import importlib
import torch
import numpy as np
from dtw import dtw
from torch.autograd import Variable

import argparse

from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

from imednet.data.smnist_loader import MatLoader, Mapping
from imednet.trainers.encoder_decoder_trainer import Trainer
from imednet.encoder_decoder import load_model

# Parse arguments
description = 'Evaluate image-to-motion network results with dynamic time warping.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--model-path', type=str, default=None,
                    help='model path (directory)')
parser.add_argument('--data-path', type=str, default=None,
                    help='data path (.mat file)')
parser.add_argument('--test-all-data', action='store_true', default=False,
                    help='test all data in dataset (ignore splits)')
args = parser.parse_args()

# Exit if no model or data path arguments are provided
if not args.model_path or not args.data_path:
    parser.print_help()
    exit(1)

# # Read network description file
# with open(os.path.join(args.model_path, 'network_description.txt')) as f:
#     network_description_str = f.read()
# 
# # Get the model class from the network description and dynamically import it
# model_module_class_str = re.search('Model: (.+?)\n', network_description_str).group(1)
# model_module_str = os.path.splitext(model_module_class_str)[0]
# model_class_str = os.path.splitext(model_module_class_str)[1][1:]
# model_module = importlib.import_module(model_module_str)
# model_class = getattr(model_module, model_class_str)
# 
# # Get the pre-trained CNN model load path from the network description
# if model_class_str == 'CNNEncoderDecoderNet':
#     pre_trained_cnn_model_path = re.search('Pre-trained CNN model load path: (.+?)\n', network_description_str).group(1)
# 
# # Load layer sizes
# layer_sizes = np.load(os.path.join(args.model_path, 'layer_sizes.npy')).tolist()
# 
# # Load scaling
# scaling = Mapping()
# scaling.x_max = np.load(os.path.join(args.model_path, 'scale_x_max.npy'))
# scaling.x_min = np.load(os.path.join(args.model_path, 'scale_x_min.npy'))
# scaling.y_max = np.load(os.path.join(args.model_path, 'scale_y_max.npy'))
# scaling.y_min = np.load(os.path.join(args.model_path, 'scale_y_min.npy'))
# 
# # Load the model
# if model_class_str == 'CNNEncoderDecoderNet':
#     model = model_class(pre_trained_cnn_model_path, layer_sizes, scaling)
# else:
#     model = model_class(layer_sizes, None, scaling)
# 
# # Load the model state parameters
# state = torch.load(os.path.join(args.model_path, 'net_parameters'))
# model.load_state_dict(state)

# Load model
model = load_model(args.model_path)

# Load data and scale it
print('Loading dataset...')
images, outputs, scale, original_trj = MatLoader.load_data(args.data_path,
                                                           load_original_trajectories=True
                                                           #,scale = model.scale
                                                           )
trainer = Trainer()

if not args.test_all_data:
    trainer.indeks = np.load(os.path.join(args.model_path, 'net_indeks.npy'))

original_trj_e = []
for i in range(0, images.shape[0]):
    c, c1, c2 = zip(*original_trj[i])
    original_trj_e.append(c)
    original_trj_e.append(c1)

if args.test_all_data:
    input_data_test_b = Variable(torch.from_numpy(images)).float()
    output_data_test_b = Variable(torch.from_numpy(np.array(original_trj_e)), requires_grad=False).float()
else:
    input_data_train_b, output_data_train_b, \
        input_data_test_b, output_data_test_b, \
        input_data_validate_b, output_data_validate_b \
        = trainer.split_dataset(images, original_trj_e)

# Reshape original trajectory matrix
print('Generating model output predictions from input data...')
if model_class_str == 'DMPEncoderDecoderNet':
    test_input = torch.from_numpy(np.array(input_data_test_b)).float().cuda()
else:
    model_output = model(input_data_test_b)
    test_output = output_data_test_b


def custom_norm(x, y):
    dif = x-y
    return (dif[0]**2 + dif[1]**2)**0.5


print('Generating DMPs from model output predictions, comparing to actual outputs, and calculating DTW errors...')
dtw_error = np.array([])
for i in range(0, model_output.shape[0]):
    print('Sample: {}'.format(i))
    DMP_parameters = torch.cat((torch.tensor([-1]).float(), model_output[i, :]), 0)
    dmp_obj = trainer.create_dmp(DMP_parameters, model.scale, 0.01, 25, True)
    dmp_obj.joint()

    b, b1, b2, b3 = dtw(np.transpose(test_output[i*2:i*2+2].numpy()), dmp_obj.Y, dist=custom_norm )
    #b, b1, b2, b3 = dtw(np.transpose(original_trj_e[i*2:i*2+2]), np.transpose(model_output[i*2:i*2+2].detach().cpu().numpy()), dist=custom_norm )
    dtw_error = np.append(dtw_error, b)

# Set up error and result save file paths
dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
dtw_errors_save_path = os.path.join(args.model_path, 'dtw_errors_' + dataset_name)
dtw_results_save_path = os.path.join(args.model_path, 'dtw_results_' + dataset_name + '.txt')
dtw_results_file = open(dtw_results_save_path, 'w')

print('Generating results...')
dtw_error_mean = np.mean(dtw_error)
dtw_error_std = np.std(dtw_error)
dtw_error_min = np.min(dtw_error)
dtw_error_max = np.max(dtw_error)
print('Model path: {}'.format(args.model_path))
print('Data path: {}'.format(args.data_path))
print('DTW error mean: {}'.format(dtw_error_mean))
print('DTW error STD: {}'.format(dtw_error_std))
print('DTW error min: {}'.format(dtw_error_min))
print('DTW error max: {}'.format(dtw_error_max))

print('Saving DTW results to: {}'.format(dtw_results_save_path))
dtw_results_file.write('Model path: {}\n'.format(args.model_path))
dtw_results_file.write('Data path: {}\n'.format(args.data_path))
dtw_results_file.write('DTW error mean: {}\n'.format(dtw_error_mean))
dtw_results_file.write('DTW error STD: {}\n'.format(dtw_error_std))
dtw_results_file.write('DTW error min: {}\n'.format(dtw_error_min))
dtw_results_file.write('DTW error max: {}\n'.format(dtw_error_max))
dtw_results_file.close()

print('Saving DTW errors to: {}'.format(dtw_errors_save_path))
np.save(dtw_errors_save_path, dtw_error)
