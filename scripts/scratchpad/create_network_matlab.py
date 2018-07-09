import sys
import torch
import numpy as np

from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

from deep_encoder_decoder_network.models.encoder_decoder import EncoderDecoderNet
from deep_encoder_decoder_network.utils.dmp_layer import DMPIntegrator, DMPParameters
from deep_encoder_decoder_network.trainers.encoder_decoder_trainer import Trainer
from deep_encoder_decoder_network.data.mat_loader import MatLoader

print()
# folders containing trajectories and mnist data
trajectories_folder = 'data/trajectories'
mnist_folder = 'data/mnist'
scale_file = 'scale.npy'

parameters_file = 'net_parameters_matlab'

# DMP data
N = 25
sampling_time = 0.1

# Learning params
epochs = 500
learning_rate=0.0005
momentum = 0.5
batch_size = 32

load = False

cuda = True
plot = False

load_from_cuda = False

# Layer sizes
numOfInputs = 1600
HiddenLayer = [ 1500, 1300, 1000, 600,200,20,35]
conv = None
# HiddenLayer = [100]
out = 2*N + 7
# out = 2*N
layerSizes = [numOfInputs] + HiddenLayer + [out]

images, outputs, scale = MatLoader.load_data('slike_780.4251')
input_data, output_data = MatLoader.data_for_network(images, outputs)

# Learn
print()
print('Starting learning')
print(" + Learning with parameters: ")
print("   - Samples of data", len(input_data))
print("   - Epochs: ", epochs)
print("   - Learning rate: ", learning_rate)
print("   - Momentum: ", momentum)
print("   - Batch size: ", batch_size)

model = EncoderDecoderNet(layerSizes, conv)
model.scale = scale
np.save(scale_file, scale)

# Inicalizacija
if load:
    print(' + Loaded parameters from file: ', parameters_file)
    if load_from_cuda:
        model.load_state_dict(torch.load(parameters_file, map_location=lambda storage, loc: storage))  # loading parameters
    else:
        model.load_state_dict(torch.load(parameters_file))  # loading parameters
else:
    print(' + Initialized paramters randomly')
    for p in list(model.parameters()):
        torch.nn.init.normal(p,0,1e+2)

if cuda:
    model.cuda()
    input_data = input_data.cuda()
    output_data = output_data.cuda()

model.learn(input_data,output_data, batch_size, epochs, learning_rate,momentum, 1, plot)
print('Learning finished\n')

parameters = list(model.parameters())

torch.save(model.state_dict(), parameters_file)  # saving parameters

# Trainer.show_network_output(model, 1, images, trajectories,DMPs, N, sampling_time, indexes)
if plot:
    for i in range(0,5):
        Trainer.show_network_output(model, i, images, trajectories,DMPs, N, sampling_time)

    Trainer.show_network_output(model, -1, test[:5], None, None, N, sampling_time)
