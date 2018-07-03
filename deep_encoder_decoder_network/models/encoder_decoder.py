# -*- coding: utf-8 -*-
"""
Network

Created on Dec 14 2017

@author: Marcel Salmic, Rok Pahic

VERSION 1.1
"""
import torch
import numpy as np

from deep_encoder_decoder_network.models.mnist_cnn import Net as MNISTNet
from deep_encoder_decoder_network.utils import DMP_layer


class CNNEncoderDecoderNet(torch.nn.Module):
    def __init__(self, pretrained_model_path, layer_sizes = [784,200,50], scale = None):
        """
        Creates a pretrained CNN + Encoder-Decoder network.

        layer_sizes -> list containing layer inputs/ouptuts (minimum length = 3)

            example:
                layer_sizes = [784,500,200,50]
                middleLayers -> [torch.nn.Linear(500,200)]
                outputLayer -> torch.nn.Linear(200,50)
        """
        super(CNNEncoderDecoderNet, self).__init__()

        # Load the MNIST CNN model
        self.cnn_model = MNISTNet()
        # Load the pretrained weights
        self.cnn_model.load_state_dict(torch.load(pretrained_model_path))
        # Chop off the FC layers (2 of them) + dropout layer,
        # leaving just the two conv layers.
        self.cnn_model = torch.nn.Sequential(*list(self.cnn_model.modules())[1:-3])
        # Get the output size of the last conv layer
        self.image_size = int(np.sqrt(layer_sizes[0]))
        self.conv1 = self.cnn_model[0].state_dict()['weight'].size()
        self.conv1_W = self.image_size - self.conv1[2] + 1
        self.conv1_size = (self.conv1_W)**2 * self.conv1[0]
        self.conv2 = self.cnn_model[1].state_dict()['weight'].size()
        self.conv2_W = self.conv1_W - self.conv2[2] + 1
        self.conv2_size = (self.conv2_W)**2 * self.conv2[0]

        # Set up the input layer for encoder-decoder part
        self.inputLayer = torch.nn.Linear(self.conv2_size, layer_sizes[1])

        self.middleLayers = []
        for i in range(1, len(layer_sizes) - 2):
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.middleLayers.append(layer)
            self.add_module("middleLayer_" + str(i), layer)
        self.outputLayer = torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.scale = scale
        self.loss = 0

    def forward(self, x):
        """
        Defines the layers connections

        forward(x) -> result of forward propagation through network
        x -> input to the Network
        """
        #activation_fn = torch.nn.ReLU6()
        activation_fn = torch.nn.Tanh()

        x = x.view(-1, 1, self.image_size, self.image_size)

        # x = self.firstLayer(x)
        # x = x.view(-1, self.convSize)

        # Run the input through the pretrained CNN
        x = self.cnn_model(x)
        x = x.view(-1, self.conv2_size)

        x = activation_fn(self.inputLayer(x))
        for layer in self.middleLayers:
            x = activation_fn(layer(x))
        output = self.outputLayer(x)
        return output

    def isCuda(self):
        return self.inputLayer.weight.is_cuda


class Network(torch.nn.Module):
    def __init__(self, layerSizes = [784,200,50], conv = None , scale = None):
        """
        Creates a custom Network

        layerSizes -> list containing layer inputs/ouptuts (minimum length = 3)
            example:
                layerSizes = [784,500,200,50]
                inputLayer -> torch.nn.Linear(784,500)
                middleLayers -> [torch.nn.Linear(500,200)]
                outputLayer -> torch.nn.Linear(200,50)
        """
        super(Network, self).__init__()
        self.conv = conv
        if self.conv:
            self.imageSize = int(np.sqrt(layerSizes[0]))
            self.convSize = (self.imageSize - conv[1] + 1)**2 * conv[0]
            self.firstLayer = torch.nn.Conv2d(1, conv[0], conv[1])
            self.inputLayer = torch.nn.Linear(self.convSize, layerSizes[1])

        else:
            self.inputLayer = torch.nn.Linear(layerSizes[0], layerSizes[1])
        self.middleLayers = []
        for i in range(1, len(layerSizes) - 2):
            layer = torch.nn.Linear(layerSizes[i], layerSizes[i+1])
            self.middleLayers.append(layer)
            self.add_module("middleLayer_" + str(i), layer)
        self.outputLayer = torch.nn.Linear(layerSizes[-2], layerSizes[-1])
        self.scale = scale
        self.loss = 0



    def forward(self, x):
        """
        Defines the layers connections

        forward(x) -> result of forward propagation through network
        x -> input to the Network
        """
        #activation_fn = torch.nn.ReLU6()
        activation_fn = torch.nn.Tanh()

        if self.conv:
            x = x.view(-1, 1, self.imageSize, self.imageSize)
            x = self.firstLayer(x)
            x = x.view(-1, self.convSize)

        x = activation_fn(self.inputLayer(x))
        for layer in self.middleLayers:
            x = activation_fn(layer(x))
        output = self.outputLayer(x)
        return output

    def isCuda(self):
        return self.inputLayer.weight.is_cuda


class training_parameters():
    #Before
    epochs = 1000
    bunch = 32
    val_fail = 5
    time = -1

    cuda = True

    validation_interval = 1
    log_interval = 1
    test_interval = 1

    training_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15
    data_samples = 0



    #After

    real_epochs = 0
    min_train_loss = -1
    min_val_loss = -1
    min_test_loss = -1
    elapsed_time = -1
    val_count = -1
    stop_criterion = ""
    min_grad = -1




    def __init__(self):
        pass

    def write_out(self):

        learn_info = "\n Setting parameters for learning:\n" + "   - Samples of data: " + str(self.data_samples) + \
                     "\n   - Epochs: " + str(self.epochs) + \
                     "\n   - Bunch size: " + str(self.bunch) \
                     + "\n   - training ratio: " + str(self.training_ratio) + "\n   - validation ratio: " + \
                     str(self.validation_ratio) + "\n   - test ratio: " + str(self.test_ratio)+\
                    "\n     -   validation_interval: " + str(self.validation_interval)+ \
                     "\n     -  test_interval: " + str(self.test_interval)+ \
                     "\n     -   log_interval: " + str(self.log_interval) +\
                    "\n     -   cuda = " + str(self.cuda)+ \
                     "\n     -  Validation fail: " + str(self.val_fail)


        return learn_info


    def write_out_after(self):

        learn_info = "\n Learning finished with this parameters:\n" + "   - Number of epochs: " + str(self.real_epochs) + \
                     "\n   - Last train loss: " + str(self.min_train_loss) + \
                     "\n   - Last validation loss: " + str(self.min_val_loss) + \
                     "\n   Last test loss: " + str(self.min_test_loss) + \
                     "\n   - Elapsed time: " + str(self.elapsed_time) + \
                     "\n   - last validation count: " + str(self.val_count) + \
                     "\n     -   Stop criterion: " + str(self.stop_criterion) + \
                     "\n     -  Minimal gradient: " + str(self.min_grad)


        return learn_info



class Network_DMP(torch.nn.Module):
    def __init__(self, layerSizes = [784,200,50], conv = None , scale = None):
        """
        Creates a custom Network

        layerSizes -> list containing layer inputs/ouptuts (minimum length = 3)
            example:
                layerSizes = [784,500,200,50]
                inputLayer -> torch.nn.Linear(784,500)
                middleLayers -> [torch.nn.Linear(500,200)]
                outputLayer -> torch.nn.Linear(200,50)
        """
        super(Network_DMP, self).__init__()
        self.conv = conv
        if self.conv:
            self.imageSize = int(np.sqrt(layerSizes[0]))
            self.convSize = (self.imageSize - conv[1] + 1)**2 * conv[0]
            self.firstLayer = torch.nn.Conv2d(1, conv[0], conv[1])
            self.inputLayer = torch.nn.Linear(self.convSize, layerSizes[1])

        else:
            self.inputLayer = torch.nn.Linear(layerSizes[0], layerSizes[1])
        self.middleLayers = []
        for i in range(1, len(layerSizes) - 2):
            layer = torch.nn.Linear(layerSizes[i], layerSizes[i+1])
            self.middleLayers.append(layer)
            self.add_module("middleLayer_" + str(i), layer)
        self.outputLayer = torch.nn.Linear(layerSizes[-2], layerSizes[-1])
        self.scale = scale
        self.loss = 0
        self.DMPparam = DMP_layer.createDMPparam(25, 3, 0.01, 2, scale)
        self.func = DMP_layer.DMP_integrator()
        '''self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('scale_t', self.DMPparam.scale_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)'''


    def forward(self, x):
        """
        Defines the layers connections

        forward(x) -> result of forward propagation through network
        x -> input to the Network
        """
        #activation_fn = torch.nn.ReLU6()
        activation_fn = torch.nn.Tanh()

        if self.conv:
            x = x.view(-1, 1, self.imageSize, self.imageSize)
            x = self.firstLayer(x)
            x = x.view(-1, self.convSize)

        x = activation_fn(self.inputLayer(x))
        for layer in self.middleLayers:
            x = activation_fn(layer(x))
        x = self.outputLayer(x)
        output = self.func.apply(x, self.DMPp,self.param_grad,self.scale_t)
        return output

    def isCuda(self):
        return self.inputLayer.weight.is_cuda
