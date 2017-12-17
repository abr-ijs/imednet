# -*- coding: utf-8 -*-
"""
Network

Created on Dec 14 2017

@author: Marcel Salmic

VERSION 1.0
"""
import torch
from torch.autograd import Variable

class Network(torch.nn.Module):
    def __init__(self, layerSizes = [784,200,50]):
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
        self.inputLayer = torch.nn.Linear(layerSizes[0],layerSizes[1])
        self.outputLayer = torch.nn.Linear(layerSizes[-2],layerSizes[-1])
        self.middleLayers = []
        for i in range(1, len(layerSizes) - 2):
            self.middleLayers.append(torch.nn.Linear(layerSizes[i],layerSizes[i+1]))

    def forward(self, x):
        """
        Defines the layers connections

        forward(x) -> result of forward propagation through network
        x -> input to the Network
        """
        sigmoid = torch.nn.Sigmoid()
        tansig = lambda x: 2*sigmoid(2*x)-1
        x = tansig(self.inputLayer(x))
        for layer in self.middleLayers:
            x = tansig(layer(x))
        output = tansig(self.outputLayer(x))
        return output

    def learn(self,x, y, epochs = 100, learning_rate = 1e-4, log_interval = 10):
        """
        teaches the network using provided data

        x -> input for the Network
        y -> desired output of the network for given x
        epochs -> how many times to repeat learning_rate
        learning_rate -> how much the weight will be changed each epoch
        log_interval -> on each epoch divided by log_interval log will be printed
        """
        criterion = torch.nn.MSELoss(size_average=False) #For calculating loss (mean squared error)
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate) # for updating weights
        for t in range(epochs):
            y_pred = self(x) # output from the network
            self.loss = criterion(y_pred,y) #loss
            if t % log_interval == 0:
                print('Epoch: ', t, ' loss: ',self.loss.data[0])
            optimizer.zero_grad()# setting gradients to zero
            self.loss.backward()# calculating gradients for every layer
            optimizer.step()#updating weights
