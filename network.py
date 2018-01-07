# -*- coding: utf-8 -*-
"""
Network

Created on Dec 14 2017

@author: Marcel Salmic

VERSION 1.0
"""
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

class Network(torch.nn.Module):
    def __init__(self, layerSizes = [784,200,50], conv = None):
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
            self.firstLayer = torch.nn.Conv2d(1,conv[0],conv[1])
            self.inputLayer = torch.nn.Linear(self.convSize, layerSizes[1])

        else:
            self.inputLayer = torch.nn.Linear(layerSizes[0],layerSizes[1])
        self.middleLayers = []
        for i in range(1, len(layerSizes) - 2):
            layer = torch.nn.Linear(layerSizes[i],layerSizes[i+1])
            self.middleLayers.append(layer)
            self.add_module("middleLayer_" + str(i), layer)
        self.outputLayer = torch.nn.Linear(layerSizes[-2],layerSizes[-1])
        self.scale = 1
        self.loss = 0

    def forward(self, x):
        """
        Defines the layers connections

        forward(x) -> result of forward propagation through network
        x -> input to the Network
        """
        tanh = torch.nn.Tanh()
        if self.conv:
            x = x.view(-1,1,self.imageSize, self.imageSize)
            x = self.firstLayer(x)
            x = x.view(-1,self.convSize)

        x = tanh(self.inputLayer(x))
        for layer in self.middleLayers:
            x = tanh(layer(x))
        output = self.outputLayer(x)
        return output

    def learn(self,x, y, bunch = 10, epochs = 100, learning_rate = 1e-4,momentum=0, log_interval = 10, livePlot = False):
        """
        teaches the network using provided data

        x -> input for the Network
        y -> desired output of the network for given x
        epochs -> how many times to repeat learning_rate
        learning_rate -> how much the weight will be changed each epoch
        log_interval -> on each epoch divided by log_interval log will be printed
        """
        criterion = torch.nn.MSELoss(size_average=False) #For calculating loss (mean squared error)
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum) # for updating weights
        oldLoss = 0
        if livePlot:
            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.ion()
            plt.show()
        for t in range(epochs):
            i = 0
            j = bunch
            self.loss = Variable(torch.Tensor([0]))
            permutations = torch.randperm(len(x))
            if self.isCuda():
                self.loss = self.loss.cuda()
                permutations = permutations.cuda()
            x = x[permutations]
            y = y[permutations]
            while j <= len(x):
                self.learn_one_step(x[i:j],y[i:j],learning_rate,criterion,optimizer)
                i = j
                j += bunch
            if i < len(x):
                self.learn_one_step(x[i:],y[i:],learning_rate,criterion,optimizer)
            if t % log_interval == 0:
                self.loss = self.loss * bunch/len(x)
                print('Epoch: ', t, ' loss: ', self.loss.data[0])
                if livePlot:
                        plt.plot(t, self.loss.data[0],'ob')
                        plt.pause(0.5)
                if (self.loss - oldLoss).data[0] == 0:
                    print("Loss hasn't changed in last ", log_interval ," iterations .Quiting...")
                    return
                oldLoss = self.loss


    def learn_one_step(self,x,y,learning_rate,criterion,optimizer):
        y_pred = self(x) # output from the network
        loss = criterion(y_pred,y) #loss
        optimizer.zero_grad()# setting gradients to zero
        loss.backward()# calculating gradients for every layer
        optimizer.step()#updating weights
        self.loss = self.loss + loss

    def isCuda(self):
        return self.inputLayer.weight.is_cuda
