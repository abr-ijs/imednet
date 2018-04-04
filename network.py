# -*- coding: utf-8 -*-
"""
Network

Created on Dec 14 2017

@author: Marcel Salmic, Rok Pahic

VERSION 1.1
"""
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
import subprocess
import webbrowser
import sys
import time
from trainer import Trainer
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
        tanh = torch.nn.Tanh()
        if self.conv:
            x = x.view(-1, 1, self.imageSize, self.imageSize)
            x = self.firstLayer(x)
            x = x.view(-1, self.convSize)

        x = tanh(self.inputLayer(x))
        for layer in self.middleLayers:
            x = tanh(layer(x))
        output = self.outputLayer(x)
        return output

    def learn(self, images, outputs, path, bunch = 10, epochs = 100, learning_rate = 1e-4,momentum=0, log_interval = 10, livePlot = False, decay = [0,0]):
        """
        teaches the network using provided data

        x -> input for the Network
        y -> desired output of the network for given x
        epochs -> how many times to repeat learning_rate
        learning_rate -> how much the weight will be changed each epoch
        log_interval -> on each epoch divided by log_interval log will be printed
        """


        val_count = 0
        writer = SummaryWriter(path+'/log')
        command = ["tensorboard", "--logdir=" + path+"/log"]
        proces = subprocess.Popen(command)
        print(proces.pid)
        print('init')
        trainer = Trainer()
        # Divide data
        print("Dividing data")
        input_data_train, output_data_train, input_data_test, output_data_test, input_data_validate, output_data_validate = trainer.databaseSplit(images, outputs)
        cuda=1



        if cuda == 1:
            input_data_train = input_data_train.cuda()
            output_data_train = output_data_train.cuda()
            input_data_test = input_data_test.cuda()
            output_data_test = output_data_test.cuda()
            input_data_validate = input_data_validate.cuda()
            output_data_validate = output_data_validate.cuda()

        validation_interval = 10
        log_interval = 5
        test_interval = 20

        print('finish dividing')

        criterion = torch.nn.MSELoss(size_average=False) #For calculating loss (mean squared error)
        #optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, mometum=momentum) # for updating weights
        optimizer = torch.optim.Adagrad(self.parameters(), lr=learning_rate, lr_decay = decay[0], weight_decay = decay[1]) #, momentum=momentum) # for updating weights
        oldLoss = 0

        y_val = self(input_data_validate)
        oldValLoss = criterion(y_val, output_data_validate)

        if livePlot:
            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            ax = plt.axes();
            ax.set_yscale('log')
            plt.ion()
            plt.show()

        #Infiniti epochs
        if epochs == -1:
            inf_k = 0
        else:
            inf_k = 1

        train=True
        t = 0
        while train == True:
            t = t+1
            i = 0
            j = bunch
            self.loss = Variable(torch.Tensor([0]))
            permutations = torch.randperm(len(input_data_train))
            if self.isCuda():
                self.loss = self.loss.cuda()
                permutations = permutations.cuda()
            input_data_train = input_data_train[permutations]
            output_data_train = output_data_train[permutations]

            while j <= len(input_data_train):
                self.learn_one_step(input_data_train[i:j], output_data_train[i:j], learning_rate,criterion,optimizer)
                i = j
                j += bunch
            if i < len(input_data_train):
                self.learn_one_step(input_data_train[i:], output_data_train[i:], learning_rate,criterion,optimizer)


            if (t-1)%validation_interval == 0:
                y_val = self(input_data_validate)
                val_loss = criterion(y_val, output_data_validate)
                writer.add_scalar('data/val_loss', val_loss, t)

                if val_loss.data[0] > oldValLoss:
                    val_count = val_count+1

                else:

                    val_count = 0
                    oldValLoss = val_loss.data[0]


                writer.add_scalar('data/val_count', val_count, t)

                #mat_t = input_data_validate.data[0, :].view(-1, 40)

                ''' dmp_v = trainer.createDMP(y_val[0], self.scale, 0.01, 25, True)
                dmp = trainer.createDMP(output_data_validate[0], self.scale, 0.01, 25, True)

                dmp.joint()
                dmp_v.joint()

                mat = trainer.show_dmp(images[0], None, dmp)       #ni prava slika!!!!!!!!!!!!!!!


                mat_n = np.array(mat, dtype='f')
                mat_n = mat_n.swapaxes(1, 2)
                mat_n = mat_n.swapaxes(0, 1)
                mat_t = torch.from_numpy(mat_n)


                writer.add_image("result", mat_t)'''

                



            if t % log_interval == 0:
                self.loss = self.loss * bunch/len(input_data_train)
                print('Epoch: ', t, ' loss: ', self.loss.data[0])
                if livePlot:
                        plt.plot(t, self.loss.data[0], 'ob')
                        plt.pause(0.5)
                #if (self.loss - oldLoss).data[0] == 0:
                    #print("Loss hasn't changed in last ", log_interval ," iterations .Quiting...")
                    #return
                oldLoss = self.loss



            if (t-1)%5 ==0:
                writer.add_scalar('data/scalar1', self.loss, t)
                if t == 1:
                    window = webbrowser.open_new('http://fangorn:6006')


            #End condition
            if inf_k*t > inf_k*epochs:

                train = False
            if val_count > 30:

                train=False


        writer.close()
        proces.terminate()


    def learn_one_step(self,x,y,learning_rate,criterion,optimizer):
        y_pred = self(x) # output from the network
        loss = criterion(y_pred,y) #loss
        optimizer.zero_grad()# setting gradients to zero
        loss.backward()# calculating gradients for every layer
        optimizer.step()#updating weights
        self.loss = self.loss + loss

    def isCuda(self):
        return self.inputLayer.weight.is_cuda
