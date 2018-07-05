# -*- coding: utf-8 -*-
"""
Trainer class

Created on Dec 14 2017

@author: Marcel Salmic

VERSION 1.0
"""
from mnist import MNIST# pip3 install python-mnist
import torch

import numpy as np
import copy
from scipy.interpolate import interpn

from help_programs.trajectory_loader import trajectory_loader as loader
from DMP_class import DMP
from tensorboardX import SummaryWriter


import subprocess
import webbrowser

import tkinter as tk
from datetime import datetime
import math
import random
from torch.autograd import Variable
import matplotlib.pyplot as plt

import custom_torch_optim


class Trainer:
    """
    Helper class containing methods for preparing data for learning
    """
    train=False
    user_stop =""
    plot_im = False
    indeks = []
    reseting_optimizer = False
    def show_dmp(self, image, trajectory, dmp, plot = False, save = -1):
        """
        Plots and shows mnist image, trajectory and dmp to one picture

        image -> mnist image of size 784
        trajectory -> a trajectory containing all the points in format point = [x,y]
        dmp -> DMP created from the trajectory
        """

        fig = plt.figure()
        if image is not None:
            plt.imshow((np.reshape(image, (40, 40))),cmap='gray', extent=[0, 40,40 ,0 ])


        if dmp is not None:
            dmp.joint()
            plt.plot(dmp.Y[:,0], dmp.Y[:,1],'--r', label='dmp')
        if trajectory is not None:
            plt.plot(trajectory[:,0], trajectory[:,1],'-g', label='trajectory')
        plt.legend()
        plt.xlim([0,40])
        plt.ylim([40,0])

        fig.canvas.draw()
        matrix = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        #if save != -1:
        #    plt.savefig("images/" + str(save) + ".pdf")
        #    plt.close(fig)
        #else:
        #    plt.show()
        if plot:
            plt.show()

        return matrix.reshape(fig.canvas.get_width_height()[::-1]+(3,))



    def loadMnistData(mnist_folder):
        """
        Loads data from the folder containing mnist files
        """
        mnistData = MNIST(mnist_folder)
        images, labels = mnistData.load_training()
        images = np.array(images)
        labels = np.array(labels)
        return images,labels

    def loadTrajectories(trajectories_folder, avaliable):
        """
        loads trajectories from the folder containing trajectory files
        """
        trajectories = []
        for i in avaliable:
            t = loader.loadNTrajectory(trajectories_folder,i)
            trajectories.append(t)
        trajectories = np.array(trajectories)
        return trajectories

    def createDMPs(trajectories,N, sampling_time):
        """
        Creates DMPs from the trajectorires

        trajectories -> list of trajectories to convert to DMPs
        N -> ampunt of base functions in the DMPs
        sampling_time -> sampling time for the DMPs
        """
        DMPs = []
        i = 0
        for trajectory in trajectories:
            dmp = DMP(N,sampling_time)
            x = trajectory[:,0]
            y = trajectory[:,1]
            time = np.array([trajectory[:,2],trajectory[:,2]]).transpose()[:-2]
            dt = np.diff(trajectory[:,2],1)
            if (dt == 0).any():
                print("Problem with ", i, " -th trajectory")
            dx = np.diff(x,1)/dt
            dy = np.diff(y,1)/dt
            ddy = np.diff(dy,1)/dt[:-1]
            ddx = np.diff(dx,1)/dt[:-1]
            path = np.array([i for i in zip(x,y)])[:-2]
            velocity = np.array([i for i in zip(dx,dy)])[:-1]
            acceleration = np.array([i for i in zip(ddx,ddy)])
            try:
                dmp.track(time, path, velocity, acceleration)
            except:
                print("Problem with ", i, " -th trajectory")
            DMPs.append(dmp)
            i += 1
        DMPs = np.array(DMPs)
        return DMPs

    def createOutputParameters(DMPs, scale = None):
        """
        Returns desired output parameters for the network from the given DMPs

        createOutputParameters(DMPs) -> parameters for each DMP in form [tau, y0, dy0, goal, w]
        DMPs -> list of DMPs that pair with the images input to the network
        """
        outputs = []
        for dmp in DMPs:
            learn = np.append(dmp.tau[0], dmp.y0)
            learn = np.append(learn, dmp.dy0)
            learn = np.append(learn, dmp.goal)
            learn = np.append(learn, dmp.w)
            outputs.append(learn)
        outputs = np.array(outputs)
        if scale is None:
            scale = np.array([np.abs(outputs[:,i]).max() for i in range(outputs.shape[1])])
            scale[7:] = scale[7:].max()
        outputs = outputs / scale
        return outputs, scale

    def getDataForNetwork(self,images,DMPs, scale = None, useData = None):
        """
        Generates data that will be given to the Network

        getDataForNetwork(images,DMPs,i,j) -> (input_data,output_data) for the Network
        images -> MNIST images that will be fed to the Network
        DMPs -> DMPs that pair with MNIST images given in the same order
        useData -> array like containing indexes of images to use
        """
        if useData is not None:
            input_data = Variable(torch.from_numpy(images[useData])).float()
        else:
            input_data = Variable(torch.from_numpy(images)).float()
        input_data = input_data/128 - 1
        if DMPs is not None:
            if scale is None:
                outputs, scale = Trainer.createOutputParameters(DMPs)
            else:
                outputs, scale = Trainer.createOutputParameters(DMPs, scale)
            output_data = Variable(torch.from_numpy(outputs),requires_grad= False).float()
        else:
            output_data = None
            scale = 1
        return input_data, output_data, scale


    def getDMPFromImage(self, network,image, N, sampling_time, cuda = False):
        if cuda:
          image = image.cuda()
        output = network(image)
        dmps = []
        if len(image.size()) == 1:
            dmps.append(Trainer.createDMP(output, network.scale,sampling_time,N, cuda))
        else:
            for data in output:
                dmps.append(Trainer.createDMP(data, network.scale,sampling_time,N, cuda))
        return dmps

    def createDMP(self, output, scale, sampling_time, N, cuda = False):
        if cuda:
          output = output.cpu()


        output = (scale.x_max-scale.x_min) * (output.double().data.numpy() -scale.y_min) / (scale.y_max-scale.y_min) + scale.x_min

        tau = output[0]
        y0 = output[1:3]
        goal = output[3:5]
        weights = output[5:]
        w = weights.reshape(N,2)
        #y0 = output[0:2]
        #dy0 = 0*output[0:2]
        #goal = output[2:4]
        #weights = output[4:]
        dmp = DMP(N,sampling_time)
        dmp.values(N,sampling_time,tau,y0,[0,0],goal,w)
        return dmp

    def showNetworkOutput(network, i, images, trajectories, DMPs, N, sampling_time, avaliable = None, cuda = False):
        input_data, output_data, scale = Trainer.getDataForNetwork(images, DMPs, avaliable)
        scale = network.scale
        if i != -1:
            input_data = input_data[i]
        dmps = Trainer.getDMPFromImage(network, input_data,N,sampling_time, cuda)
        for dmp in dmps:
            dmp.joint()

        if i != -1:
            print('Dmp from network:')
            Trainer.printDMPdata(dmps[0])
            print()
        if DMPs is not None and i != -1:
            print('Original DMP from trajectory:')
            Trainer.printDMPdata(DMPs[i])
        if i == -1:
            plt.ion()
            for i in range(len(dmps)):
                Trainer.show_dmp(images[i], None, dmps[i])
            plt.ioff()
        else:
            if avaliable is not None:
                Trainer.show_dmp(images[avaliable[i]], trajectories[i], dmps[0])
            elif trajectories is not None:
                Trainer.show_dmp(images[i], trajectories[i], dmps[0])
            else:
                Trainer.show_dmp(images[i], None, dmps[0])

    def printDMPdata(self,dmp):
        print('Tau: ', dmp.tau)
        print('y0: ', dmp.y0)
        print('dy0: ', dmp.dy0)
        print('goal: ', dmp.goal)
        print('w_sum: ', dmp.w.sum())

    def rotationMatrix(theta, dimensions = 3):
        c, s = np.cos(theta), np.sin(theta)
        if dimensions == 3:
            return np.array([[c,-s, 0],[s,c, 0], [0,0,1]])
        else:
            return np.array([[c,-s],[s,c]])

    def translate(points, movement):
        pivotPoint = np.array(movement)
        new_points = np.array(points) + pivotPoint
        return new_points

    def rotateAround(trajectory, pivotPoint, theta):
        pivotPoint = np.append(pivotPoint,0)
        transformed_trajectory = Trainer.translate(trajectory, -pivotPoint)
        transformed_trajectory = Trainer.rotationMatrix(theta).dot(transformed_trajectory.transpose()).transpose()
        transformed_trajectory = Trainer.translate(transformed_trajectory, pivotPoint)
        return transformed_trajectory

    def rotateImage(image, theta):
        new_image = image.reshape(28,28)
        points =  np.array([[j,i,0] for j in np.arange(28) for i in range(28)])
        transformed = Trainer.rotateAround(points, [12,12], theta)[:,:2]
        t = (points[:28,1], points[:28,1])
        return interpn(t, image.reshape(28,28), transformed, method = 'linear', bounds_error=False, fill_value=0)

    def randomlyRotateData(trajectories, images, n):
        transformed_trajectories = []
        transformed_images =[]
        for i in range(len(trajectories)):
            trajectory = trajectories[i]
            image = images[i]
            transformed_images.append(image)
            transformed_trajectories.append(trajectory)
            for j in range(n):
                theta = (np.random.rand(1)*np.pi/9)[0]
                new_trajectory = Trainer.rotateAround(trajectory, [12,12], theta)
                new_image = Trainer.rotateImage(image, theta)
                transformed_images.append(new_image)
                transformed_trajectories.append(new_trajectory)
        transformed_trajectories = np.array(transformed_trajectories)
        transformed_images = np.array(transformed_images)
        return transformed_trajectories, transformed_images


    def testOnImage(file, model):
        image = plt.imread(file)
        transformed = np.zeros([28,28])
        for i in range(28):
            for j in range(28):
                transformed[i,j] = image[i,j].sum()/3
        transformed /= transformed.max()
        plt.figure()
        plt.imshow(image)
        plt.figure()
        plt.imshow(transformed,cmap='gray')
        plt.show()
        Trainer.showNetworkOutput(model, 0, np.array([transformed.reshape(784)*255]), None, None, N, sampling_time, cuda = cuda)

    def databaseSplit(self, images, outputs, train_set = 0.7, validation_set = 0.15, test_set = 0.15):

        r=len(images)
        de =int(len(outputs)/r)
        trl=round(r*train_set)
        tel=round(r*test_set)
        val=r-trl-tel

        indeks = np.append(np.zeros(trl), np.ones(tel))
        indeks = np.append(indeks, 2*np.ones(val))

        random.shuffle(indeks)
        x_t=[]
        y_t =[]
        x_v= []
        y_v = []
        x_te= []
        y_te= []

        if self.indeks != []:
            indeks = self.indeks
        else:
            self.indeks = indeks

        for i in range(0, len(indeks)):

            if indeks[i] == 0:
                x_t.append(images[i])
                y_t.append(outputs[i * de])

                if de >1:
                    y_t.append(outputs[i * de+1])

            if indeks[i] == 2:
                x_v.append(images[i])
                y_v.append(outputs[i*de])
                if de >1:
                    y_v.append(outputs[i * de+1])

            if indeks[i] == 1:
                x_te.append(images[i])
                y_te.append(outputs[i*de])
                if de >1:
                    y_te.append(outputs[i * de+1])

        x_train = np.array(x_t)
        y_train = np.array(y_t)
        x_validate = np.array(x_v)
        y_validate = np.array(y_v)
        x_test = np.array(x_te)
        y_test = np.array(y_te)

        input_data_train = Variable(torch.from_numpy(x_train)).float()
        output_data_train = Variable(torch.from_numpy(y_train), requires_grad=False).float()
        input_data_test = Variable(torch.from_numpy(x_test)).float()
        output_data_test = Variable(torch.from_numpy(y_test), requires_grad=False).float()
        input_data_validate = Variable(torch.from_numpy(x_validate)).float()
        output_data_validate = Variable(torch.from_numpy(y_validate), requires_grad=False).float()

        return input_data_train, output_data_train, input_data_test, output_data_test,  input_data_validate, output_data_validate

    def learn(self, model, images, outputs, path, train_param, file, learning_rate = 1e-4, momentum = 0, decay = [0,0]):
        """
        teaches the network using provided data

        x -> input for the Network
        y -> desired output of the network for given x
        epochs -> how many times to repeat learning_rate
        learning_rate -> how much the weight will be changed each epoch
        log_interval -> on each epoch divided by log_interval log will be printed
        """
        root = tk.Tk()


        button = tk.Button(root,
                           text="QUIT",
                           fg="red",
                           command=self.cancel_training)
        button1 = tk.Button(root,
                           text="plot",
                           fg="blue",
                           command=self.plot_image)

        buttonResetAdam = tk.Button(root,
                            text="reset ADAM",
                            fg="blue",
                            command=self.reset_ADAM)

        button.pack(side=tk.LEFT)
        button1.pack(side=tk.RIGHT)
        buttonResetAdam.pack(side=tk.RIGHT)


        #prepare parameters
        starting_time = datetime.now()
        train_param.data_samples = len(images)
        val_count = 0
        old_time_d = 0
        oldLoss = 0
        saving_epochs = 0

        file.write(train_param.write_out())
        print('Starting learning')
        print(train_param.write_out())

        # learn

        writer = SummaryWriter(path+'/log')



        command = ["tensorboard", "--logdir=" + path+"/log"]
        proces = subprocess.Popen(command)
        print(proces.pid)
        print('init')



        # Divide data
        print("Dividing data")
        input_data_train_b, output_data_train_b, input_data_test_b, output_data_test_b, input_data_validate_b, output_data_validate_b = self.databaseSplit(images, outputs)

        #dummy = model(torch.autograd.Variable(torch.rand(1,1600)))
        #writer.add_graph(model, dummy)
        window = webbrowser.open_new('http://fangorn:6006')


        if train_param.cuda:
            '''model = model.double().cuda()
            input_data_train_b = input_data_train_b.double().cuda()
            output_data_train_b = output_data_train_b.double().cuda()
            input_data_test_b = input_data_test_b.double().cuda()
            output_data_test_b = output_data_test_b.double().cuda()
            input_data_validate_b = input_data_validate_b.double().cuda()
            output_data_validate_b = output_data_validate_b.double().cuda()'''
            model = model.cuda()
            input_data_train_b = input_data_train_b.cuda()
            output_data_train_b = output_data_train_b.cuda()
            input_data_test_b = input_data_test_b.cuda()
            output_data_test_b = output_data_test_b.cuda()
            input_data_validate_b = input_data_validate_b.cuda()
            output_data_validate_b = output_data_validate_b.cuda()


        print('finish dividing')

        criterion = torch.nn.MSELoss(size_average=True) #For calculating loss (mean squared error)
        #optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, mometum=momentum) # for updating weights
        #optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay = decay[0], weight_decay = decay[1]) #, momentum=momentum) # for updating weights
        #optimizer = torch.optim.Adam(model.parameters(), eps = 0.001 )
        #optimizer = torch.optim.SGD(model.parameters(), lr = 0.4) # for updating weights
        #optimizer = torch.optim.RMSprop(model.parameters())

        optimizer = custom_torch_optim.SCG(model.parameters())
        #optimizer = correct_adam.Adam(model.parameters(), lr = 0.0001, amsgrad=True)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)

        y_val = model(input_data_validate_b)
        oldValLoss = criterion(y_val, output_data_validate_b[:, 1:]).data[0]
        bestValLoss = oldValLoss
        best_nn_parameters = copy.deepcopy(model.state_dict())
        #Infiniti epochs
        if train_param.epochs == -1:
            inf_k = 0
        else:
            inf_k = 1

        self.train = True

        t = 0

        t_init = 200
        lr = 0

        while self.train:

            input_data_train = input_data_train_b.clone()
            output_data_train = output_data_train_b.clone()
            input_data_test = input_data_test_b.clone()
            output_data_test = output_data_test_b.clone()
            input_data_validate = input_data_validate_b.clone()
            output_data_validate = output_data_validate_b.clone()

            root.update()
            t = t+1
            i = 0
            j = train_param.bunch


            #scheduler.step()
            #if t%t_init == 0:
            #    scheduler.last_epoch = -1


            #writer.add_scalar('data/learning_rate', scheduler.get_lr()[0], t)



            self.loss = Variable(torch.Tensor([0]))
            permutations = torch.randperm(len(input_data_train))
            if model.isCuda():
                permutations = permutations.cuda()
                self.loss = self.loss.cuda()
            input_data_train = input_data_train[permutations]
            output_data_train = output_data_train[permutations]
            ena = []
            while j <= len(input_data_train):
                self.learn_one_step(model,input_data_train[i:j], output_data_train[i:j, 1:], learning_rate,criterion,optimizer)
                i = j
                j += train_param.bunch

                '''for group in optimizer.param_groups:
                    i = 0
                    for p in group['params']:
                        i = i+1
                        if i ==15:

                            r1 = p.data[0][0]'''

            if i < len(input_data_train):
                self.learn_one_step(model,input_data_train[i:], output_data_train[i:, 1:], learning_rate,criterion,optimizer)


            if (t-1)%train_param.log_interval ==0:

                self.loss = self.loss * train_param.bunch / len(input_data_train)
                if t == 1:

                    oldLoss = self.loss

                print('Epoch: ', t, ' loss: ', self.loss.data[0])
                time_d = datetime.now()-starting_time
                writer.add_scalar('data/time', t, time_d.total_seconds())
                writer.add_scalar('data/training_loss', math.log( self.loss), t)
                writer.add_scalar('data/epochs_speed', 60*train_param.log_interval/(time_d.total_seconds()-old_time_d), t)
                writer.add_scalar('data/gradient_of_performance', (self.loss-oldLoss)/train_param.log_interval, t)
                old_time_d = time_d.total_seconds()
                oldLoss = self.loss


            if (t-1)%train_param.validation_interval == 0:
                y_val = model(input_data_validate)
                val_loss = criterion(y_val, output_data_validate[:, 1:])
                writer.add_scalar('data/val_loss', math.log(val_loss), t)
                if val_loss.data[0] < bestValLoss:
                    bestValLoss = val_loss.data[0]
                    best_nn_parameters = copy.deepcopy(model.state_dict())
                    saving_epochs = t

                if val_loss.data[0] > bestValLoss:#oldValLoss:
                    val_count = val_count+1

                else:

                    val_count = 0



                oldValLoss = val_loss.data[0]
                writer.add_scalar('data/val_count', val_count, t)
                print('Validatin: ', t, ' loss: ', val_loss.data[0], ' best loss:', bestValLoss)

                if (t - 1) % 10 == 0:
                    state = model.state_dict()
                    mean_dict = dict()
                    max_dict = dict()
                    min_dict = dict()
                    var_dict = dict()
                    for grup in state:
                        mean = torch.mean(state[grup])
                        max = torch.max(state[grup])
                        min = torch.min(state[grup])
                        var = torch.var(state[grup])
                        mean_dict[grup] = mean
                        max_dict[grup] = max
                        min_dict[grup] = min
                        var_dict[grup] = var

                    writer.add_scalars('data/mean', mean_dict, t)
                    writer.add_scalars('data/max', max_dict, t)
                    writer.add_scalars('data/min', min_dict, t)
                    writer.add_scalars('data/var', var_dict, t)

               
                if self.plot_im:

                    plot_vector = torch.cat((output_data_validate[0,0:1], y_val[0, :]), 0)
                    dmp_v = self.createDMP(plot_vector, model.scale, 0.01, 25, True)
                    dmp = self.createDMP(output_data_validate[0,:], model.scale, 0.01, 25, True)
                    dmp.joint()
                    dmp_v.joint()
                    mat = self.show_dmp((input_data_validate.data[0]).cpu().numpy(), dmp.Y , dmp_v, plot=False)
                    a = output_data_validate[0, :]
                    writer.add_image('image'+str(t), mat)
                    self.plot_im = False

                    torch.save(model.state_dict(), path + '/net_parameters' +str(t))





            if (t - 1) % train_param.test_interval == 0:
                y_test = model(input_data_test)
                test_loss = criterion(y_test, output_data_test[:, 1:])
                writer.add_scalar('data/test_loss', math.log(test_loss), t)

            '''if (t-1) % 1500 == 0:
                optimizer.reset = True
                print('reset optimizer')
            '''
            if self.reseting_optimizer:
                optimizer.reset = True
            if val_count > 7:

                train_param.stop_criterion = "reset optimizer"
                optimizer.reset = True

            #End condition
            if inf_k*t > inf_k*train_param.epochs:
                self.train = False
                train_param.stop_criterion = "max epochs reached"

            if val_count > train_param.val_fail:
                self.train = False
                train_param.stop_criterion = "max validation fail reached"



            '''
            writer.add_scalar('data/test_lr', lr, t)


            writer.add_scalar('data/loss_lr', self.loss.data[0], lr)

            lr = 1.1**(t/40)-1
            print(lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            '''

        train_param.real_epochs = t
        train_param.min_train_loss = self.loss.data[0]
        train_param.min_val_loss = bestValLoss
        train_param.min_test_loss = test_loss
        train_param.elapsed_time = time_d.total_seconds()
        train_param.val_count = val_count
        k = (self.loss-oldLoss)/train_param.log_interval
        train_param.min_grad = k.data[0]
        train_param.stop_criterion = train_param.stop_criterion + self.user_stop

        file.write('\n'+str(optimizer))
        file.write('\n'+str(criterion))
        file.write('\n saving_epochs = ' + str(saving_epochs))
        file.write(train_param.write_out_after())
        writer.close()
        proces.terminate()

        print('Learning finished\n')

        return best_nn_parameters

    def learn_DMP(self, model, images, outputs, path, train_param, file, learning_rate=1e-4, momentum=0, decay=[0, 0]):
        """
        teaches the network using provided data

        x -> input for the Network
        y -> desired output of the network for given x
        epochs -> how many times to repeat learning_rate
        learning_rate -> how much the weight will be changed each epoch
        log_interval -> on each epoch divided by log_interval log will be printed
        """
        root = tk.Tk()

        button = tk.Button(root,
                           text="QUIT",
                           fg="red",
                           command=self.cancel_training)
        button1 = tk.Button(root,
                            text="plot",
                            fg="blue",
                            command=self.plot_image)

        buttonResetAdam = tk.Button(root,
                                    text="reset ADAM",
                                    fg="blue",
                                    command=self.reset_ADAM)

        button.pack(side=tk.LEFT)
        button1.pack(side=tk.RIGHT)
        buttonResetAdam.pack(side=tk.RIGHT)

        # prepare parameters
        starting_time = datetime.now()
        train_param.data_samples = len(images)
        val_count = 0
        old_time_d = 0
        oldLoss = 0
        saving_epochs = 0

        file.write(train_param.write_out())
        print('Starting learning')
        print(train_param.write_out())

        # learn

        writer = SummaryWriter(path + '/log')

        command = ["tensorboard", "--logdir=" + path + "/log"]
        proces = subprocess.Popen(command)
        print(proces.pid)
        print('init')

        # Divide data
        print("Dividing data")
        input_data_train_b, output_data_train_b, input_data_test_b, output_data_test_b, input_data_validate_b, output_data_validate_b = self.databaseSplit(
            images, outputs)

        # dummy = model(torch.autograd.Variable(torch.rand(1,1600)))
        # writer.add_graph(model, dummy)
        window = webbrowser.open_new('http://fangorn:6006')

        if train_param.cuda:
            '''model = model.double().cuda()
            input_data_train_b = input_data_train_b.double().cuda()
            output_data_train_b = output_data_train_b.double().cuda()
            input_data_test_b = input_data_test_b.double().cuda()
            output_data_test_b = output_data_test_b.double().cuda()
            input_data_validate_b = input_data_validate_b.double().cuda()
            output_data_validate_b = output_data_validate_b.double().cuda()'''
            model = model.cuda()
            input_data_train_b = input_data_train_b.cuda()
            output_data_train_b = output_data_train_b.cuda()
            input_data_test_b = input_data_test_b.cuda()
            output_data_test_b = output_data_test_b.cuda()
            input_data_validate_b = input_data_validate_b.cuda()
            output_data_validate_b = output_data_validate_b.cuda()

        print('finish dividing')

        criterion = torch.nn.MSELoss(size_average=True)  # For calculating loss (mean squared error)
        # optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, mometum=momentum) # for updating weights
        # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay = decay[0], weight_decay = decay[1]) #, momentum=momentum) # for updating weights
        #optimizer = torch.optim.Adam(model.parameters(), eps = 0.001 )
        # optimizer = torch.optim.SGD(model.parameters(), lr = 0.4) # for updating weights
        # optimizer = torch.optim.RMSprop(model.parameters())

        optimizer = custom_torch_optim.SCG(model.parameters())
        # optimizer = correct_adam.Adam(model.parameters(), lr = 0.0001, amsgrad=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)

        y_val = model(input_data_validate_b)

        oldValLoss = criterion(y_val, output_data_validate_b[:, :])
        bestValLoss = oldValLoss
        best_nn_parameters = copy.deepcopy(model.state_dict())

        fig = plt.figure()

        plt.imshow((np.reshape(input_data_validate_b.data[0].cpu().numpy(), (40, 40))), cmap='gray',
                   extent=[0, 40, 40, 0])

        plt.plot(output_data_validate_b.data[0].cpu().numpy(), output_data_validate_b.data[1].cpu().numpy(), '--r',
                 label='dmp')

        plt.plot(y_val.data[0].cpu().numpy(), y_val.data[1].cpu().numpy(), '-g', label='trajectory')
        plt.legend()
        plt.xlim([0, 40])
        plt.ylim([40, 0])

        fig.canvas.draw()
        matrix = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')

        mat = matrix.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        writer.add_image('image' + str(0), mat)
        self.plot_im = False

        torch.save(model.state_dict(), path + '/net_parameters' + str(0))










        # Infiniti epochs
        if train_param.epochs == -1:
            inf_k = 0
        else:
            inf_k = 1

        self.train = True

        t = 0

        t_init = 200
        lr = 0

        while self.train:

            input_data_train = input_data_train_b.clone()
            output_data_train = output_data_train_b.clone()
            input_data_test = input_data_test_b.clone()
            output_data_test = output_data_test_b.clone()
            input_data_validate = input_data_validate_b.clone()
            output_data_validate = output_data_validate_b.clone()

            root.update()
            t = t + 1
            i = 0
            j = train_param.bunch

            # scheduler.step()
            # if t%t_init == 0:
            #    scheduler.last_epoch = -1

            # writer.add_scalar('data/learning_rate', scheduler.get_lr()[0], t)

            self.loss = Variable(torch.Tensor([0]))
            if t==1:
                permutations = torch.randperm(len(input_data_train))
                if model.isCuda():
                    permutations = permutations.cuda()
                    self.loss = self.loss.cuda()
            if model.isCuda():

                self.loss = self.loss.cuda()
            input_data_train = input_data_train[permutations]
            per = torch.stack([permutations*2,permutations*2+1]).transpose(1,0).contiguous().view(1,-1).squeeze()

            output_data_train = output_data_train[per]
            ena = []

            while j <= len(input_data_train):
                self.learn_one_step(model, input_data_train[i:j], output_data_train[i*2:j*2, :], learning_rate, criterion,
                                    optimizer)
                i = j
                j += train_param.bunch

                '''for group in optimizer.param_groups:
                    i = 0
                    for p in group['params']:
                        i = i+1
                        if i ==15:

                            r1 = p.data[0][0]'''

            if i < len(input_data_train):
                self.learn_one_step(model, input_data_train[i:], output_data_train[i*2:, :], learning_rate, criterion,
                                    optimizer)

            if (t - 1) % train_param.log_interval == 0:

                self.loss = self.loss * train_param.bunch / len(input_data_train)
                if t == 1:
                    oldLoss = self.loss

                print('Epoch: ', t, ' loss: ', self.loss.data[0])
                time_d = datetime.now() - starting_time
                writer.add_scalar('data/time', t, time_d.total_seconds())
                writer.add_scalar('data/training_loss', math.log(self.loss), t)
                writer.add_scalar('data/epochs_speed',
                                  60 * train_param.log_interval / (time_d.total_seconds() - old_time_d), t)
                writer.add_scalar('data/gradient_of_performance', (self.loss - oldLoss) / train_param.log_interval, t)
                old_time_d = time_d.total_seconds()
                oldLoss = self.loss

            if (t - 1) % train_param.validation_interval == 0:
                y_val = model(input_data_validate)

                val_loss = criterion(y_val, output_data_validate[:, :])

                writer.add_scalar('data/val_loss', math.log(val_loss), t)
                if val_loss < bestValLoss:
                    bestValLoss = val_loss
                    best_nn_parameters = copy.deepcopy(model.state_dict())
                    saving_epochs = t

                if val_loss > bestValLoss:  # oldValLoss:
                    val_count = val_count + 1

                else:

                    val_count = 0

                oldValLoss = val_loss
                writer.add_scalar('data/val_count', val_count, t)
                print('Validatin: ', t, ' loss: ', val_loss, ' best loss:', bestValLoss)

                if (t - 1) % 10 == 0:
                    state = model.state_dict()
                    mean_dict = dict()
                    max_dict = dict()
                    min_dict = dict()
                    var_dict = dict()
                    for grup in state:
                        mean = torch.mean(state[grup])
                        max = torch.max(state[grup])
                        min = torch.min(state[grup])
                        var = torch.var(state[grup])
                        mean_dict[grup] = mean
                        max_dict[grup] = max
                        min_dict[grup] = min
                        var_dict[grup] = var

                    writer.add_scalars('data/mean', mean_dict, t)
                    writer.add_scalars('data/max', max_dict, t)
                    writer.add_scalars('data/min', min_dict, t)
                    writer.add_scalars('data/var', var_dict, t)

                if self.plot_im:

                    fig = plt.figure()

                    plt.imshow((np.reshape(input_data_validate.data[0].cpu().numpy(), (40, 40))), cmap='gray', extent=[0, 40, 40, 0])


                    plt.plot(output_data_validate.data[0].cpu().numpy(), output_data_validate.data[1].cpu().numpy(), '--r', label='dmp')

                    plt.plot(y_val.data[0].cpu().numpy(),y_val.data[1].cpu().numpy(), '-g', label='trajectory')
                    plt.legend()
                    plt.xlim([0, 40])
                    plt.ylim([40, 0])

                    fig.canvas.draw()
                    matrix = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')



                    mat= matrix.reshape(fig.canvas.get_width_height()[::-1] + (3,))



                    writer.add_image('image' + str(t), mat)
                    self.plot_im = False

                    torch.save(model.state_dict(), path + '/net_parameters' + str(t))

            if (t - 1) % train_param.test_interval == 0:
                y_test = model(input_data_test)
                test_loss = criterion(y_test, output_data_test[:, :])
                writer.add_scalar('data/test_loss', math.log(test_loss), t)

            '''if (t-1) % 1500 == 0:
                optimizer.reset = True
                print('reset optimizer')
            '''
            if self.reseting_optimizer:
                optimizer.reset = True
            if val_count > 7:
                train_param.stop_criterion = "reset optimizer"
                optimizer.reset = True

            # End condition
            if inf_k * t > inf_k * train_param.epochs:
                self.train = False
                train_param.stop_criterion = "max epochs reached"

            if val_count > train_param.val_fail:
                self.train = False
                train_param.stop_criterion = "max validation fail reached"

            '''
            writer.add_scalar('data/test_lr', lr, t)


            writer.add_scalar('data/loss_lr', self.loss.data[0], lr)

            lr = 1.1**(t/40)-1
            print(lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            '''

        train_param.real_epochs = t
        train_param.min_train_loss = self.loss.data[0]
        train_param.min_val_loss = bestValLoss
        train_param.min_test_loss = test_loss.data[0]
        train_param.elapsed_time = time_d.total_seconds()
        train_param.val_count = val_count
        k = (self.loss - oldLoss) / train_param.log_interval
        train_param.min_grad = k.data[0]
        train_param.stop_criterion = train_param.stop_criterion + self.user_stop

        file.write('\n' + str(optimizer))
        file.write('\n' + str(criterion))
        file.write('\n saving_epochs = ' + str(saving_epochs))
        file.write(train_param.write_out_after())
        writer.close()
        proces.terminate()

        print('Learning finished\n')

        return best_nn_parameters

    def learn_one_step(self, model, x, y, learning_rate, criterion, optimizer):

        def wrap():
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            return loss

        '''
        y_pred = model(x) # output from the network
        loss = criterion(y_pred,y) #loss
        optimizer.zero_grad()# setting gradients to zero
        loss.backward()# calculating gradients for every layer
        
        
        optimizer.step()#updating weights'''
        loss = optimizer.step(wrap)

        self.loss = self.loss + loss.item()

    def cancel_training(self):

        self.user_stop = "User stop"
        self.train = False

    def plot_image(self):
        print("plot image")
        self.plot_im = True

    def reset_ADAM(self):
        self.reseting_optimizer = True
        print('reseting ADAM')