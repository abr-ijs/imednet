# -*- coding: utf-8 -*-
"""
Trainer class

Created on Dec 14 2017

@author: Marcel Salmic

VERSION 1.0
"""
from mnist import MNIST# pip3 install python-mnist
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

from trajectory_loader import trajectory_loader as loader
from DMP_class import DMP


class Trainer:
    """
    Helper class containing methods for preparing data for learning
    """

    def show_dmp(image,trajectory, dmp):
        """
        Plots and shows mnist image, trajectory and dmp to one picture

        image -> mnist image of size 784
        trajectory -> a trajectory containing all the points in format point = [x,y]
        dmp -> DMP created from the trajectory
        """
        dmp.joint()
        if (image != None).any():
            plt.imshow((np.reshape(image, (28, 28))).astype(np.uint8), cmap='gray')
        plt.plot(dmp.Y[:,0], dmp.Y[:,1],'--r', label='dmp')
        plt.plot(trajectory[:,0], trajectory[:,1],'-g', label='trajectory')
        plt.legend()
        plt.xlim([0,28])
        plt.ylim([0,28])
        plt.show()

    def loadMnistData(mnist_folder):
        """
        Loads data from the folder containing mnist files
        """
        mnistData = MNIST(mnist_folder)
        images, labels = mnistData.load_training()
        images = np.array(images)
        labels = np.array(labels)
        return images,labels

    def loadTrajectories(trajectories_folder):
        """
        loads trajectories from the folder containing trajectory files
        """
        avaliable = loader.getAvaliableTrajectoriesNumbers(trajectories_folder)
        trajectories = []
        for i in avaliable:
            trajectories.append(loader.loadNTrajectory(trajectories_folder,i))
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
        for trajectory in trajectories:
            dmp = DMP(N,sampling_time)
            x = trajectory[:,0]
            y = trajectory[:,1]
            time = np.array([trajectory[:,2],trajectory[:,2]]).transpose()[:-2]
            dt = np.diff(trajectory[:,2],1)
            dx = np.diff(x,1)/dt
            dy = np.diff(y,1)/dt
            ddy = np.diff(dy,1)/dt[:-1]
            ddx = np.diff(dx,1)/dt[:-1]
            path = np.array([i for i in zip(x,y)])[:-2]
            velocity = np.array([i for i in zip(dx,dy)])[:-1]
            acceleration = np.array([i for i in zip(ddx,ddy)])
            dmp.track(time, path, velocity, acceleration)
            DMPs.append(dmp)
        DMPs = np.array(DMPs)
        return DMPs

    def createOutputParameters(DMPs):
        """
        Returns desired output parameters for the network from the given DMPs

        createOutputParameters(DMPs) -> parameters for each DMP in form [tau, y0, dy0, goal, w]
        DMPs -> list of DMPs that pair with the images input to the network
        """
        outputs = []
        for dmp in DMPs:
            learn = np.append(dmp.tau[0],dmp.y0)
            learn = np.append(learn,dmp.dy0)
            learn = np.append(learn,dmp.goal)
            learn = np.append(learn,dmp.w)
            outputs.append(learn)
        outputs = np.array(outputs)
        return outputs

    def getDataForNetwork(images,DMPs,i,j):
        """
        Generates data that will be given to the Network

        getDataForNetwork(images,DMPs,i,j) -> (input_data,output_data) for the Network
        images -> MNIST images that will be fed to the Network
        DMPs -> DMPs that pair with MNIST images given in the same order
        i -> lower bound of data that should be prepared for the network
        j -> upper bound of data that should be prepared for the network
        """
        outputs = Trainer.createOutputParameters(DMPs)
        input_data = Variable(torch.from_numpy(images[i:j])).float()
        output_data = Variable(torch.from_numpy(outputs[i:j]),requires_grad= False).float()
        return input_data, output_data
