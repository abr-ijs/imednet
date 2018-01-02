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
from scipy.interpolate import interpn

from trajectory_loader import trajectory_loader as loader
from DMP_class import DMP


class Trainer:
    """
    Helper class containing methods for preparing data for learning
    """

    def show_dmp(image,trajectory, dmp, save = -1):
        """
        Plots and shows mnist image, trajectory and dmp to one picture

        image -> mnist image of size 784
        trajectory -> a trajectory containing all the points in format point = [x,y]
        dmp -> DMP created from the trajectory
        """
        fig = plt.figure()
        if (image != None).any():
            plt.imshow((np.reshape(image, (28, 28))).astype(np.uint8), cmap='gray')
        if dmp != None:
            dmp.joint()
            plt.plot(dmp.Y[:,0], dmp.Y[:,1],'--r', label='dmp')
        plt.plot(trajectory[:,0], trajectory[:,1],'-g', label='trajectory')
        plt.legend()
        plt.xlim([0,28])
        plt.ylim([28,0])
        if save != -1:
            plt.savefig("images/" + str(save) + ".pdf")
            plt.close(fig)
        else:
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

    def loadTrajectories(trajectories_folder, avaliable):
        """
        loads trajectories from the folder containing trajectory files
        """
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
            dmp.track(time, path, velocity, acceleration)
            DMPs.append(dmp)
            i += 1
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
        scale = np.array([np.abs(outputs[:,i]).max() for i in range(outputs.shape[1])])
        outputs = outputs / scale
        return outputs, scale

    def getDataForNetwork(images,DMPs,useData = None):
        """
        Generates data that will be given to the Network

        getDataForNetwork(images,DMPs,i,j) -> (input_data,output_data) for the Network
        images -> MNIST images that will be fed to the Network
        DMPs -> DMPs that pair with MNIST images given in the same order
        useData -> array like containing indexes of images to use
        """
        outputs, scale = Trainer.createOutputParameters(DMPs)
        if useData is not None:
            input_data = Variable(torch.from_numpy(images[useData])).float()
        else:
            input_data = Variable(torch.from_numpy(images)).float()
        input_data = input_data/128 - 1
        output_data = Variable(torch.from_numpy(outputs),requires_grad= False).float()
        return input_data, output_data, scale


    def getDMPFromImage(network,image, N, sampling_time):
        output = network(image)
        output = output.double().data.numpy()[0]*network.scale
        tau = output[0]
        y0 = output[1:3]
        dy0 = output[3:5]
        goal = output[5:7]
        weights = output[7:]
        w = weights.reshape(N,2)
        dmp = DMP(N,sampling_time)
        dmp.values(N,sampling_time,tau,y0,dy0,goal,w)
        return dmp

    def showNetworkOutput(network, i, images, trajectories, avaliable, DMPs, N, sampling_time):
        input_data, output_data, scale = Trainer.getDataForNetwork(images, DMPs, avaliable)
        dmp = Trainer.getDMPFromImage(network, input_data[i],N,sampling_time)
        dmp.joint()
        print('Dmp from network:')
        Trainer.printDMPdata(dmp)
        print()
        print('Original DMP from trajectory:')
        Trainer.printDMPdata(DMPs[i])
        Trainer.show_dmp(images[avaliable[i]], trajectories[i], dmp)

    def printDMPdata(dmp):
        print('Tau: ', dmp.tau)
        print('y0: ', dmp.y0)
        print('dy0: ', dmp.dy0)
        print('goal: ', dmp.goal)
        print('w_sum: ', dmp.w.sum())

    def rotationMatrix(theta, dimensions = 3):
        c, s = np.cos(theta)[0], np.sin(theta)[0]
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
        transformed_trajectory = Trainer.translate(trajectory, pivotPoint)
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
                theta = np.random.rand(1)*np.pi*2
                new_trajectory = Trainer.rotateAround(trajectory, [12,12], theta)
                new_image = Trainer.rotateImage(image, theta)
                transformed_images.append(new_image)
                transformed_trajectories.append(new_trajectory)
        print(len(transformed_images))
        print(len(transformed_trajectories))
        transformed_trajectories = np.array(transformed_trajectories)
        transformed_images = np.array(transformed_images)
        print(transformed_images.shape)
        print(transformed_trajectories.shape)
        return transformed_trajectories, transformed_images
