import scipy.io as sio
import torch
from torch.autograd import Variable
import numpy as np


class Mapping:
    y_max = 1
    y_min = -1
    x_max = []
    x_min = []


class MatLoader:
    def load_data(file, load_original_trajectories=False):
        y_max = 1
        y_min = -1
        data = sio.loadmat(file)
        data = data['slike']
        images = []

        for image in data['im'][0, 0][0]:
            images.append(image.astype('float').reshape(40*40))

        images = np.array(images)
        DMP_data = data['DMP_object'][0, 0][0]
        outputs = []
        for dmp in DMP_data:
            tau = dmp['tau'][0, 0][0, 0]
            w = dmp['w'][0, 0]
            goal = dmp['goal'][0, 0][0]
            y0 = dmp['y0'][0, 0][0]
            # dy0 = np.array([0,0])
            learn = np.append(tau,y0)
            # learn = np.append(learn,dy0)
            learn = np.append(learn,goal)#korekcija
            learn = np.append(learn,w)
            outputs.append(learn)
        outputs = np.array(outputs)

        '''scale = np.array([np.abs(outputs[:, i]).max() for i in range(0, 5)])

        scale = np.concatenate((scale, np.array([np.abs(outputs[:, 5:outputs.shape[1]]).max() for i in range(5, outputs.shape[1])])))
        '''

        x_max = np.array([outputs[:, i].max() for i in range(0, 5)])
        x_max = np.concatenate(
            (x_max, np.array([outputs[:, 5:outputs.shape[1]].max() for i in range(5, outputs.shape[1])])))

        x_min = np.array([outputs[:, i].min() for i in range(0, 5)])
        x_min = np.concatenate(
            (x_min, np.array([outputs[:, 5:outputs.shape[1]].min() for i in range(5, outputs.shape[1])])))

        scale = x_max-x_min
        scale[np.where(scale == 0)] = 1

        outputs =(y_max-y_min) * (outputs-x_min) / scale + y_min

        original_trj = []
        if load_original_trajectories:
            trj_data = data['trj'][0, 0][0]

            original_trj = [(trj) for trj in trj_data[:]]

        scaling = Mapping()
        scaling.x_max = x_max
        scaling.x_min = x_min
        scaling.y_max = y_max
        scaling.y_min = y_min

        return images, outputs, scaling, original_trj

    def data_for_network(images, outputs):
        input_data = Variable(torch.from_numpy(images)).float()
        output_data = Variable(torch.from_numpy(outputs), requires_grad=False).float()
        return input_data, output_data