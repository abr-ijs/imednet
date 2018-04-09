import scipy.io as sio
import torch
from torch.autograd import Variable
import numpy as np

class matLoader:

    def loadData(file):
        data = sio.loadmat(file)
        data = data['slike']
        images = []
        for image in data['im'][0,0][0]:
            images.append(image.astype('float').reshape(40*40))
        images = np.array(images)
        DMP_data = data['DMP_object'][0, 0][0]
        outputs = []
        for dmp in DMP_data:
            tau = dmp['tau'][0, 0][0, 0]
            w = dmp['w'][0,0]
            goal = dmp['goal'][0,0][0]
            y0 = dmp['y0'][0,0][0]
            #dy0 = np.array([0,0])
            learn = np.append(tau,y0)
            #learn = np.append(learn,dy0)
            learn = np.append(learn,goal)#korekcija
            learn = np.append(learn,w)
            outputs.append(learn)
        outputs = np.array(outputs)

        scale = np.array([np.abs(outputs[:,i]).max() for i in range(0,5)])
        scale = np.concatenate((scale,np.array([np.abs(outputs[:,5:outputs.shape[1]]).max() for i in range(5,outputs.shape[1])])))
        scale[np.where(scale == 0)] = 1
        outputs = outputs / scale
        return images, outputs, scale

    def dataForNetwork(images, outputs):
        input_data = Variable(torch.from_numpy(images)).float()
        output_data = Variable(torch.from_numpy(outputs),requires_grad= False).float()
        return input_data, output_data
