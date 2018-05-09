import torch

import numpy as np
from matLoader import matLoader
from torch.autograd import Variable
from trainer import Trainer
from network import Network_DMP, training_parameters


N = 25

numOfInputs = 1600
HiddenLayer = [1500, 1300, 1000, 600, 200, 20, 35]
conv = None


out = 2*N + 4
#out = 2*N

layerSizes = [numOfInputs] + HiddenLayer + [out]



directory_path = '/home/rpahic/Documents/Neural_networks/'
directory_name = 'NN ' + '2018-04-25 13:36:32.095726'

model_old = torch.load(directory_path+directory_name+'/model.pt')

state = torch.load(directory_path+directory_name+'/net_parameters')

dateset_name = 'slike_780.4251'
images, outputs, scale, original_trj = matLoader.loadData(dateset_name, load_original_trajectories=True)



model = Network_DMP(layerSizes, conv, scale)

model.load_state_dict(state)
input_image = Variable(torch.from_numpy(np.array(images[2]))).float()
trajektorija=model(input_image)
trainer = Trainer()
test_output = Variable(torch.from_numpy(np.array(outputs[2])), requires_grad=True).float()
dmp = trainer.createDMP(test_output, scale, 0.01, 25, cuda=False)
dmp.joint()
mat = trainer.show_dmp(input_image.data.numpy(), trajektorija.data.numpy(), dmp, plot=True)

