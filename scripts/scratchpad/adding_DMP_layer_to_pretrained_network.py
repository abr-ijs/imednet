import sys
import torch
import numpy as np
from torch.autograd import Variable
from os import makedirs
from datetime import datetime

from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

from deep_encoder_decoder_network.models.encoder_decoder import DMPEncoderDecoderNet, TrainingParameters
from deep_encoder_decoder_network.trainers.encoder_decoder_trainer import Trainer
from deep_encoder_decoder_network.data.mat_loader import MatLoader


N = 25

numOfInputs = 1600
HiddenLayer = [1500, 1300, 1000, 600, 200, 20, 35]
conv = None

date = str(datetime.now())
out = 2*N + 4
# out = 2*N

layerSizes = [numOfInputs] + HiddenLayer + [out]

net_id = '2018-06-07 15:31:10.101849'

directory_path = '/home0/barry.ridge/Documents/Neural_networks/'
directory_name = 'NN ' + net_id

date = str(datetime.now())
directory_name_new = 'NN ' + date
parameters_file = directory_path + directory_name + '/net_parameters'
makedirs(directory_path+directory_name_new)

file = open(directory_path+directory_name+'/Network_description.txt','w')

file.write('Network created: ' + date)

model_old = torch.load(directory_path+directory_name+'/model.pt')
parameters_file_new = directory_path + directory_name_new + '/net_parameters'
state = torch.load(directory_path+directory_name+'/net_parameters')

dateset_name = 'slike_780.4251'
images, outputs, scale, original_trj = MatLoader.load_data(dateset_name, load_original_trajectories=True)

file = open(directory_path+directory_name_new+'/Network_description.txt','w')

file.write('Network created: ' + date)

model = DMPEncoderDecoderNet(layerSizes, conv, scale)

model.load_state_dict(state)
model.register_buffer('DMPp', model.DMPparam.data_tensor)
model.register_buffer('scale_t', model.DMPparam.scale_tensor)
model.register_buffer('param_grad', model.DMPparam.grad_tensor)

input_image = Variable(torch.from_numpy(np.array(images[2]))).float().view(1,-1)
# trajektorija=model(input_image)
trainer = Trainer()
test_output = Variable(torch.from_numpy(np.array(outputs[2])), requires_grad=True).float()
dmp = trainer.create_dmp(test_output, scale, 0.01, 25, cuda=False)
dmp.joint()
# mat = trainer.show_dmp(input_image.data.numpy(), trajektorija.data.numpy(), dmp, plot=True)

torch.save(model, (directory_path+directory_name_new+'/model.pt'))

# Set learning

# Learning params

train_param = TrainingParameters()
train_param.epochs = -1
learning_rate = 0.0005
momentum = 0.5
train_param.batch_size = 128
train_param.training_ratio = 0.7
train_param.validation_ratio = 0.15
train_param.test_ratio = 0.15
train_param.val_fail = 60

trainer = Trainer()

trainer.indeks = np.load(directory_path + 'NN ' + net_id +'/net_indeks.npy')#[0:1000]

original_trj_e = []
for i in range(0,images.shape[0]):
    c,c1,c2 = zip(*original_trj[i])
    original_trj_e.append(c)
    original_trj_e.append(c1)

# Testing
best_nn_parameters = trainer.train_DMP(model, images, original_trj_e, directory_path + directory_name_new, train_param, file, learning_rate, momentum)

original_trj[0].reshape(1,-1)

# parameters = list(model.parameters())

# torch.save(model.state_dict(), parameters_file) # saving parameters

np.save(directory_path+directory_name_new+'/net_indeks', trainer.indeks)
torch.save(best_nn_parameters, parameters_file_new)
file.close()

