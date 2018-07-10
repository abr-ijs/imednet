import sys
import numpy as np
import torch
from torch.autograd import Variable
import network_cutter

from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

from imednet.trainers.encoder_decoder_trainer import Trainer
from imednet.data.mat_loader import MatLoader

# Load model and parameters
# directory_path = '/home/rpahic/Documents/Neural_networks/'
# directory_name = 'NN ' + '2018-06-07 15:31:10.101849'
directory_path='/home/rpahic/imednet/deep_encoder_decoder_data_and_networks/networks/NN 2018-06-14 13:23:28.813445'
model = torch.load(directory_path+'/model.pt')

state = torch.load(directory_path+'/net_parameters')

model.load_state_dict(state)

dataset_name = '/home/rpahic/imednet/deep_encoder_decoder_data_and_networks/slike_780.4251'

encoder,decoder = network_cutter.NN_cut(model,6)

images, outputs, scale, original_trj = MatLoader.load_data(dataset_name, load_original_trajectories=True)

test_input = Variable(torch.from_numpy(np.array(images))).float()
test_output = Variable(torch.from_numpy(np.array(outputs))).float()

# test_input = test_input.cuda()
# test_output = test_output.cuda()
# model = model.cuda()

# nn_output = model(test_input)

latent_out = encoder(test_input)
dec_output = decoder(latent_out )

w = np.zeros((outputs.shape[0], outputs.shape[1]+1),dtype = object)
w1 = np.zeros((latent_out.shape[0], latent_out.shape[1]+1))

for i in range(0,outputs.shape[0]):
    w[i,0] = int(i + 1)
    w[i,1:] = outputs[i, :]
    w1[i, 0] = int(i + 1)
    w1[i, 1:] = latent_out.detach().numpy()[i, :]
    # w[i]=np.concatenate([[int(i+1)],outputs[i,:]])

a = ['%i']
for i in range(0, 55):
    a.append('%f')

np.savetxt('output_data_2.csv',w,fmt=a)
b = ['%i']
for i in range(0, 20):
    b.append('%f')

np.savetxt('latent_space_2.csv',w1,fmt=b)
