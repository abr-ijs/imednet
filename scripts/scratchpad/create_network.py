import sys
import torch
import numpy as np

from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

from imednet.models.encoder_decoder import EncoderDecoderNet
from imednet.trainers.encoder_decoder_trainer import Trainer
from imednet.data.trajectory_loader import TrajectoryLoader

print()
## folders containing trajectories and mnist data
trajectories_folder = 'data/trajectories'
mnist_folder = 'data/mnist'
scale_file = 'scale.npy'

parameters_file = 'net_parameters'

#DMP data
N = 25
sampling_time = 0.1

#learning params
epochs = 20000
learning_rate=0.01
momentum = 0.2
decay = [1e-9,1e-6]
batch_size = 1
oneDigidOnly = False
#Good data: 0-100, 200-4500, 5000-5100
#Bad data: 100-200
use_data = np.arange(0,100)
use_data = np.append(use_data,np.arange(200,4500))
use_data = np.append(use_data,np.arange(5000,5100))
data = 4500
s_data = 2500
artificial_samples = 14
digit = 0

use_good_data = True
plot_only = False

if plot_only:
    load = True
    cuda = False
    plot = True
    load_from_cuda = True
    epochs = 0
else:
    load = False
    cuda = True
    plot = False
    load_from_cuda = False

#layers size
numOfInputs = 784
HiddenLayer = [ 600, 350, 150, 40]
conv = None
#HiddenLayer = [100]
out = 2*N + 7
#out = 2*N
layer_sizes = [numOfInputs] + HiddenLayer + [out]

print('Loading Mnist images')
#get mnist data
images, labels = Trainer.load_mnist_data(mnist_folder)
print(' Done loading Mnist images')
#get trajectories
avaliable = np.array(TrajectoryLoader.getAvaliableTrajectoriesNumbers(trajectories_folder))

if oneDigidOnly:
    indexes = np.where(labels==digit)
    indexes = np.intersect1d(indexes,avaliable)
else:
    if use_good_data:
        avaliable = avaliable[use_data]
    else:
        avaliable = avaliable[s_data:data]
    indexes = avaliable

print('Loading ',  len(indexes), ' trajectories')
trajectories = Trainer.load_trajectories(trajectories_folder, indexes)
print(' Done loading trajectories')
test = images[-100:]
print('Multiplying data')
images = images[indexes]
#trajectories, images = Trainer.randomly_rotate_data(trajectories, images, artificial_samples)
#print('Done multiplying data. Now having ',  len(trajectories), ' data')

# get DMPs
print('Creating DMPs')
DMPs = Trainer.create_dmps(trajectories, N, sampling_time)
print(' Done creating DMPs')
# get data to learn


#Code to find wrong data
#wrong = []
#for i in range(500,6100):
#    DMPs[i].joint()
#    if DMPs[i].Y.max() > 28 or DMPs[i].Y.min() < 0:
#        wrong.append(indexes[i])
#        print(indexes[i])
#
#print('rm ' + " ".join(['image_' +str(i) + '.json' for i in wrong]))

#scale = np.load(scale_file)
input_data, output_data, scale = Trainer.get_data_for_network(images, DMPs)


# for i in range(int(len(indexes))):
#     Trainer.show_dmp(images[i],trajectories[i],DMPs[i],indexes[i])


#learn
print()
print('Starting training')
print(" + Training with parameters: ")
print("   - Samples of data", len(input_data))
print("   - Epochs: ", epochs)
print("   - Learning rate: ", learning_rate)
print("   - Momentum: ", momentum)
print("   - Decay: ", decay)
print("   - Batch size: ", batch_size)


model = EncoderDecoderNet(layer_sizes, conv)
model.scale = scale
np.save(scale_file, scale)
#inicalizacija
if load:
    print(' + Loaded parameters from file: ', parameters_file)
    if load_from_cuda:
        model.load_state_dict(torch.load(parameters_file, map_location=lambda storage, loc: storage)) # loading parameters
    else:


else:
    print(' + Initialized paramters randomly')
    for p in list(model.parameters()):
        torch.nn.init.normal(p,0,0.1)

if cuda:
    model.cuda()
    input_data = input_data.cuda()
    output_data = output_data.cuda()

model.train(input_data, output_data, batch_size, epochs, learning_rate, momentum, 10, plot, decay)
print('Training finished\n')

parameters = list(model.parameters())

torch.save(model.state_dict(), parameters_file) # saving parameters

#Trainer.show_network_output(model, 1, images, trajectories,DMPs, N, sampling_time, indexes)
if plot:
    for i in np.random.rand(10)*(data-s_data):
        Trainer.show_network_output(model, int(i), images, trajectories,DMPs, N, sampling_time, cuda = cuda)

    Trainer.show_network_output(model, -1, test[:5], None, None, N, sampling_time, cuda = cuda)