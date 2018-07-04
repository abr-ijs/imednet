
import torch
from matLoader import matLoader
import numpy as np
from torch.autograd import Variable
from trainer import Trainer

import csv

import torch.nn as nn
import DMP_layer



#Load model and parameters
directory_path = '/home/rpahic/Documents/Neural_networks/'
directory_name = 'NN ' + '2018-06-14 13:23:28.813445'

model = torch.load(directory_path+directory_name+'/model.pt')

state = torch.load(directory_path+directory_name+'/net_parameters')

model.load_state_dict(state)



import os
cwd = os.getcwd()
data_path=os.path.dirname(cwd)

dateset_name = data_path + '/' + 'slike_780.4251'

model_test =torch.nn.Sequential(*list(model.children())[:])


encoder = torch.nn.Sequential()

for i in range(0,6):
    encoder.add_module( str(i*2),model_test[i])
    encoder.add_module(str(i*2+1), *[torch.nn.Tanh()])



decoder = torch.nn.Sequential()

for i in range(6,7):
    decoder.add_module( str(i*2),model_test[i])
    decoder.add_module(str(i*2+1), *[torch.nn.Tanh()])

decoder.add_module( str((i+1)*2),model_test[i+1])


images, outputs, scale, original_trj = matLoader.loadData(dateset_name, load_original_trajectories=True)

original_trj_e = []
for i in range(0,images.shape[0]):
    c,c1,c2 = zip(*original_trj[i])
    original_trj_e.append(c)
    original_trj_e.append(c1)



test_input = Variable(torch.from_numpy(np.array(images))).float().cuda()
test_output = Variable(torch.from_numpy(np.array(original_trj_e))).float().cuda()

model=model.cuda()




nn_output = model(test_input)




criterion = torch.nn.MSELoss(size_average=True)

loss = criterion(nn_output, test_output[:,:])

letent_out = encoder(test_input)

#np.savetxt('output_data.csv',outputs)
#np.savetxt('latent_space.csv', letent_out.detach().numpy())

input_image = Variable(torch.from_numpy(np.array(images[40:50]))).float().cuda()
real_output = Variable(torch.from_numpy(np.array(outputs[40:50]))).float().cuda()
nn_output1 = model(input_image)
#nn_output = decoder(encoder(input_image))
print(loss.item())

for number in range(0,10):




    #output_dmp = torch.cat((real_output[0:1], nn_output), 0)

    trainer = Trainer()

    dmp = trainer.createDMP(real_output.cpu()[number], model.scale, 0.01, 25, cuda=False)
    #dmp_v = trainer.createDMP(output_dmp, model.scale, 0.01, 25, cuda=False)

    dmp.joint()
    #dmp_v.joint()
    #dmp.plot_j()

    #mat = trainer.show_dmp(input_image.cpu().data.numpy()[number], original_trj[40+number], dmp, plot=True)
    #'''
    import matplotlib.pyplot as plt
    fig = plt.figure()

    plt.imshow((np.reshape(input_image.cpu().data.numpy()[number], (40, 40))), cmap='gray', extent=[0, 40, 40, 0])

    plt.plot(original_trj_e[80+2*number], original_trj_e[80+2*number+1], '--r', label='dmp')

    plt.plot(nn_output1.cpu().detach().numpy()[2*number,:], nn_output1.cpu().detach().numpy()[2*number+1,:], '-g', label='trajectory')
    plt.show()
    plt.legend()
    plt.xlim([0, 40])
    plt.ylim([40, 0])

#    fig.canvas.draw()
    #matrix = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # if save != -1:
    #    plt.savefig("images/" + str(save) + ".pdf")
    #    plt.close(fig)
    # else:


#'''