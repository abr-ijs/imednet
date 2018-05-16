
import torch
from matLoader import matLoader
import numpy as np
from torch.autograd import Variable
from trainer import Trainer
import csv

#Load model and parameters
directory_path = '/home/rpahic/Documents/Neural_networks/'
directory_name = 'NN ' + '2018-04-25 13:36:32.095726'

model = torch.load(directory_path+directory_name+'/model.pt')

state = torch.load(directory_path+directory_name+'/net_parameters')

model.load_state_dict(state)





dateset_name = 'slike_780.4251'

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

test_input = Variable(torch.from_numpy(np.array(images))).float()
test_output = Variable(torch.from_numpy(np.array(outputs))).float()
nn_output = model(test_input)

criterion = torch.nn.MSELoss(size_average=True)

loss = criterion(nn_output, test_output[:,1:])

letent_out=encoder(test_input)

np.savetxt('output_data.csv',outputs)
np.savetxt('latent_space.csv', letent_out.detach().numpy())



print(loss.item())
for number in range(0,10):

    input_image = Variable(torch.from_numpy(np.array(images[40+number]))).float()
    real_output = Variable(torch.from_numpy(np.array(outputs[40+number]))).float()



    nn_output1 = model(input_image)
    nn_output = decoder(encoder(input_image))






    output_dmp = torch.cat((real_output[0:1], nn_output), 0)

    trainer = Trainer()

    dmp = trainer.createDMP(real_output, model.scale, 0.01, 25, cuda=False)
    dmp_v = trainer.createDMP(output_dmp, model.scale, 0.01, 25, cuda=False)

    dmp.joint()
    dmp_v.joint()
    #dmp.plot_j()
    mat = trainer.show_dmp(input_image.data.numpy(), original_trj[40+number], dmp_v, plot=True)

