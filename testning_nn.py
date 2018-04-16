
import torch
from matLoader import matLoader
import numpy as np
from torch.autograd import Variable
from trainer import Trainer

#Load model and parameters
directory_path = '/home/rpahic/Documents/Neural_networks/'
directory_name = 'NN ' + '2018-04-16 10:03:11.290010'

model = torch.load(directory_path+directory_name+'/model.pt')

state = torch.load(directory_path+directory_name+'/net_parameters')

model.load_state_dict(state)


dateset_name = 'slike_780.4251'

images, outputs, scale, original_trj = matLoader.loadData(dateset_name, load_original_trajectories=True)

test_input = Variable(torch.from_numpy(np.array(images))).float()
test_output = Variable(torch.from_numpy(np.array(outputs))).float()
nn_output = model(test_input)

criterion = torch.nn.MSELoss(size_average=True)

loss = criterion(nn_output, test_output[:,5:])

for grup in state:
    mean = torch.mean(state[grup])
    max = torch.max(state[grup])
    min = torch.min(state[grup])
    var = torch.var(state[grup])
    print(grup)
    print('mean = ' + str(mean))
    print('max = ' + str(max))
    print('min = ' + str(min))
    print('var = ' + str(var))


print(loss.data[0])
for number in range(0,10):

    input_image = Variable(torch.from_numpy(np.array(images[40+number]))).float()
    real_output = Variable(torch.from_numpy(np.array(outputs[40+number]))).float()

    nn_output = model(input_image)



    output_dmp = torch.cat((real_output[0:5], nn_output), 0)

    trainer = Trainer()

    dmp = trainer.createDMP(real_output, model.scale, 0.01, 25, cuda=False)
    dmp_v = trainer.createDMP(output_dmp, model.scale, 0.01, 25, cuda=False)

    dmp.joint()
    dmp_v.joint()
    #dmp.plot_j()
    mat = trainer.show_dmp(input_image.data.numpy(), original_trj[number], dmp_v, plot=True)