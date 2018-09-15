import os
import re
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from imednet.data.smnist_loader import Mapping
from imednet.models.mnist_cnn import Net as MNISTNet
from imednet.utils.dmp_layer import DMPIntegrator, DMPParameters


def load_model(model_path):
    # Read network description file
    with open(os.path.join(model_path, 'network_description.txt')) as f:
        network_description_str = f.read()

    # Get the model class from the network description and
    # dynamically import it
    model_module_class_str = re.search('Model: (.+?)\n', network_description_str).group(1)
    model_module_str = os.path.splitext(model_module_class_str)[0]
    model_class_str = os.path.splitext(model_module_class_str)[1][1:]
    model_module = importlib.import_module(model_module_str)
    model_class = getattr(model_module, model_class_str)

    # Get the pre-trained CNN model load path from the network description
    if model_class_str == 'CNNEncoderDecoderNet' or model_class_str == 'FullCNNEncoderDecoderNet':
        pretrained_cnn_model_path = re.search('Pe-trained CNN model load path: (.+?)\n', network_description_str).group(1)
    elif model_class_str == 'STIMEDNet' or model_class_str == 'FullSTIMEDNet':
        pretrained_imednet_model_path = re.search('Pre-trained IMEDNet model load path: (.+?)\n', network_description_str).group(1)

    # Load layer sizes
    layer_sizes = np.load(os.path.join(model_path, 'layer_sizes.npy')).tolist()

    # Get the image size from the network description
    try:
        image_size_str = re.search('Image size: \[(.+?)\]\n', network_description_str).group(1)
        image_size = [int(s) for s in image_size_str.split(',')]
    except:
        pass

    # Load scaling
    try:
        scaling = Mapping()
        scaling.x_max = np.load(os.path.join(model_path, 'scale_x_max.npy'))
        scaling.x_min = np.load(os.path.join(model_path, 'scale_x_min.npy'))
        scaling.y_max = np.float(np.load(os.path.join(model_path, 'scale_y_max.npy')))
        scaling.y_min = np.float(np.load(os.path.join(model_path, 'scale_y_min.npy')))
    except:
        scaling = np.load(os.path.join(model_path, 'scale.npy'))

    # Load the model
    if model_class_str == 'CNNEncoderDecoderNet' or model_class_str == 'FullCNNEncoderDecoderNet':
        model = model_class(pretrained_cnn_model_path=pretrained_cnn_model_path,
                            layer_sizes=layer_sizes,
                            scale=scaling)
        model.cuda()
    elif model_class_str == 'STIMEDNet' or model_class_str == 'FullSTIMEDNet':
        try:
            model = model_class(pretrained_imednet_model_path=pretrained_imednet_model_path,
                                scale=scaling,
                                image_size=image_size)
        except:
            try:
                model = model_class(pretrained_imednet_model_path=pretrained_imednet_model_path,
                                    scale=scaling)
            except:
                raise
        model.cuda()
    else:
        model = model_class(layer_sizes, None, scaling)

    # Load the model state parameters
    state = torch.load(os.path.join(model_path, 'net_parameters'))
    model.load_state_dict(state)

    return model


class TrainingParameters():
    # Before
    epochs = 1000
    batch_size = 32
    val_fail = 5
    time = -1

    cuda = True
    device = 0

    validation_interval = 1
    log_interval = 1
    test_interval = 1

    training_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15
    data_samples = 0

    # After
    real_epochs = 0
    min_train_loss = -1
    min_val_loss = -1
    min_test_loss = -1
    elapsed_time = -1
    val_count = -1
    stop_criterion = ""
    min_grad = -1

    def __init__(self):
        pass

    def write_out(self):
        learn_info = "\n Setting parameters for learning:\n" + "   - Samples of data: " + str(self.data_samples) + \
                     "\n   - Epochs: " + str(self.epochs) + \
                     "\n   - Batch size: " + str(self.batch_size) \
                     + "\n   - training ratio: " + str(self.training_ratio) + "\n   - validation ratio: " + \
                     str(self.validation_ratio) + "\n   - test ratio: " + str(self.test_ratio)+\
                     "\n     -   validation_interval: " + str(self.validation_interval)+ \
                     "\n     -  test_interval: " + str(self.test_interval)+ \
                     "\n     -   log_interval: " + str(self.log_interval) +\
                     "\n     -   cuda = " + str(self.cuda)+ \
                     "\n     -  Validation fail: " + str(self.val_fail)

        return learn_info

    def write_out_after(self):
        learn_info = "\n Learning finished with this parameters:\n" + "   - Number of epochs: " + str(self.real_epochs) + \
                     "\n   - Last train loss: " + str(self.min_train_loss) + \
                     "\n   - Last validation loss: " + str(self.min_val_loss) + \
                     "\n   Last test loss: " + str(self.min_test_loss) + \
                     "\n   - Elapsed time: " + str(self.elapsed_time) + \
                     "\n   - last validation count: " + str(self.val_count) + \
                     "\n     -   Stop criterion: " + str(self.stop_criterion) + \
                     "\n     -  Minimal gradient: " + str(self.min_grad)

        return learn_info


class EncoderDecoderNet(torch.nn.Module):
    def __init__(self,
                 layer_sizes=[784, 200, 50],
                 conv=None,
                 scale=None):
        """
        Creates a custom Network

        layer_sizes -> list containing layer inputs/ouptuts
                       (minimum length = 3)
            example:
                layer_sizes = [784,500,200,50]
                input_layer -> torch.nn.Linear(784,500)
                middle_layers -> [torch.nn.Linear(500,200)]
                output_layer -> torch.nn.Linear(200,50)
        """
        super(EncoderDecoderNet, self).__init__()
        self.conv = conv
        if self.conv:
            self.imageSize = int(np.sqrt(layer_sizes[0]))
            self.convSize = (self.imageSize - conv[1] + 1)**2 * conv[0]
            self.firstLayer = torch.nn.Conv2d(1, conv[0], conv[1])
            self.input_layer = torch.nn.Linear(self.convSize, layer_sizes[1])

        else:
            self.input_layer = torch.nn.Linear(layer_sizes[0], layer_sizes[1])
        self.middle_layers = []
        for i in range(1, len(layer_sizes) - 2):
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.middle_layers.append(layer)
            self.add_module("middle_layer_" + str(i), layer)
        self.output_layer = torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.scale = scale
        self.loss = 0

    def forward(self, x):
        """
        Defines the layers connections

        forward(x) -> result of forward propagation through network
        x -> input to the Network
        """
        #activation_fn = torch.nn.ReLU6()
        activation_fn = torch.nn.Tanh()

        if self.conv:
            x = x.view(-1, 1, self.imageSize, self.imageSize)
            x = self.firstLayer(x)
            x = x.view(-1, self.convSize)

        x = activation_fn(self.input_layer(x))
        for layer in self.middle_layers:
            x = activation_fn(layer(x))
        output = self.output_layer(x)
        return output

    def isCuda(self):
        return self.input_layer.weight.is_cuda


class DMPEncoderDecoderNet(torch.nn.Module):
    def __init__(self,
                 layer_sizes=[784, 200, 50],
                 conv=None,
                 scale=None):
        """
        Creates a custom Network

        layer_sizes -> list containing layer inputs/ouptuts
                       (minimum length = 3)
            example:
                layer_sizes = [784,500,200,50]
                input_layer -> torch.nn.Linear(784,500)
                middle_layers -> [torch.nn.Linear(500,200)]
                output_layer -> torch.nn.Linear(200,50)
        """
        super(DMPEncoderDecoderNet, self).__init__()
        self.conv = conv
        if self.conv:
            self.imageSize = int(np.sqrt(layer_sizes[0]))
            self.convSize = (self.imageSize - conv[1] + 1)**2 * conv[0]
            self.firstLayer = torch.nn.Conv2d(1, conv[0], conv[1])
            self.input_layer = torch.nn.Linear(self.convSize, layer_sizes[1])

        else:
            self.input_layer = torch.nn.Linear(layer_sizes[0], layer_sizes[1])
        self.middle_layers = []
        for i in range(1, len(layer_sizes) - 2):
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.middle_layers.append(layer)
            self.add_module("middle_layer_" + str(i), layer)
        self.output_layer = torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.scale = scale
        self.loss = 0
        self.DMPparam = DMPParameters(25, 3, 0.01, 2, scale)
        self.func = DMPIntegrator()
        '''self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('scale_t', self.DMPparam.scale_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)'''

    def forward(self, x):
        """
        Defines the layers connections

        forward(x) -> result of forward propagation through network
        x -> input to the Network
        """
        # activation_fn = torch.nn.ReLU6()
        activation_fn = torch.nn.Tanh()

        if self.conv:
            x = x.view(-1, 1, self.imageSize, self.imageSize)
            x = self.firstLayer(x)
            x = x.view(-1, self.convSize)

        x = activation_fn(self.input_layer(x))
        for layer in self.middle_layers:
            x = activation_fn(layer(x))
        x = self.output_layer(x)
        output = self.func.apply(x, self.DMPp, self.param_grad, self.scale_t)
        return output

    def isCuda(self):
        return self.input_layer.weight.is_cuda


class CNNEncoderDecoderNet(torch.nn.Module):
    def __init__(self,
                 pretrained_cnn_model_path=None,
                 layer_sizes=[784, 200, 50],
                 scale=None):
        """
        Creates a convolutional image-to-motion encoder-decoder
        (CIMEDNet) network without DMP integration.

        layer_sizes -> list containing layer inputs/ouptuts
                       (minimum length = 3)

            example:
                layer_sizes = [784,500,200,50]
                input_layer -> torch.nn.Linear(784,500)
                middle_layers -> [torch.nn.Linear(500,200)]
                output_layer -> torch.nn.Linear(200,50)
        """
        super(CNNEncoderDecoderNet, self).__init__()

        # Initialize MNIST CNN model
        self.cnn_model = MNISTNet()

        # Load the pretrained weights
        if pretrained_cnn_model_path:
            try:
                self.cnn_model.load_state_dict(torch.load(pretrained_cnn_model_path))
            except:
                self.cnn_model.load_state_dict(torch.load(os.path.join('../', pretrained_cnn_model_path)))

        # Chop off the FC layers (2 of them) + dropout layer,
        # leaving just the two conv layers.
        self.cnn_model = torch.nn.Sequential(*list(self.cnn_model.modules())[1:-3])
        # Get the output size of the last conv layer
        self.image_size = int(np.sqrt(layer_sizes[0]))
        self.conv1 = self.cnn_model[0].state_dict()['weight'].size()
        self.conv1_W = self.image_size - self.conv1[2] + 1
        self.conv1_size = (self.conv1_W)**2 * self.conv1[0]
        self.conv2 = self.cnn_model[1].state_dict()['weight'].size()
        self.conv2_W = self.conv1_W - self.conv2[2] + 1
        self.conv2_size = (self.conv2_W)**2 * self.conv2[0]

        # Set up the input layer for encoder-decoder part
        self.input_layer = torch.nn.Linear(self.conv2_size, layer_sizes[1])

        self.middle_layers = []
        for i in range(1, len(layer_sizes) - 2):
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.middle_layers.append(layer)
            self.add_module("middle_layer_" + str(i), layer)
        self.output_layer = torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.scale = scale
        self.loss = 0

    def forward(self, x):
        """
        Defines the layers connections

        forward(x) -> result of forward propagation through network
        x -> input to the Network
        """
        # activation_fn = torch.nn.ReLU6()
        activation_fn = torch.nn.Tanh()

        x = x.view(-1, 1, self.image_size, self.image_size)

        # Run the input through the pretrained CNN
        x = self.cnn_model(x)
        x = x.view(-1, self.conv2_size)

        x = activation_fn(self.input_layer(x))

        for layer in self.middle_layers:
            x = activation_fn(layer(x))

        output = self.output_layer(x)

        return output

    def isCuda(self):
        return self.input_layer.weight.is_cuda


class FullCNNEncoderDecoderNet(torch.nn.Module):
    def __init__(self,
                 pretrained_cnn_model_path=None,
                 layer_sizes=[784, 200, 50],
                 scale=None):
        """
        Creates a full convolutional image-to-motion encoder-decoder
        (CIMEDNet) network with DMP integration.

        layer_sizes -> list containing layer inputs/ouptuts
                       (minimum length = 3)

            example:
                layer_sizes = [784,500,200,50]
                input_layer -> torch.nn.Linear(784,500)
                middle_layers -> [torch.nn.Linear(500,200)]
                output_layer -> torch.nn.Linear(200,50)
        """
        super(FullCNNEncoderDecoderNet, self).__init__()

        # Initialize MNIST CNN model
        self.cnn_model = MNISTNet()

        # Load the pretrained weights
        if pretrained_cnn_model_path:
            try:
                self.cnn_model.load_state_dict(torch.load(pretrained_cnn_model_path))
            except:
                self.cnn_model.load_state_dict(torch.load(os.path.join('../', pretrained_cnn_model_path)))

        # Chop off the FC layers (2 of them) + dropout layer,
        # leaving just the two conv layers.
        self.cnn_model = torch.nn.Sequential(*list(self.cnn_model.modules())[1:-3])
        # Get the output size of the last conv layer
        self.image_size = int(np.sqrt(layer_sizes[0]))
        self.conv1 = self.cnn_model[0].state_dict()['weight'].size()
        self.conv1_W = self.image_size - self.conv1[2] + 1
        self.conv1_size = (self.conv1_W)**2 * self.conv1[0]
        self.conv2 = self.cnn_model[1].state_dict()['weight'].size()
        self.conv2_W = self.conv1_W - self.conv2[2] + 1
        self.conv2_size = (self.conv2_W)**2 * self.conv2[0]

        # Set up the input layer for encoder-decoder part
        self.input_layer = torch.nn.Linear(self.conv2_size, layer_sizes[1])

        self.middle_layers = []
        for i in range(1, len(layer_sizes) - 2):
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.middle_layers.append(layer)
            self.add_module("middle_layer_" + str(i), layer)
        self.output_layer = torch.nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.scale = scale
        self.loss = 0

        self.dmp_params = DMPParameters(25, 3, 0.01, 2, scale)
        self.dmp_integrator = DMPIntegrator()

        self.register_buffer('dmp_p', self.dmp_params.data_tensor)
        self.register_buffer('scale_t', self.dmp_params.scale_tensor)
        self.register_buffer('param_grad', self.dmp_params.grad_tensor)

        if self.isCuda():
            self.dmp_p.cuda()
            self.scale_t.cuda()
            self.param_grad.cuda()

    def forward(self, x):
        """
        Defines the layers connections

        forward(x) -> result of forward propagation through network
        x -> input to the Network
        """
        # activation_fn = torch.nn.ReLU6()
        activation_fn = torch.nn.Tanh()

        x = x.view(-1, 1, self.image_size, self.image_size)

        # Run the input through the pretrained CNN
        x = self.cnn_model(x)
        x = x.view(-1, self.conv2_size)

        x = activation_fn(self.input_layer(x))

        for layer in self.middle_layers:
            x = activation_fn(layer(x))

        x = self.output_layer(x)

        # Integrate the DMPs to calculate the predicted output trajectories
        output = self.dmp_integrator.apply(x, self.dmp_p, self.param_grad, self.scale_t)

        return output

    def isCuda(self):
        return self.input_layer.weight.is_cuda


class STIMEDNet(torch.nn.Module):
    def __init__(self,
                 pretrained_imednet_model_path=None,
                 image_size=[40, 40, 1],
                 scale=None):
        """
        image_size: [H, W, C]
        """
        super(STIMEDNet, self).__init__()

        # Save the image size
        self.image_size = image_size

        # Load the IMEDNet model
        if pretrained_imednet_model_path:
            self.imednet_model = load_model(pretrained_imednet_model_path)
        else:
            self.imednet_model = CNNEncoderDecoderNet()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Calculate localizer output size by running a
        # dummy image through the localizer.
        self.localizer_out_size = np.prod(np.asarray(
            self.localization(torch.zeros(1,
                                          self.image_size[2],
                                          self.image_size[0],
                                          self.image_size[1])).shape[1:])) 

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.localizer_out_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)

        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.scale = scale
        self.loss = 0

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.localizer_out_size)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        """
        """
        # Resize the input
        x = x.view(-1, self.image_size[2], self.image_size[0], self.image_size[1])

        # Transform the input
        x = self.stn(x)

        # Run the transformed input through the pretrained IMEDNet model
        output = self.imednet_model(x)

        return output

    def isCuda(self):
        return self.imednet_model.input_layer.weight.is_cuda


class FullSTIMEDNet(torch.nn.Module):
    def __init__(self,
                 pretrained_imednet_model_path,
                 image_size=[40, 40, 1],
                 grid_size=None,
                 scale=None):
        """
        image_size: [H, W, C]
        grid_size: [H, W, C]
        """
        super(FullSTIMEDNet, self).__init__()

        # Save the image size input arg
        self.image_size = image_size

        # Load the IMEDNet model
        if pretrained_imednet_model_path:
            self.imednet_model = load_model(pretrained_imednet_model_path)
        else:
            self.imednet_model = CNNEncoderDecoderNet()

        # Save the grid size input arg or infer it
        # from the IMEDNet mode
        # TODO: Fix all of these image size recording issues!
        if grid_size:
            self.grid_size = grid_size
        else:
            self.grid_size = [self.imednet_model.image_size,
                              self.imednet_model.image_size,
                              1]

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Calculate localizer output size by running a
        # dummy image through it.
        dummy_image = torch.zeros(1,
                                  self.image_size[2],
                                  self.image_size[0],
                                  self.image_size[1])
        dummy_localizer_out = self.localization(dummy_image)
        self.localizer_out_shape = dummy_localizer_out.shape
        self.localizer_out_size = np.prod(np.asarray(dummy_localizer_out.shape[1:])) 

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.localizer_out_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Regressor for the 3 * 3 affine transform matrix
        self.fc_T = nn.Sequential(
            nn.Linear(2*3, 32),
            nn.ReLU(True),
            # nn.Dropout2d(),
            nn.Linear(32, 32),
            nn.ReLU(True),
            # nn.Dropout2d(),
            nn.Linear(32, 3 * 3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Initialize the weights/bias with identity transformation
        self.fc_T[4].weight.data.zero_()
        self.fc_T[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float))

        self.scale = scale
        self.loss = 0

        self.dmp_params = DMPParameters(25, 3, 0.01, 2, scale)
        self.dmp_integrator = DMPIntegrator()

        self.register_buffer('dmp_p', self.dmp_params.data_tensor)
        self.register_buffer('scale_t', self.dmp_params.scale_tensor)
        self.register_buffer('param_grad', self.dmp_params.grad_tensor)

        if self.isCuda():
            self.dmp_p.cuda()
            self.scale_t.cuda()
            self.param_grad.cuda()

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.localizer_out_size)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid_size = torch.Size([x.shape[0], self.grid_size[2], self.grid_size[0], self.grid_size[1]])
        grid = F.affine_grid(theta, grid_size)
        x = F.grid_sample(x, grid)

        return x, theta

    # Motion transformer network forward function
    def mtn(self, x, theta):
        # 1. Integrate the DMPs to calculate the predicted canonical motion trajectories.
        x = self.dmp_integrator.apply(x, self.dmp_p, self.param_grad, self.scale_t)

        # 2. Reshape the DMP integrator output into vector trajectories.
        x_traj_vectors = x.view(int(x.shape[0]/2), 2, x.shape[1]).transpose(0,1)
        x_traj_vectors_with_ones = torch.cat((x_traj_vectors, torch.ones(1,int(x.shape[0]/2), x.shape[1]).cuda()), 0).cuda()

        # 3. Do the transformations.
        transformed_x_traj_vectors = torch.einsum('nij,jnm->nim', [theta, x_traj_vectors_with_ones])

        # 4. Reshape the transformed trajectories to conform with the expected output format.
        output = transformed_x_traj_vectors.view(x.shape[0], x.shape[1])

        return output

    def forward(self, x):
        """
        """
        # Resize the input image
        x = x.view(-1, self.image_size[2], self.image_size[0], self.image_size[1])

        # Image Transformer:
        # Rectify the input images using the spatial transformer network (STN)
        # such that the attended objects are transformed to their canonical forms.
        x, theta = self.stn(x)

        # Image-to-Motion Encoder-Decoder:
        # Run the rectified images through the pre-trained IMEDNet model in
        # order to predict DMP parameters for the canonical associated motion
        # trajectories of the attended object.
        x = self.imednet_model(x)

        # Motion Transformer:
        # Integrate the rectified DMP and transform the resulting trajectory
        # using the theta transformation from the spatial transformer.
        output = self.mtn(x, theta)

        return output

    def isCuda(self):
        return self.imednet_model.input_layer.weight.is_cuda
