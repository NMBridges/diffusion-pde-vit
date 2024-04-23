from enum import Enum
from torch import device as dev
from torch import Tensor, norm, exp, sin
import numpy as np


device = dev('cuda:1')


class DataType(Enum):
    mnist = 0
    images = 1
    heat_1d = 2
    heat_2d = 3


class ConvType(Enum):
    Conv2d = 0
    Conv3d = 1


class Arguments:
    def __init__(self):
        self.T_max = 3.0
        self.num_T = 96
        self.num_X = 32
        self.num_Y = 32
        self.dx = 1 / (num_X - 1) # spatial grid tile length
        self.dx2 = dx**2
        self.dy = 1 / (num_Y - 1) # spatial grid tile length
        self.dy2 = dy**2
        self.dt = 1 / (num_T - 1) # timestep size
        self.one_d_grid_size = num_X * num_T
        self.two_d_grid_size = num_X * num_Y * num_T
        self.loss_lambda = 0.0003


channel_map = {
    ConvType.Conv2d: {
        DataType.mnist: 1,
        DataType.images: 3,
        DataType.heat_1d: 32, # for the 32xYxX discretization for 32 timesteps from t=0 to t=1. this will produce bad results
        DataType.heat_2d: 32, # note: this will produce bad results
    },
    ConvType.Conv3d: {
        DataType.mnist: 1,
        DataType.images: 1,
        DataType.heat_1d: 1, # for the 32xYxX discretization for 32 timesteps from t=0 to t=1
        DataType.heat_2d: 1, # for the 32xYxX discretization for 32 timesteps from t=0 to t=1
    },
}


conv_map = { # for running convolutions, what should the dimensions be
    ConvType.Conv2d: {
        DataType.mnist: {
            'kernel': (3,3),
            'stride': (1,1),
            'padding': (1,1),
            'dilation': (1,1),
            'down_up_kernel_and_stride': (2,2)
        },
        DataType.images: {
            'kernel': (3,3),
            'stride': (1,1),
            'padding': (1,1),
            'dilation': (1,1),
            'down_up_kernel_and_stride': (2,2)
        },
        DataType.heat_1d: {
            'kernel': (1,3),
            'stride': (1,1),
            'padding': (0,1),
            'dilation': (1,1),
            'down_up_kernel_and_stride': (1,2)
        },
        DataType.heat_2d: {
            'kernel': (3,3),
            'stride': (1,1),
            'padding': (1,1),
            'dilation': (1,1),
            'down_up_kernel_and_stride': (2,2)
        }
    },
    ConvType.Conv3d: {
        DataType.mnist: {
            'kernel': (1,3,3),
            'stride': (1,1,1),
            'padding': (0,1,1),
            'dilation': (1,1,1),
            'down_up_kernel_and_stride': (1,2,2)
        },
        DataType.images: {
            'kernel': (3,3,3),
            'stride': (1,1,1),
            'padding': (1,1,1),
            'dilation': (1,1,1),
            'down_up_kernel_and_stride': (2,2,2)
        },
        DataType.heat_1d: {
            'kernel': (3,1,3),
            'stride': (1,1,1),
            'padding': (1,0,1),
            'dilation': (1,1,1),
            'down_up_kernel_and_stride': (2,1,2)
        },
        DataType.heat_2d: {
            'kernel': (3,3,3),
            'stride': (1,1,1),
            'padding': (1,1,1),
            'dilation': (1,1,1),
            'down_up_kernel_and_stride': (2,2,2)
        }
    }
}


label_dim_map = {
    DataType.mnist: 1,
    DataType.images: 1,
    DataType.heat_1d: 3,
    DataType.heat_2d: 3
}


num_classes_map = {
    DataType.mnist: 10, # discrete labels
    DataType.images: 10,
    DataType.heat_1d: None, # continuous labels
    DataType.heat_2d: None
}


cifar_label_map = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}


def get_label_name(data_type, label):
    if data_type == DataType.images:
        return cifar_label_map[int(label.item())]
    else:
        return label


T_max = 3.0
Y_max = 1.0
X_max = 1.0
num_T = 96
num_X = 32
num_Y = 32
dx = 1 / (num_X - 1) # spatial grid tile length
dx2 = dx**2
dy = 1 / (num_Y - 1) # spatial grid tile length
dy2 = dy**2
dt = 1 / (num_T - 1) # timestep size
one_d_grid_size = num_X * num_T
two_d_grid_size = num_X * num_Y * num_T
loss_lambda = 0.0003

def heat_1d_loss(pred_x_i1, label, t, conv_type, T):
    # TODO: fix this, this is using t for timesteps rather than one per batch item
    if conv_type == ConvType.Conv2d:
        time_deriv = (pred_x_i1[:,1:,:,:] - pred_x_i1[:,:-1,:,:]) / dt
        second_spat_deriv_x = (pred_x_i1[:,:,:,2:] - 2 * pred_x_i1[:,:,:,1:-1] + pred_x_i1[:,:,:,:-2]) / dx2
        f = label
        resid = time_deriv[:,:,:,1:-1] - second_spat_deriv_x[:,:-1,:,:] - f # physics loss on terms that have derivatives (loss doesn't apply to some edge values)
        heat_loss = norm((T - t[1:-1]) / T * resid) / one_d_grid_size
        return loss_lambda * heat_loss
    else:
        time_deriv = (pred_x_i1[:,:,1:,:,:] - pred_x_i1[:,:,:-1,:,:]) / dt
        second_spat_deriv_x = (pred_x_i1[:,:,:,:,2:] - 2 * pred_x_i1[:,:,:,:,1:-1] + pred_x_i1[:,:,:,:,:-2]) / dx2
        f = label
        resid = time_deriv[:,:,:,:,1:-1] - second_spat_deriv_x[:,:,:-1,:,:] - f # physics loss on terms that have derivatives (loss doesn't apply to some edge values)
        heat_loss = norm((T - t[1:-1]) / T * resid) / one_d_grid_size
        return loss_lambda * heat_loss

t_tens = Tensor(np.arange(num_T))[None,:,None,None].to(device) / (num_T - 1) * T_max
y_tens = Tensor(np.arange(num_Y))[None,None,:,None].to(device) / (num_Y - 1) * Y_max
x_tens = Tensor(np.arange(num_X))[None,None,None,:].to(device) / (num_X - 1) * X_max

def eval_f(params):
    return 3 * sin(3.1415 * x_tens) * sin(3.1415 * y_tens) * exp(2-params[:,0,None,None,None] * t_tens) * (exp(-5 * (x_tens - params[:,1,None,None,None])**2) + exp(-5 * (y_tens - params[:,2,None,None,None])**2))

def heat_2d_loss(pred_x_i1, label, t, conv_type, T):
    if conv_type == ConvType.Conv2d:
        time_deriv = (pred_x_i1[:,1:,:,:] - pred_x_i1[:,:-1,:,:]) / dt
        second_spat_deriv_x = (pred_x_i1[:,:,:,2:] - 2 * pred_x_i1[:,:,:,1:-1] + pred_x_i1[:,:,:,:-2]) / dx2
        second_spat_deriv_y = (pred_x_i1[:,:,2:,:] - 2 * pred_x_i1[:,:,1:-1,:] + pred_x_i1[:,:,:-2,:]) / dy2
        f = eval_f(label)[:,:-1,1:-1,1:-1]
        resid = time_deriv[:,:,1:-1,1:-1] - second_spat_deriv_x[:,:-1,1:-1,:] - second_spat_deriv_y[:,:-1,:,1:-1] - f # physics loss on terms that have derivatives (loss doesn't apply to some edge values)
        heat_loss = norm(((T - t) / T)[:,None,None] * resid) / two_d_grid_size
        return loss_lambda * heat_loss
    else:
        time_deriv = (pred_x_i1[:,:,1:,:,:] - pred_x_i1[:,:,:-1,:,:]) / dt
        second_spat_deriv_x = (pred_x_i1[:,:,:,:,2:] - 2 * pred_x_i1[:,:,:,:,1:-1] + pred_x_i1[:,:,:,:,:-2]) / dx2
        second_spat_deriv_y = (pred_x_i1[:,:,:,2:,:] - 2 * pred_x_i1[:,:,:,1:-1,:] + pred_x_i1[:,:,:,:-2,:]) / dy2
        f = eval_f(label)[:,None,:-1,1:-1,1:-1]
        resid = time_deriv[:,:,:,1:-1,1:-1] - second_spat_deriv_x[:,:,:-1,1:-1,:] - second_spat_deriv_y[:,:,:-1,:,1:-1] - f # physics loss on terms that have derivatives (loss doesn't apply to some edge values)
        heat_loss = norm(((T - t) / T)[:,None,None,None] * resid) / two_d_grid_size
        return loss_lambda * heat_loss