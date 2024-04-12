from torch import Tensor, load
from src.diff_utils import DataType, ConvType, device
import numpy as np
from keras.datasets import mnist
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def load_dataset(args : dict):
    '''Takes in arguments about the training process, returns dataset components
        as well as data loaders for the training and testing process.'''
    data_type = args['data_type']
    conv_type = args['conv_type']
    batch_size = args['batch_size']
    num_T = args['num_T']
    num_Y = args['num_Y']
    num_X = args['num_X']

    dataset_dict = {}

    if data_type == DataType.mnist:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.pad(x_train, pad_width=2)[2:-2] # So images are 32x32 and UNet can do more invertible max poolings
        x_test = np.pad(x_test, pad_width=2)[2:-2]
        x_train = x_train / 127.5 - 1.0
        x_test = x_test / 127.5 - 1.0

        if conv_type == ConvType.Conv3d:
            x_train = x_train[:,None,None,:,:]
            x_test = x_test[:,None,None,:,:]

        train_dataset = TensorDataset(Tensor(x_train).to(device), Tensor(y_train[:,None]).to(device))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TensorDataset(Tensor(x_test).to(device), Tensor(y_test[:,None]).to(device))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        dataset_dict['train_dataset'] = train_dataset
        dataset_dict['train_dataloader'] = train_dataloader
        dataset_dict['test_dataset'] = test_dataset
        dataset_dict['test_dataloader'] = test_dataloader

    elif data_type == DataType.images:
        dataset = load('patches.pkl')
        x_train = dataset[:(len(dataset) * 4) // 5] * 2 - 1.0
        x_test = dataset[(len(dataset) * 4) // 5:] * 2 - 1.0

        if conv_type == ConvType.Conv3d:
            x_train = x_train[None,:,:,:]
            x_test = x_test[None,:,:,:]

        train_dataset = TensorDataset(Tensor(x_train).to(device))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TensorDataset(Tensor(x_test).to(device))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        dataset_dict['train_dataset'] = train_dataset
        dataset_dict['train_dataloader'] = train_dataloader
        dataset_dict['test_dataset'] = test_dataset
        dataset_dict['test_dataloader'] = test_dataloader

    elif data_type == DataType.heat_1d:
        dataset_X = Tensor(np.load('datasets/heat/32x32x1_data.npy'))
        dataset_Y = Tensor(np.load('datasets/heat/32x32x1_label.npy'))
        x_train = (dataset_X[:(len(dataset_X) * 4) // 5] + dataset_X.min()) * 2 / (dataset_X.max() - dataset_X.min()) - 1.0
        x_test = (dataset_X[(len(dataset_X) * 4) // 5:] + dataset_X.min()) * 2 / (dataset_X.max() - dataset_X.min()) - 1.0
        y_train = dataset_Y[:(len(dataset_Y) * 4) // 5]
        y_test = dataset_Y[(len(dataset_Y) * 4) // 5:]

        if conv_type == ConvType.Conv3d:
            x_train = x_train[:,None,:,:,:]
            x_test = x_test[:,None,:,:,:]

        train_dataset = TensorDataset(Tensor(x_train).to(device), Tensor(y_train).to(device))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TensorDataset(Tensor(x_test).to(device), Tensor(y_test).to(device))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        dataset_dict['train_dataset'] = train_dataset
        dataset_dict['train_dataloader'] = train_dataloader
        dataset_dict['test_dataset'] = test_dataset
        dataset_dict['test_dataloader'] = test_dataloader

    elif data_type == DataType.heat_2d:
        dataset_X = Tensor(np.load(f'datasets/heat/1500x{num_T}x{num_Y}x{num_X}_data.npy'))
        dataset_Y = Tensor(np.load(f'datasets/heat/1500x{num_T}x{num_Y}x{num_X}_label.npy'))
        x_train = (dataset_X[:(len(dataset_X) * 4) // 5] + dataset_X.min()) * 2 / (dataset_X.max() - dataset_X.min()) - 1.0
        x_test = (dataset_X[(len(dataset_X) * 4) // 5:] + dataset_X.min()) * 2 / (dataset_X.max() - dataset_X.min()) - 1.0
        y_train = dataset_Y[:(len(dataset_Y) * 4) // 5]
        y_test = dataset_Y[(len(dataset_Y) * 4) // 5:]

        if conv_type == ConvType.Conv3d:
            x_train = x_train[:,None,:,:,:]
            x_test = x_test[:,None,:,:,:]

        train_dataset = TensorDataset(Tensor(x_train).to(device), Tensor(y_train).to(device))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TensorDataset(Tensor(x_test).to(device), Tensor(y_test).to(device))
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        dataset_dict['train_dataset'] = train_dataset
        dataset_dict['train_dataloader'] = train_dataloader
        dataset_dict['test_dataset'] = test_dataset
        dataset_dict['test_dataloader'] = test_dataloader

    return dataset_dict