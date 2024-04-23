from torch import no_grad, Tensor
import numpy as np
from src.diff_utils import DataType, device
import matplotlib.pyplot as plt
import matplotlib.animation as animation


@no_grad()
def sampling_traj(d, x_T, T, times, y, num_samples):
    i = T
    x_Ts = [x_T]
    while i >= 1:
        x_T = d.sample(x_T, times[i:i+1], y)
        i -= 1
        if i % (T // (num_samples - 1)) == 0:
            x_Ts.append(x_T)
    return x_Ts, y


def show_img_from_tensor(x_T, scale=True):
    new_img = (x_T.cpu().detach().numpy())
    if scale:
        new_img -= new_img.min()
        new_img /= (new_img.max() - new_img.min())
    if new_img.shape[0] == 1:
        plt.imshow(new_img[0], cmap='gray')
        # plt.show()
    else:
        plt.imshow(np.moveaxis(new_img, 0, 2))
        # plt.show()


def pick_random_label(data_type : DataType):
    if data_type == DataType.mnist or data_type == DataType.images:
        return Tensor([[np.random.randint(10)]]).to(device)
    elif data_type == DataType.heat_1d:
        return Tensor([[np.random.random() * 2]]).to(device)
    elif data_type == DataType.heat_2d:
        return Tensor([np.random.random(size=3)]).to(device)
    

@no_grad()
def denoise(d, x_T, T, times, y):
    i = T
    while i >= 1:
        x_T = d.sample(x_T, times[i:i+1], y)
        i -= 1
    return x_T

def show_img(x_T, scale=True):
    new_img = (x_T.cpu().detach().numpy())
    if scale:
        new_img -= new_img.min()
        new_img /= (new_img.max() - new_img.min())
    if new_img.shape[0] == 1:
        plt.imshow(new_img[0], cmap='gray')
        plt.show()
    else:
        plt.imshow(np.moveaxis(new_img, 0, 2))
        plt.show()


def make_gif(filename, np_arr, num_T, fps):
    fig = plt.figure()
    im = plt.imshow(np_arr[0])
    def animate_func(i):
        im.set_array(np_arr[i])
        return [im]
    anim = animation.FuncAnimation(fig, animate_func, frames=num_T, interval=1000/fps)
    anim.save(filename, fps=fps)