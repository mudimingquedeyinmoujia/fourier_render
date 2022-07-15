import torch
from Fourier_render import Fourier_render_patch, Fourier_render_patch_int, Fourier_render_patch_avg
from torchvision import utils
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import PIL.Image
from torchvision import transforms
from fourier_embed import *

device = "cuda:0"


def loss_curve(epochs, loss_list):
    epochs_list = np.arange(epochs) + 1

    plt.plot(epochs_list, loss_list, label="loss")
    plt.xlabel('freq')
    plt.ylabel('value')
    plt.legend(loc=0, ncol=1)  # 参数：loc设置显示的位置，0是自适应；ncol设置显示的列数

    plt.show()


def imageToten(loadpath, resize, norm=True):
    img = Image.open(loadpath)

    if norm:
        transform_bicub = transforms.Compose([
            transforms.Resize((resize, resize), interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform_bicub = transforms.Compose([
            transforms.Resize((resize, resize), interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
        ])

    img_res = transform_bicub(img).unsqueeze(0).to(device)
    return img_res


def tenToimage(imgTensor, svname, norm=True):
    utils.save_image(
        imgTensor,
        os.path.join('./analysis_image', svname),
        nrow=1,
        normalize=norm,
        range=(-1, 1),
    )


def load_ckpt(ckpt_path):
    img_implicit = torch.load(ckpt_path, map_location=lambda storage, loc: storage).to(device)
    return img_implicit


def freqAnalysis():
    epochs = 289

    ckpt_path = './ckpts/res32to512_v3_10step_000230.pth'
    device = torch.device('cuda:0')
    info = 'analyse'

    img_implicit = torch.load(ckpt_path, map_location=lambda storage, loc: storage).to(device)
    # print(img_implicit[0,:289,0,0])
    loss_list = [img_implicit[0, i, 18, 8].item() for i in range(289)]
    loss_curve(epochs, loss_list)


def channelExtract(implicit, s, e):
    n, c, h, w = implicit.shape
    cc = int(c / 3)
    s_ind, e_ind, num = calc_ind(s, e)
    channel_list = [implicit[:, s_ind:e_ind, :, :], implicit[:, cc + s_ind:cc + e_ind, :, :],
                    implicit[:, 2 * cc + s_ind:2 * cc + e_ind, :, :]]
    res=torch.cat(channel_list,dim=1)
    return res


def freqVis():
    img_implicit = load_ckpt(ckpt_path='./ckpts/res32to512_v3_10step_lr001_000200.pth')
    channel_no = 0
    render = Fourier_render_patch_int()
    img_list=[]
    for i in range(9):
        s=0
        e=i
        img_ext=channelExtract(img_implicit,s,e)
        img_rend = render(img_ext, w=256, h=256, s=s, e=e)
        img_list.append(img_rend)
        tenToimage(img_rend, svname=f'test03-to{i}.png')
    # img_all=sum(img_list)
    # tenToimage(img_all,svname=f'test03-all.png')


if __name__ == "__main__":
    freqVis()
