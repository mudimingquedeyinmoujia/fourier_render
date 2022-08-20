import torch
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import PIL.Image
from torchvision import transforms
import random


def loss_curve(epochs, loss_list):
    epochs_list = np.arange(epochs) + 1

    plt.plot(epochs_list, loss_list, label="loss")
    plt.xlabel('freq')
    plt.ylabel('value')
    plt.legend(loc=0, ncol=1)  # 参数：loc设置显示的位置，0是自适应；ncol设置显示的列数

    plt.show()


def load_imageToten(loadpath, resize, norm=True):
    """
    from load path to load image to tensor
    return: [1,3,H,W]
    """
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

    img_res = transform_bicub(img).unsqueeze(0)
    return img_res


def save_tenimage(imgTensor, svpath, svname, norm=True):
    utils.save_image(
        imgTensor,
        os.path.join(svpath, svname),
        nrow=1,
        normalize=norm,
        range=(-1, 1),
    )


def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    return ckpt


def freqAnalysis():
    epochs = 289

    ckpt_path = './ckpts/res32to512_v3_10step_000230.pth'
    device = torch.device('cuda:0')
    info = 'analyse'

    img_implicit = torch.load(ckpt_path, map_location=lambda storage, loc: storage).to(device)
    # print(img_implicit[0,:289,0,0])
    loss_list = [img_implicit[0, i, 18, 8].item() for i in range(289)]
    loss_curve(epochs, loss_list)


def calc_ind_(sf, ef):
    """
    get start frequency and end frequency
    return: the start index, end index in orthogonal position encoding and tmp elements number : P_sf + ... + P_ef
    """
    if sf == 0:
        start_ind = 0
    else:
        start_ind = 4 * (sf - 1) * (sf - 1) + 4 * (sf - 1) + 1
    end_ind = 4 * ef * ef + 4 * ef + 1
    return start_ind, end_ind, end_ind - start_ind


def channelExtract(implicit, sf, ef):
    """
    suppose the implicit is all_freq
    return: [N,(P_sf+...+P_ef) * 3,H,W]
    """
    n, c, h, w = implicit.shape
    cc = int(c / 3)
    s_ind, e_ind, num = calc_ind_(sf, ef)
    channel_list = [implicit[:, s_ind:e_ind, :, :], implicit[:, cc + s_ind:cc + e_ind, :, :],
                    implicit[:, 2 * cc + s_ind:2 * cc + e_ind, :, :]]
    res = torch.cat(channel_list, dim=1)
    return res


def cutMix(imageLR, imageHR):
    """
    cut mix operation for img super-resolution
    :param imageLR:
    :param imageHR:
    :return: image with size 256(LR) which have a patch of img512(HR)
    """
    N, C, H, W = imageLR.shape
    cN, cC, cH, cW = imageHR.shape
    ratio = random.random() * 0.5 + 0.5  # patch range in 0.3-0.6
    rH, rW = random.randint(0, H - int(ratio * H)), random.randint(0, W - int(ratio * W))
    ds_imageHR = torch.nn.functional.interpolate(imageHR, (H, W))
    ret = imageLR.clone()
    mix0 = random.random()
    mix1 = random.random()
    mix2 = random.random()
    if mix0 < 0.5:
        ret[:, 0, rH:rH + int(ratio * H), rW: rW + int(ratio * W)] = ds_imageHR[:, 0, rH: rH + int(ratio * H),
                                                                     rW: rW + int(ratio * W)]
    if mix1 < 0.5:
        ret[:, 1, rH:rH + int(ratio * H), rW: rW + int(ratio * W)] = ds_imageHR[:, 1, rH: rH + int(ratio * H),
                                                                     rW: rW + int(ratio * W)]
    if mix2 < 0.5:
        ret[:, 2, rH:rH + int(ratio * H), rW: rW + int(ratio * W)] = ds_imageHR[:, 2, rH: rH + int(ratio * H),
                                                                     rW: rW + int(ratio * W)]
    return ret


if __name__ == "__main__":
    img = Image.open('butterfly_15_near.png')
    resolution = 512
    transform_mo = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=PIL.Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    img_res = transform_mo(img).unsqueeze(0)
    save_tenimage(img_res, './', f'butterfly_15_to_512.png')
