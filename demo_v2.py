import torch
from torch.nn import functional as F
import math
from ckpt_analysis import *
from Fourier_net import FourierNet
from torch import nn, autograd, optim
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


def ssim_loss(ssim_metric):
    return 1 - ssim_metric


device = torch.device('cuda:0')
target_path = '00018.png'
info = 'net_vvv7'
img_target = imageToten(target_path, resize=512)
net = FourierNet(all_freq=8).to(device)
optimizer = optim.Adam(
    net.parameters(),
    lr=0.01,
    betas=(0.9, 0.99),
)

num_iters = 3000
ms_ssim_metric = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
ssim_metric = SSIM(data_range=1, size_average=True, channel=3)
for i in range(num_iters + 1):
    # optimizer.zero_grad()
    # net.zero_grad()
    # img_render = net(h=1024, w=1024)
    # ssim_m = ssim_metric((img_render[8] + 1) / 2, (img_target + 1) / 2)
    # loss1 = ssim_loss(ssim_m)
    # loss1.backward()
    # optimizer.step()
    # l2 loss
    optimizer.zero_grad()
    net.zero_grad()
    img_render = net(h=512, w=512)
    loss2 = (img_render[0][8] - img_target).pow(2).mean()
    loss2.backward()
    optimizer.step()

    if i % 10 == 0:
        for j in range(9):
            utils.save_image(
                img_render[0][j],
                f'./imgs_net/vvv7/{info}_Sn_{str(i).zfill(6)}_{j}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
            utils.save_image(
                img_render[1][j],
                f'./imgs_net/vvv7/{info}_Pn_{str(i).zfill(6)}_{j}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

        # utils.save_image(
        #     img_render1,
        #     f'./imgs_net/{info}_{str(i).zfill(6)}_1.png',
        #     nrow=1,
        #     normalize=True,
        #     range=(-1, 1),
        # )
        # utils.save_image(
        #     img_render2,
        #     f'./imgs_net/{info}_{str(i).zfill(6)}_2.png',
        #     nrow=1,
        #     normalize=True,
        #     range=(-1, 1),
        # )
        # utils.save_image(
        #     img_render,
        #     f'./imgs_net/{info}_{str(i).zfill(6)}_full.png',
        #     nrow=1,
        #     normalize=True,
        #     range=(-1, 1),
        # )
        # print(f'iter: {i}, loss l2: {loss2.item()}, loss ssim: {loss1.item()}')
        print(f'iter: {i}, loss l2: {loss2.item()}')
    if i % 100 == 0:
        torch.save(net.state_dict(), f'./ckpts/{info}_{str(i).zfill(6)}.pth')
