import torch
from torch.nn import functional as F
import math
from Fourier_SRNet import *
from torch import nn, autograd, optim

net = Fourier_SRNet(input_size=256)
info = 'net_v1'
device = torch.device("cuda:0")
resolution = 256
epochs=1000
batch = 4
optimizer = optim.Adam(
    net.parameters(),
    lr=0.01,
    betas=(0.9, 0.99),
)

for i in range(epochs+1):
    img_init = torch.randn(batch, 3, resolution, resolution)
    optimizer.zero_grad()
    img_render=net(img_init,resolution,resolution)
    loss = (img_render - img_init).pow(2).mean()
    loss.backward()
    optimizer.step()
    if i%10 ==0:
        print(f'iter: {i}, loss l2: {loss.item()}')
    if i%100 == 0:
        torch.save(net.state_dict(), f'./net_ckpts/{info}_{str(i).zfill(6)}.pth')

