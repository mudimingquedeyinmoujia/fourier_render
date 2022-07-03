import torch
from liif_render import Fourier_render,Fourier_render2
import fourier_embed
from torchvision import utils
import os
import PIL.Image
from PIL import Image
from torchvision import transforms
import os
from torch import nn, autograd, optim
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def ssim_loss(ssim_metric):
    return 1 - ssim_metric

imgpath='./00018.png'
img=Image.open(imgpath)
device=torch.device('cuda:1')

transform_bicub = transforms.Compose([
    transforms.Resize((256, 256),interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

img_target=transform_bicub(img).to(device).unsqueeze(0)
# transforms.ToPILImage()(img_target).save('00018_256_target.png')
img_init=torch.rand(1,867,32,32).to(device)
optimizer = optim.Adam(
        [img_init],
        lr=0.0005,
        betas=(0.9, 0.99),
    )
render=Fourier_render2()

num_iters=10000
img_init.requires_grad=True
ms_ssim_metric = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
ssim_metric=SSIM(data_range=1,size_average=True,channel=3)
for i in range(num_iters+1):
    optimizer.zero_grad()
    render.zero_grad()
    img_render=render(img_init,h=256,w=256)
    # ms_ssim_m = ms_ssim_metric((img_render + 1) / 2, (img_target + 1) / 2)
    ssim_m = ssim_metric((img_render + 1) / 2, (img_target + 1) / 2)
    loss1 = ssim_loss(ssim_m)
    loss1.backward()
    optimizer.step()
    # l2 loss
    optimizer.zero_grad()
    render.zero_grad()
    img_render = render(img_init, h=256, w=256)
    loss2=(img_render-img_target).pow(2).mean()
    loss2.backward()
    optimizer.step()

    if i % 1000 == 0:
        utils.save_image(
            img_render,
            f'./imgs/res32bili-ssiml2_iter_{str(i).zfill(6)}.png',
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )
        torch.save(img_init,f'./ckpts/res32bili-ssiml2_{str(i).zfill(6)}.pth')
    if i% 10 ==0:
        print(f'iter: {i}, loss l2: {loss2.item()}, loss ssim: {loss1.item()}')