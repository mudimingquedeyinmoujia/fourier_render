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


imgpath='./00018.png'
img=Image.open(imgpath)

transform_near = transforms.Compose([
    transforms.Resize((256, 256),interpolation=PIL.Image.NEAREST),
    transforms.ToTensor()])

transform_bili = transforms.Compose([
    transforms.Resize((256, 256),interpolation=PIL.Image.BILINEAR),
    transforms.ToTensor()])

transform_bicub = transforms.Compose([
    transforms.Resize((256, 256),interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor()])

img_target=transform_bicub(img)
transforms.ToPILImage()(img_target).save('00018_256.png')
img_init=torch.rand(1,867,2,2)
optimizer = optim.Adam(
        [img_init],
        lr=0.001,
        betas=(0.9, 0.99),
    )
render=Fourier_render2()

num_iters=500
img_init.requires_grad=True
for i in range(num_iters+1):
    optimizer.zero_grad()
    img_render=render(img_init,h=256,w=256)
    loss=(img_render-img_target).pow(2).mean()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        utils.save_image(
            img_render,
            f'./imgs/iter_{str(i).zfill(6)}.png',
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )
    print(f'loss: {loss.item()}')