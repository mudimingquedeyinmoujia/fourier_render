from FourierSR_net import *
import torch
import ckpt_analysis
from torchvision import utils

srnet=lambda x: []
net=FourierSR_net(input_size=512, max_freq=4, srnet=srnet)

image_path='../00018.png'
img_lr=ckpt_analysis.imageToten(image_path,resize=512)
img_render=net(img_lr,h=1024,w=1024)
utils.save_image(
                img_render[0][0],
                f'./Sn_1024.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )