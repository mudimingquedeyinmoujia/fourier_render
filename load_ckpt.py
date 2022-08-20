import torch
from Fourier_render import Fourier_render_patch, Fourier_render_patch_int, Fourier_render_patch_avg
from torchvision import utils
from tqdm import tqdm
import math
from Fourier_net import FourierNet

ckpt_path = 'ckpts/net1024_v8_10step_lr001_000500.pth'
device = torch.device('cuda:0')
info = 'net1024_v8_10step_lr001_000500'

ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
net = FourierNet(all_freq=8).to(device)
net.load_state_dict(ckpt)

# img_implicit=torch.clamp(img_implicit,max=1.)
# res_list=[res for res in range(256,900,100)]+[512]
res_list = [1024]
for res in tqdm(res_list):
    img_render = net(h=res, w=res)
    for j in range(9):
        utils.save_image(
            img_render[j],
            f'./imgs_net/{info}_{str(res).zfill(6)}_{j}.png',
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )


    # utils.save_image(
    #     img_render,
    #     f'./ckpt_imgs/{info}_{str(res)}_full.png',
    #     nrow=1,
    #     normalize=True,
    #     range=(-1, 1),
    # )
    # utils.save_image(
    #     img_render1,
    #     f'./ckpt_imgs/{info}_{str(res)}_1.png',
    #     nrow=1,
    #     normalize=True,
    #     range=(-1, 1),
    # )
    # utils.save_image(
    #     img_render2,
    #     f'./ckpt_imgs/{info}_{str(res)}_2.png',
    #     nrow=1,
    #     normalize=True,
    #     range=(-1, 1),
    # )
