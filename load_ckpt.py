import torch
from Fourier_render import Fourier_render_patch,Fourier_render_patch_int,Fourier_render_patch_avg
from torchvision import utils
from tqdm import tqdm
import math
ckpt_path='./ckpts/res32to512_v6_000800.pth'
device=torch.device('cuda:0')
info='res32to512_v6_800iter'

img_implicit=torch.load(ckpt_path,map_location=lambda storage,loc:storage).to(device)
render=Fourier_render_patch()
# res_list=[res for res in range(256,1024,50)]+[512]
res_list=[512]
for res in tqdm(res_list):
    img_render=render(img_implicit,h=res,w=res,omega=0.8*math.pi)
    utils.save_image(
        img_render,
        f'./ckpt_imgs/{info}_{str(res)}.png',
        nrow=1,
        normalize=True,
        range=(-1,1),
    )