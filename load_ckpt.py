import torch
from liif_render import Fourier_render,Fourier_render_patch,Fourier_render_patch_int
from torchvision import utils
from tqdm import tqdm

ckpt_path='./ckpts/res32to1024_v1_000200.pth'
device=torch.device('cuda:0')
info='res32to1024_v1_200iter'

img_implicit=torch.load(ckpt_path,map_location=lambda storage,loc:storage).to(device)
render=Fourier_render_patch()
# res_list=[res for res in range(256,1024,50)]+[1024]
res_list=[1200]
for res in tqdm(res_list):
    img_render=render(img_implicit,h=res,w=res)
    utils.save_image(
        img_render,
        f'./ckpt_imgs/{info}_{str(res)}.png',
        nrow=1,
        normalize=True,
        range=(-1,1),
    )