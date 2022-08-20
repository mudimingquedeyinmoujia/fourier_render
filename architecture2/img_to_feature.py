import random

from fourier_net import Fourier_render_patch_int, calc_ind
import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import myutils
import utils
from torch import optim
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import os


class Feature_net_lrbase(torch.nn.Module):
    """
    every feature represent an img
    """

    def __init__(self, all_freq, lr_img):
        super().__init__()
        self.all_freq = all_freq
        self.feature_size = lr_img.shape[-2:]
        self.render = Fourier_render_patch_int(all_freq=all_freq)
        self.channel_list = []
        self.first_feature = lr_img
        self.feature_list = nn.ParameterList()
        for i in range(1, all_freq + 1):
            _, _, dim = calc_ind(sf=i, ef=i)
            self.channel_list.append(dim * 3)
            self.feature_list.append(nn.Parameter(torch.randn(1, dim * 3, self.feature_size[0], self.feature_size[1])))

    def get_feature(self):
        return [feat.data for feat in self.feature_list]

    def reload_feature(self, ckpt_dic):
        self.first_feature = ckpt_dic['first_feature']
        self.feature_list = nn.ParameterList()
        self.feature_list.extend(ckpt_dic['feature_list'])

    def save_features(self):
        return {
            'first_feature': self.first_feature,
            'feature_list': self.get_feature()
        }

    def pixel_rend(self, coords):
        self.render.tmp_start_f = 0
        self.render.tmp_end_f = 0
        skip = self.render.query_rgb(self.first_feature, coords)  # N,qsize,3
        for i in range(self.all_freq):
            self.render.tmp_start_f = i + 1
            self.render.tmp_end_f = i + 1
            tmp_rend = self.render.query_rgb(self.feature_list[i], coords)
            skip = tmp_rend + skip

        return skip  # N,qsize,3

    def forward(self, h, w, verbose=False):
        if verbose:
            Pn_list = []
            Sn_list = []

            skip = self.render(self.first_feature, h=h, w=w, sf=0, ef=0)
            Pn_list.append(skip)
            Sn_list.append(skip)
            for i in range(self.all_freq):
                tmp_rend = self.render(self.feature_list[i], h=h, w=w, sf=i + 1, ef=i + 1)
                Pn_list.append(tmp_rend)
                skip = tmp_rend + skip
                Sn_list.append(skip)

            return Pn_list, Sn_list
        else:
            skip = self.render(self.first_feature, h=h, w=w, sf=0, ef=0)
            for i in range(self.all_freq):
                tmp_rend = self.render(self.feature_list[i], h=h, w=w, sf=i + 1, ef=i + 1)
                skip = tmp_rend + skip

            return skip


def query_gt(gt_img, coords):
    """
    gt_img: [N,3,H,W]
    coord: [N,qbatch,2]
    """
    gt_query = F.grid_sample(
        gt_img, coords.flip(-1).unsqueeze(1),
        mode='bicubic', align_corners=False)[:, :, 0, :] \
        .permute(0, 2, 1)  # N,qsize,3
    return gt_query


def random_sample_coords(big_coords, q_batch):
    """
    the detailed coords, set to H: 64*lr_h, W: 64*lr_w [1,all_size,2]
    recommend q_batch: lr_h*lr*w
    """
    sample_lst = np.random.choice(
        len(big_coords), q_batch, replace=False)
    coords = big_coords[sample_lst]  # 1,qbatch,2
    return coords.unsqueeze(0)


def img2Feature(img_path, downscale, save_path, all_freq=2, lr=0.1, max_iter=1000, t_psnr=42, t_ssim=0.99):

    target_name = img_path.split('/')[-1].split('.')[-2]
    save_name = f'{target_name}_feat_x{downscale}'
    full_save_path = os.path.join(save_path, save_name + '.pt')
    full_log_path = os.path.join(save_path, save_name + '.txt')
    img_target = myutils.load_imageToten(img_path)
    f_h = int(img_target.shape[-2] // downscale)
    f_w = int(img_target.shape[-1] // downscale)
    img_target = img_target[:, :, :f_h * downscale, :f_w * downscale].cuda()

    big_coords = utils.make_coord((f_h*24, f_w*24)).cuda()
    first_feature = F.interpolate(img_target, size=(f_h, f_w), mode='bicubic')
    net = Feature_net_lrbase(all_freq=all_freq, lr_img=first_feature).cuda()
    log_info = [f'# img: {img_path}']
    log_info.append(f'# feature params: {utils.compute_num_params(net)}')
    optimizer = optim.Adam(
        net.parameters(),
        lr=lr,
        betas=(0.9, 0.99)
    )
    log_info.append(f'# lr: {lr}, max_iter: {max_iter}')
    metric_fn_psnr = utils.calc_psnr
    metric_fn_ssim = utils.calc_ssim
    log_info.append(f'# target psnr: {t_psnr}, target ssim: {t_ssim}')
    i = 0
    res_psnr = 0
    res_ssim = 0
    with open(full_log_path, 'a') as f:
        print('\n'.join(log_info), file=f)

    for i in range(max_iter + 1):
        optimizer.zero_grad()
        sampled_coord = random_sample_coords(big_coords, 48)  # 1,f_h*f_w,2
        gt_coord = query_gt(img_target, sampled_coord)
        pred_coord = net.pixel_rend(sampled_coord)
        loss = (gt_coord - pred_coord).pow(2).mean()

        # rend_scale = int(downscale + random.random())
        # rend_h = f_h * rend_scale
        # rend_w = f_w * rend_scale
        # img_rend = net(h=rend_h, w=rend_w)
        # img_rend = F.interpolate(img_rend, size=(img_target.shape[-2], img_target.shape[-1]))
        # img_rend.clamp_(-1, 1)
        # loss = (img_rend - img_target).pow(2).mean()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            img_rend = net(h=img_target.shape[-2], w=img_target.shape[-1])
            img_rend.clamp_(-1, 1)
            res_psnr = metric_fn_psnr((img_rend + 1) / 2, (img_target + 1) / 2)
            res_ssim = metric_fn_ssim((img_rend + 1) / 2, (img_target + 1) / 2, norm=False)
            log_info_train = f'iter: {i}, loss:{loss.item()}, psnr: {res_psnr}, ssim: {res_ssim}'
            with open(full_log_path, 'a') as f:
                print(log_info_train, file=f)
            if res_psnr > t_psnr and res_ssim > t_ssim:
                break

    log_info_train = f'# total iter: {i}, final psnr: {res_psnr}, final ssim: {res_ssim}'
    with open(full_log_path, 'a') as f:
        print(log_info_train, file=f)
    torch.save(net.save_features(), full_save_path)


if __name__ == "__main__":
    img_path = './test_imgs/0801.png'
    save_path = 'ckpts_0801_sample'
    downscale = 4
     # 1,all_size,2
    img2Feature(img_path, downscale, save_path)
