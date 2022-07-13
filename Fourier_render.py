import torch
from torch import nn
from torch.nn import functional as F
import fourier_embed
import math


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    follow the coord system as:
          +
          +
    +  +  +  +  + >
          +        y
          +
          V x
    need to use flip(-1) when use grid_sample to change to
          +
          +
    +  +  +  +  + >
          +        x
          +
          V y
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:  # H,W,2 ---> H*W,2
        ret = ret.view(-1, ret.shape[-1])
    return ret


class Fourier_render_F(torch.nn.Module):
    """
    use F(x,y) to replace f(x,y)
    """

    def __init__(self):
        super().__init__()

    def query_F(self, feat, coord):
        # feat is 特征图N,3C,H,W / coord  N,bsize,2

        coord_ = coord.clone()  # N,bsize,2
        fourier_projection = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='bilinear', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)  # N,bsize,3C

        emb_fn, out_dims = fourier_embed.get_embedder(8, math.pi, if_embed=True)
        fourier_base, _ = emb_fn(coord)  # N,bsize,3C

        fourier_series = fourier_projection * fourier_base  # N,bsize,3C
        f_R = torch.sum(fourier_series[:, :, :out_dims + 1], dim=2, keepdim=True)
        f_G = torch.sum(fourier_series[:, :, out_dims + 1:2 * out_dims + 1], dim=2, keepdim=True)
        f_B = torch.sum(fourier_series[:, :, 2 * out_dims + 1:], dim=2, keepdim=True)
        ret = torch.cat([f_R, f_G, f_B], dim=-1)  # N,bsize,3
        return ret

    def query_rgb(self, feat, coord, cell):
        # feat is 特征图N,3C,H,W / coord  N,bsize,2
        a_coord = torch.cat([coord[:, :, 0].reshape(coord.shape[0], coord.shape[1], 1) + cell,
                             coord[:, :, 1].reshape(coord.shape[0], coord.shape[1], 1) + cell], dim=2)
        b_coord = torch.cat([coord[:, :, 0].reshape(coord.shape[0], coord.shape[1], 1) - cell,
                             coord[:, :, 1].reshape(coord.shape[0], coord.shape[1], 1) + cell], dim=2)
        c_coord = torch.cat([coord[:, :, 0].reshape(coord.shape[0], coord.shape[1], 1) + cell,
                             coord[:, :, 1].reshape(coord.shape[0], coord.shape[1], 1) - cell], dim=2)
        d_coord = torch.cat([coord[:, :, 0].reshape(coord.shape[0], coord.shape[1], 1) - cell,
                             coord[:, :, 1].reshape(coord.shape[0], coord.shape[1], 1) - cell], dim=2)
        a = self.query_F(feat, a_coord)
        b = self.query_F(feat, b_coord)
        c = self.query_F(feat, c_coord)
        d = self.query_F(feat, d_coord)

        return (a - b - c + d) / (4 * cell * cell)  # N,bsize,3

    def batched_predict(self, inp, coord, cell, bsize):
        n = coord.shape[1]  # pixels
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = self.query_rgb(inp, coord[:, ql: qr, :], cell)  # query_rgb : N,bsize,2 ---> N,bsize,3
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)

        return pred

    def forward(self, img_feature, h=256, w=256, cell_scale=1, bsize=30000):
        coord = make_coord((h, w)).to(img_feature.device)  # h*w,2
        cell = 2 / (h * cell_scale) / 2  # cell_scale>1 for SR
        N = img_feature.shape[0]
        pred = self.batched_predict(img_feature,
                                    coord.unsqueeze(0).repeat(N, 1, 1), cell,
                                    bsize=bsize)  # 输入坐标UV，输出RGB
        pred = pred.view(N, h, w, 3).permute(0, 3, 1, 2)  # N,pixels,3  --->  N,C,H,W

        return pred


class Fourier_render_patch(torch.nn.Module):
    """
    means every patch is independent
    not interpolation
    """

    def __init__(self):
        super().__init__()
        self.emb_fn = None
        self.out_dims = None

    def query_F(self, feat, coord, mul):
        # feat: N,3C,H,W / coord:  N,bsize,2
        assert feat.shape[1] == (4 * mul * mul + 4 * mul + 1) * 3
        coord_ = coord.clone()  # N,bsize,2
        coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
        fourier_projection = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)  # N,bsize,3C

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device) \
            .permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])  # N,2,H,W

        q_coord = F.grid_sample(  # 把特征图像素的坐标得到
            feat_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        rel_coord = coord - q_coord  # N,bsize,2
        rel_coord[:, :, 0] *= feat.shape[-2]
        rel_coord[:, :, 1] *= feat.shape[-1]  # scale to -1,1

        fourier_base, _ = self.emb_fn(rel_coord)  # N,bsize,3C

        fourier_series = fourier_projection * fourier_base  # N,bsize,3C
        f_R = torch.sum(fourier_series[:, :, :self.out_dims + 1], dim=2, keepdim=True)
        f_G = torch.sum(fourier_series[:, :, self.out_dims + 1:2 * self.out_dims + 1], dim=2, keepdim=True)
        f_B = torch.sum(fourier_series[:, :, 2 * self.out_dims + 1:], dim=2, keepdim=True)
        ret = torch.cat([f_R, f_G, f_B], dim=-1)  # N,bsize,3
        return ret

    def query_rgb(self, feat, coord, cell, mul):
        # feat : N,3C,H,W / coord:  N,bsize,2

        return self.query_F(feat, coord, mul)

    def batched_predict(self, inp, coord, cell, mul, bsize):
        n = coord.shape[1]  # pixels
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = self.query_rgb(inp, coord[:, ql: qr, :], cell, mul)  # query_rgb : N,bsize,2 ---> N,bsize,3
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)

        return pred

    def forward(self, img_feature, h=256, w=256, mul=8, cell_scale=1, omega=math.pi, bsize=30000):
        self.emb_fn, self.out_dims = fourier_embed.get_embedder(multires=mul, omega=omega,
                                                                if_embed=True)  # out_dims = C not 3C
        coord = make_coord((h, w)).to(img_feature.device)  # h*w,2
        cell = 2 / (h * cell_scale) / 2  # cell_scale>1 for SR
        N = img_feature.shape[0]
        pred = self.batched_predict(img_feature,
                                    coord.unsqueeze(0).repeat(N, 1, 1), cell, mul,
                                    bsize=bsize)  # 输入坐标UV，输出RGB
        pred = pred.view(N, h, w, 3).permute(0, 3, 1, 2)  # N,pixels,3  --->  N,C,H,W

        return pred


class Fourier_render_patch_int(torch.nn.Module):
    """
    means every patch is independent
    use interpolation in liif
    """

    def __init__(self):
        super().__init__()
        self.emb_fn = None
        self.out_dims = None

    def query_F(self, feat, coord, mul):
        # feat: N,3C,H,W / coord:  N,bsize,2
        assert feat.shape[1] == (4 * mul * mul + 4 * mul + 1) * 3
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        rx = 2 / feat.shape[-2] / 2  # half pixel of feature map
        ry = 2 / feat.shape[-1] / 2
        # get the center point coord of feature map
        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device) \
            .permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])  # N,2,H,W

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()  # N,bsize,2
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                fourier_projection_tmp = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)  # N,bsize,3C
                q_coord = F.grid_sample(  # 把特征图像素的坐标得到
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord  # N,bsize,2
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]  # scale to -1,1
                fourier_base_tmp, _ = self.emb_fn(rel_coord)  # N,bsize,3C
                fourier_series = fourier_projection_tmp * fourier_base_tmp  # N,bsize,3C
                f_R = torch.sum(fourier_series[:, :, :self.out_dims + 1], dim=2, keepdim=True)
                f_G = torch.sum(fourier_series[:, :, self.out_dims + 1:2 * self.out_dims + 1], dim=2, keepdim=True)
                f_B = torch.sum(fourier_series[:, :, 2 * self.out_dims + 1:], dim=2, keepdim=True)
                pred = torch.cat([f_R, f_G, f_B], dim=-1)  # N,bsize,3
                preds.append(pred)
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)  # area: N,30000

        tot_area = torch.stack(areas).sum(dim=0)  # areas: 4*N,bsize tot_area: 1*N,30000

        t = areas[0];
        areas[0] = areas[3];
        areas[3] = t
        t = areas[1];
        areas[1] = areas[2];
        areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret  # N,bsize,3

    def query_rgb(self, feat, coord, cell, mul):
        # feat : N,3C,H,W / coord:  N,bsize,2

        return self.query_F(feat, coord, mul)

    def batched_predict(self, inp, coord, cell, mul, bsize):
        n = coord.shape[1]  # pixels
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = self.query_rgb(inp, coord[:, ql: qr, :], cell, mul)  # query_rgb : N,bsize,2 ---> N,bsize,3
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)

        return pred

    def forward(self, img_feature, h=256, w=256, mul=8, cell_scale=1, omega=math.pi, bsize=30000):
        self.emb_fn, self.out_dims = fourier_embed.get_embedder(multires=mul, omega=omega,
                                                                if_embed=True)  # out_dims = C not 3C
        coord = make_coord((h, w)).to(img_feature.device)  # h*w,2
        cell = 2 / (h * cell_scale) / 2  # cell_scale>1 for SR
        N = img_feature.shape[0]
        pred = self.batched_predict(img_feature,
                                    coord.unsqueeze(0).repeat(N, 1, 1), cell, mul,
                                    bsize=bsize)  # 输入坐标UV，输出RGB
        pred = pred.view(N, h, w, 3).permute(0, 3, 1, 2)  # N,pixels,3  --->  N,C,H,W

        return pred


class Fourier_render_patch_avg(torch.nn.Module):
    """
    means every patch is independent
    use interpolation in liif
    """

    def __init__(self):
        super().__init__()
        self.emb_fn = None
        self.out_dims = None

    def query_F(self, feat, coord, mul):
        # feat: N,3C,H,W / coord:  N,bsize,2
        assert feat.shape[1] == (4 * mul * mul + 4 * mul + 1) * 3
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        rx = 2 / feat.shape[-2] / 2  # half pixel of feature map
        ry = 2 / feat.shape[-1] / 2
        # get the center point coord of feature map
        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device) \
            .permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])  # N,2,H,W

        preds = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()  # N,bsize,2
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                fourier_projection_tmp = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)  # N,bsize,3C
                q_coord = F.grid_sample(  # 把特征图像素的坐标得到
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord_ - q_coord  # N,bsize,2
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]  # scale to -1,1
                fourier_base_tmp,_ = self.emb_fn(rel_coord)  # N,bsize,3C
                fourier_series = fourier_projection_tmp * fourier_base_tmp  # N,bsize,3C
                f_R = torch.sum(fourier_series[:, :, :self.out_dims + 1], dim=2, keepdim=True)
                f_G = torch.sum(fourier_series[:, :, self.out_dims + 1:2 * self.out_dims + 1], dim=2, keepdim=True)
                f_B = torch.sum(fourier_series[:, :, 2 * self.out_dims + 1:], dim=2, keepdim=True)
                pred = torch.cat([f_R, f_G, f_B], dim=-1)  # N,bsize,3
                preds.append(pred)

        ret = sum(preds) / 4.
        return ret  # N,bsize,3

    def query_rgb(self, feat, coord, cell, mul):
        # feat : N,3C,H,W / coord:  N,bsize,2

        return self.query_F(feat, coord, mul)

    def batched_predict(self, inp, coord, cell, mul, bsize):
        n = coord.shape[1]  # pixels
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = self.query_rgb(inp, coord[:, ql: qr, :], cell, mul)  # query_rgb : N,bsize,2 ---> N,bsize,3
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)

        return pred

    def forward(self, img_feature, h=256, w=256, mul=8, cell_scale=1, omega=math.pi, bsize=30000):
        self.emb_fn, self.out_dims = fourier_embed.get_embedder(multires=mul, omega=omega,
                                                                if_embed=True)  # out_dims = C not 3C
        coord = make_coord((h, w)).to(img_feature.device)  # h*w,2
        cell = 2 / (h * cell_scale) / 2  # cell_scale>1 for SR
        N = img_feature.shape[0]
        pred = self.batched_predict(img_feature,
                                    coord.unsqueeze(0).repeat(N, 1, 1), cell, mul,
                                    bsize=bsize)  # 输入坐标UV，输出RGB
        pred = pred.view(N, h, w, 3).permute(0, 3, 1, 2)  # N,pixels,3  --->  N,C,H,W

        return pred
