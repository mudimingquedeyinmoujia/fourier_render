import torch
from torch import nn
from torch.nn import functional as F
import fourier_embed


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):  # in_dim=64*9+2+2 out_dim=3 hidden_list=[256,256,256,256]
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
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


class Fourier_render(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def query_F(self, feat, coord):
        # feat is 特征图N,3C,H,W / coord  N,bsize,2

        coord_ = coord.clone()  # N,bsize,2
        fourier_projection = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)  # N,bsize,3C

        emb_fn, out_dims = fourier_embed.get_embedder(8, if_embed=True)
        fourier_base = emb_fn(coord)  # N,bsize,3C

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

        return (a - b - c + d) / (4 * cell *cell) # N,bsize,3

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

class Fourier_render2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def query_F(self, feat, coord):
        # feat is 特征图N,3C,H,W / coord  N,bsize,2

        coord_ = coord.clone()  # N,bsize,2
        fourier_projection = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)  # N,bsize,3C

        emb_fn, out_dims = fourier_embed.get_embedder(8, if_embed=True)
        fourier_base = emb_fn(coord)  # N,bsize,3C

        fourier_series = fourier_projection * fourier_base  # N,bsize,3C
        f_R = torch.sum(fourier_series[:, :, :out_dims + 1], dim=2, keepdim=True)
        f_G = torch.sum(fourier_series[:, :, out_dims + 1:2 * out_dims + 1], dim=2, keepdim=True)
        f_B = torch.sum(fourier_series[:, :, 2 * out_dims + 1:], dim=2, keepdim=True)
        ret = torch.cat([f_R, f_G, f_B], dim=-1)  # N,bsize,3
        return ret

    def query_rgb(self, feat, coord, cell):
        # feat is 特征图N,3C,H,W / coord  N,bsize,2

        return self.query_F(feat,coord)

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


class LIIF_rendercopy(torch.nn.Module):
    def __init__(self, feature_channel=64, local_ensemble=True, feat_unfold=False, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode
        self.feat_unfold = feat_unfold
        self.mlp_in_dim = feature_channel
        if self.feat_unfold:
            self.mlp_in_dim *= 9
        self.mlp_in_dim += 2
        if self.cell_decode:
            self.mlp_in_dim += 2
        self.imnet = MLP(in_dim=self.mlp_in_dim, out_dim=3, hidden_list=[256, 256, 256, 256])

    def query_rgb(self, feat, coord, cell=None):
        # feat is 特征图N,C,H,W / coord  N,bsize,2

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),  # 1,1,n_sample,yx  为了匹配到正常坐标
                                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)  # N,C,H=1,W=n_sample --->
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2],
                feat.shape[3])  # 1,64,67,75 ---> 1,576,67,75 因为有9个像素相邻，所以待查询的特征图每一像素由编码特征的9个相邻像素相关

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2  # half pixel of feature map
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])  # N,2,H,W 是特征图 1,576,67,75的像素中心坐标
        # print('rx {} ry {} of feature'.format(rx, ry))
        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()  # 1,30000,2
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(  # 采样点在原特征图上肯定覆盖了四个特征图像素，把特征图像素的特征得到
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)  # 1,30000,576
                q_coord = F.grid_sample(  # 把特征图像素的坐标得到
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord  # 1,bsize,2
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]  # scale to -1,1
                inp = torch.cat([q_feat, rel_coord], dim=-1)  # 1,30000,578(576+2)
                # print(coord[0,:5,:])
                # print(q_coord[0,:5,:])
                # print(rel_coord[0,:5,:])

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]  # cell是高分辨图图像像素大小，也就是查询点周围区域大小
                    rel_cell[:, :, 1] *= feat.shape[-1]  # rel_cell是相对大小
                    inp = torch.cat([inp, rel_cell], dim=-1)  # 1,30000,580(576+2+2)

                bs, q = coord.shape[:2]  # 1,n_sample
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)  # 是这个特征点应有的颜色
                preds.append(pred)  # pred : 1,30000,3

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)  # area: 1,30000

        tot_area = torch.stack(areas).sum(dim=0)  # areas: 4*1,30000 tot_area: 1,30000
        if self.local_ensemble:
            t = areas[0];
            areas[0] = areas[3];
            areas[3] = t
            t = areas[1];
            areas[1] = areas[2];
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret  # 1,30000,3

    def batched_predict(self, inp, coord, cell, bsize):

        n = coord.shape[1]  # pixels
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = self.query_rgb(inp, coord[:, ql: qr, :], cell[:, ql: qr, :])  # query_rgb : 1,bsize,2 ---> 1,bsize,3
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)

        return pred

    def forward(self, img_feature, h=256, w=256, cell_scale=1, bsize=30000):
        coord = make_coord((h, w)).to(img_feature.device)  # h*w,2
        cell = torch.ones_like(coord).to(img_feature.device)
        cell[:, 0] *= 2 / (h * cell_scale)  # cell_scale>=1
        cell[:, 1] *= 2 / (w * cell_scale)
        N = img_feature.shape[0]
        pred = self.batched_predict(img_feature,
                                    coord.unsqueeze(0).repeat(N, 1, 1), cell.unsqueeze(0).repeat(N, 1, 1),
                                    bsize=bsize)  # 输入坐标UV，输出RGB
        pred = pred.view(N, h, w, 3).permute(0, 3, 1, 2)  # N,pixels,3  --->  N,C,H,W

        return pred
