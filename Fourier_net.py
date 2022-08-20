import torch
from torch import nn
from torch.nn import functional as F
import math


def calc_ind(s, e):
    if s == 0:
        start_ind = 0
    else:
        start_ind = 4 * (s - 1) * (s - 1) + 4 * (s - 1) + 1
    end_ind = 4 * e * e + 4 * e + 1
    return start_ind, end_ind, end_ind - start_ind


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        max_freq = self.kwargs['all_freq']

        # when freqx && freqy==0
        embed_fns.append(lambda x: torch.ones((x.shape[0], x.shape[1], 1)))
        p_fns = [lambda x, y: torch.cos(x) * torch.cos(y),
                 lambda x, y: torch.cos(x) * torch.sin(y),
                 lambda x, y: torch.sin(x) * torch.cos(y),
                 lambda x, y: torch.sin(x) * torch.sin(y)]
        for freq in range(1, max_freq + 1):
            # when freqx==0 || freqy==0
            embed_fns.append(lambda x, freq=freq: torch.sin(x[:, :, 0] * freq).unsqueeze(-1))
            embed_fns.append(lambda x, freq=freq: torch.cos(x[:, :, 0] * freq).unsqueeze(-1))
            embed_fns.append(lambda x, freq=freq: torch.sin(x[:, :, 1] * freq).unsqueeze(-1))
            embed_fns.append(lambda x, freq=freq: torch.cos(x[:, :, 1] * freq).unsqueeze(-1))

            for freq_tmp in range(1, freq):
                for p_fn in p_fns:
                    embed_fns.append(lambda x, p_fn=p_fn, freqx=freq, freqy=freq_tmp:
                                     p_fn(x[:, :, 0] * freqx, x[:, :, 1] * freqy).unsqueeze(-1))
                    embed_fns.append(lambda x, p_fn=p_fn, freqx=freq_tmp, freqy=freq:
                                     p_fn(x[:, :, 0] * freqx, x[:, :, 1] * freqy).unsqueeze(-1))

            for p_fn in p_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freqx=freq, freqy=freq:
                                 p_fn(x[:, :, 0] * freqx, x[:, :, 1] * freqy).unsqueeze(-1))

        out_dim = 4 * max_freq * max_freq + 4 * max_freq + 1
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs, s, e, repeat):
        if e is None:
            e = self.kwargs['all_freq']
        assert e <= self.kwargs['all_freq'] and e >= 0
        assert s <= self.kwargs['all_freq'] and s >= 0
        start_ind, end_ind, dim = calc_ind(s, e)
        tmp_embed_fns = self.embed_fns[start_ind:end_ind]
        channel = torch.cat([fn(inputs * self.kwargs['omega']).to(inputs.device) for fn in tmp_embed_fns], -1)
        if repeat:
            return channel.repeat(1, 1, 3), dim  # channel copy from 1 to 3 (grey to RGB)
        return channel, dim


def get_embedder(all_freq, omega, if_embed=True):
    """
    获取位置编码函数与位置编码后的维度,omega与周期有关
    :param all_freq: n
    :param if_embed: if use embed
    :return: example: x,y ---> 1,cosy,siny,cos2y,sin2y,... ; outdim=4*n*n+2*n*2+1(only one channel)
    """
    if not if_embed:
        return nn.Identity(), 2

    embed_kwargs = {
        'all_freq': all_freq,
        'omega': omega,
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed_fn = lambda x, eo=embedder_obj, s=0, e=None, repeat=True: eo.embed(x, s, e, repeat)
    return embed_fn, embedder_obj.out_dim


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


class Fourier_render_patch(torch.nn.Module):
    """
    means every patch is independent
    not interpolation
    """

    def __init__(self, all_freq, omega):
        super().__init__()
        self.emb_fn, self.out_dims = get_embedder(all_freq=all_freq, omega=omega,
                                                  if_embed=True)  # out_dims = C not 3C
        self.omega = omega
        self.all_freq = all_freq

    def query_F(self, feat, coord):
        # feat: N,3C,H,W / coord:  N,bsize,2
        # assert feat.shape[1] == (4 * mul * mul + 4 * mul + 1) * 3
        coord_ = coord.clone()  # N,bsize,2
        coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
        fourier_projection = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)  # N,bsize,3C

        # feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device) \
        #     .permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])  # N,2,H,W

        center_coord = make_coord((256, 256), flatten=False).to(feat.device) \
            .permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, 256, 256)  # N,2,H,W

        q_coord = F.grid_sample(  # 把特征图像素的坐标得到
            center_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        rel_coord = coord - q_coord  # N,bsize,2
        # rel_coord = coord - center_coord  # N,bsize,2
        rel_coord[:, :, 0] *= feat.shape[-2]
        rel_coord[:, :, 1] *= feat.shape[-1]  # scale to -1,1
        # rel_coord = coord_
        fourier_base, tmp_dim = self.emb_fn(rel_coord, s=self.tmp_start, e=self.tmp_end)  # N,bsize,3C / C

        fourier_series = fourier_projection * fourier_base  # N,bsize,3C
        f_R = torch.sum(fourier_series[:, :, :tmp_dim], dim=2, keepdim=True)
        f_G = torch.sum(fourier_series[:, :, tmp_dim:2 * tmp_dim], dim=2, keepdim=True)
        f_B = torch.sum(fourier_series[:, :, 2 * tmp_dim:], dim=2, keepdim=True)
        ret = torch.cat([f_R, f_G, f_B], dim=-1)  # N,bsize,3
        return ret

    def query_rgb(self, feat, coord):
        # feat : N,3C,H,W / coord:  N,bsize,2

        return self.query_F(feat, coord)

    def batched_predict(self, inp, coord, bsize):
        n = coord.shape[1]  # pixels
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = self.query_rgb(inp, coord[:, ql: qr, :])  # query_rgb : N,bsize,2 ---> N,bsize,3
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)

        return pred

    def forward(self, img_feature, h=256, w=256, s=0, e=None, bsize=30000):

        self.tmp_start = s
        if e is None:
            self.tmp_end = self.all_freq
        else:
            self.tmp_end = e
        _, _, dms = calc_ind(s, e)
        assert img_feature.shape[1] == dms * 3
        coord = make_coord((h, w)).to(img_feature.device)  # h*w,2
        N = img_feature.shape[0]
        pred = self.batched_predict(img_feature,
                                    coord.unsqueeze(0).repeat(N, 1, 1),
                                    bsize=bsize)  # 输入坐标UV，输出RGB
        pred = pred.view(N, h, w, 3).permute(0, 3, 1, 2)  # N,pixels,3  --->  N,C,H,W

        return pred


class FourierNet(torch.nn.Module):
    """
    multi subnetwork for img generation
    """

    def __init__(self, all_freq, split_list=None, omega=math.pi):
        super().__init__()
        self.all_freq = all_freq
        self.split_list = split_list
        if split_list is None:
            self.split_list = {
                0: [0, 0],
                1: [1, 1],
                2: [2, 2],
                3: [3, 3],
                4: [4, 4],
                5: [5, 5],
                6: [6, 6],
                7: [7, 7],
                8: [8, 8],
            }
        # embed_fn_all: (input: [N,bsize,2] ,s,e,repeat)---> (output: [N,bsize,C*3],C )
        # self.embed_fn_all,self.all_dims=get_embedder(all_freq,omega=omega)
        self.render = Fourier_render_patch(all_freq, omega=omega)
        # self.p0 = nn.Parameter(torch.randn(1, 867, 32, 32))
        self.p0 = nn.Parameter(torch.randn(1, 3, 256, 256))
        self.p1 = nn.Parameter(torch.randn(1, 24, 256, 256))
        self.p2 = nn.Parameter(torch.randn(1, 48, 128, 128))
        self.p3 = nn.Parameter(torch.randn(1, 72, 128, 128))
        self.p4 = nn.Parameter(torch.randn(1, 96, 64, 64))
        self.p5 = nn.Parameter(torch.randn(1, 120, 64, 64))
        self.p6 = nn.Parameter(torch.randn(1, 144, 32, 32))
        self.p7 = nn.Parameter(torch.randn(1, 168, 32, 32))
        self.p8 = nn.Parameter(torch.randn(1, 192, 16, 16))
        self.p_list = [self.p0, self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8]

    def FeatToRGB_Block(self, feature, block_id, h, w, bsize=30000, skip=None):
        s, e = self.split_list[block_id]
        out = self.render(feature, h=h, w=w, s=s, e=e, bsize=bsize)
        if skip is None:
            return out, out
        else:
            return out + skip, out

    def forward(self, h, w):
        tmp0, f0 = self.FeatToRGB_Block(self.p0, block_id=0, h=h, w=w)
        tmp1, f1 = self.FeatToRGB_Block(self.p1, block_id=1, h=h, w=w, skip=tmp0)
        tmp2, f2 = self.FeatToRGB_Block(self.p2, block_id=2, h=h, w=w, skip=tmp1)
        tmp3, f3 = self.FeatToRGB_Block(self.p3, block_id=3, h=h, w=w, skip=tmp2)
        tmp4, f4 = self.FeatToRGB_Block(self.p4, block_id=4, h=h, w=w, skip=tmp3)
        tmp5, f5 = self.FeatToRGB_Block(self.p5, block_id=5, h=h, w=w, skip=tmp4)
        tmp6, f6 = self.FeatToRGB_Block(self.p6, block_id=6, h=h, w=w, skip=tmp5)
        tmp7, f7 = self.FeatToRGB_Block(self.p7, block_id=7, h=h, w=w, skip=tmp6)
        tmp8, f8 = self.FeatToRGB_Block(self.p8, block_id=8, h=h, w=w, skip=tmp7)

        return [[tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8], [f0, f1, f2, f3, f4, f5, f6, f7, f8]]


if __name__ == "__main__":
    print('ok')
