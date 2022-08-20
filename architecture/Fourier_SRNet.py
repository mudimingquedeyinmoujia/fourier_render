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

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device) \
            .permute(2, 0, 1).unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])  # N,2,H,W

        q_coord = F.grid_sample(  # 把特征图像素的坐标得到
            feat_coord, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        rel_coord = coord - q_coord  # N,bsize,2
        rel_coord[:, :, 0] *= feat.shape[-2]
        rel_coord[:, :, 1] *= feat.shape[-1]  # scale to -1,1

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


class Render_block(torch.nn.Module):
    """
    feature to tmp RGB
    """

    def __init__(self, block_id, in_channels, render):
        super().__init__()
        self.block_id = block_id
        self.in_channels = in_channels
        self.s_ind, self.e_ind, self.c = calc_ind(s=block_id, e=block_id)
        self.ccc = self.c * 3
        self.render = render
        if self.block_id != 0:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.ccc, 1),  # in_channels, out_channels, kernel_size
                nn.ReLU(),
                nn.Conv2d(self.ccc, self.ccc, 1),  # in_channels, out_channels, kernel_size
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(self.in_channels, 3, 1),
                nn.ReLU(),
            )

    def forward(self, feature, h, w, bsize, skip=None):
        feature_extract = self.conv(feature)
        out = self.render(feature_extract, h=h, w=w, s=self.block_id, e=self.block_id, bsize=bsize)
        if skip is None:
            return out
        else:
            return out + skip


class ConvDown_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, id):
        super().__init__()
        self.id = id
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.id != 0:
            self.convEqual = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
                # in_channels, out_channels, kernel_size
                nn.ReLU(),
            )
            self.convDown = nn.Sequential(
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=2, stride=2),
                # in_channels, out_channels, kernel_size
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1),
                # in_channels, out_channels, kernel_size
                nn.ReLU(),
            )

    def forward(self, input):
        if self.id == 0:
            return input, None
        else:
            out_tmp = self.convEqual(input)
            out = self.convDown(out_tmp)
            return out_tmp, out


class Fourier_SRNet(torch.nn.Module):
    """
    input: image with h,w and H,W
    output: HR image with H,W
    """

    def __init__(self, input_size, omega=math.pi):
        super().__init__()
        self.input_size = input_size
        self.all_freq = int(math.log2(input_size))
        self.render = Fourier_render_patch(self.all_freq, omega=omega)
        self.convDowns = nn.ModuleList()
        self.renders = nn.ModuleList()
        self.conv_dict = {
            8: 512,
            7: 512,
            6: 256,
            5: 256,
            4: 128,
            3: 128,
            2: 64,
            1: 64,
            0: 64,
        }
        self.convDowns.append(
            ConvDown_block(in_channels=3, out_channels=self.conv_dict[self.all_freq], id=self.all_freq))
        self.renders.append(
            Render_block(block_id=self.all_freq, in_channels=self.conv_dict[self.all_freq], render=self.render))
        for id in range(self.all_freq - 1, -1, -1):
            self.convDowns.append(
                ConvDown_block(in_channels=self.conv_dict[id + 1], out_channels=self.conv_dict[id], id=id))
            self.renders.append(Render_block(block_id=id, in_channels=self.conv_dict[id], render=self.render))

    def forward(self, lr_image, h, w, bsize=30000):
        skip = None
        feature_out = lr_image
        for convB, renderB in zip(self.convDowns, self.renders):
            feature_tmp, feature_out = convB(feature_out)
            skip = renderB(feature_tmp, h, w, bsize, skip)
        return skip


if __name__ == "__main__":
    net = Fourier_SRNet(input_size=256)
    print('ok')
