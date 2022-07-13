import numpy as np
import torch
import torch.nn as nn
import math


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        max_freq = self.kwargs['multires']

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

    def calc_ind(self, s, e):
        if s == 0:
            start_ind = 0
        else:
            start_ind = 4 * (s - 1) * (s - 1) + 4 * (s - 1) + 1
        end_ind = 4 * e * e + 4 * e + 1
        return start_ind, end_ind, end_ind - start_ind

    def embed(self, inputs, s, e, repeat):
        if e is None:
            e = self.kwargs['multires']
        assert e <= self.kwargs['multires'] and e >= 0
        assert s <= self.kwargs['multires'] and s >= 0
        start_ind, end_ind, dim = self.calc_ind(s, e)
        tmp_embed_fns = self.embed_fns[start_ind:end_ind]
        channel = torch.cat([fn(inputs * self.kwargs['omega']).to(inputs.device) for fn in tmp_embed_fns], -1)
        if repeat:
            return channel.repeat(1, 1, 3), dim  # channel copy from 1 to 3 (grey to RGB)
        return channel, dim


def get_embedder(multires, omega, if_embed=True):
    """
    获取位置编码函数与位置编码后的维度,omega与周期有关
    :param multires: n
    :param if_embed: if use embed
    :return: example: x,y ---> 1,cosy,siny,cos2y,sin2y,... ; outdim=4*n*n+2*n*2+1(only one channel)
    """
    if not if_embed:
        return nn.Identity(), 2

    embed_kwargs = {
        'multires': multires,
        'omega': omega,
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed_fn = lambda x, eo=embedder_obj, s=0, e=None, repeat=True: eo.embed(x, s, e, repeat)
    return embed_fn, embedder_obj.out_dim


if __name__ == '__main__':
    features1 = np.arange(10).reshape(5, 2)
    features2 = np.arange(10).reshape(5, 2)
    features_ten1 = torch.Tensor(features1)
    features_ten2 = torch.Tensor(features2)
    features_ten = torch.stack((features_ten1, features_ten2)) / math.pi
    print(features_ten)
    print(features_ten.shape)
    emb_fn, all_dims = get_embedder(multires=8, omega=math.pi, if_embed=True)
    res, tmp_dim = emb_fn(features_ten)
    print(res[0, :, :5])
    print(tmp_dim, all_dims)
    print(res.shape)
