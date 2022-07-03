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
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:  # false
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq=self.kwargs['multires']

        # when x=0 or y=0
        for freq in range(1,max_freq+1):
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq * math.pi))
                out_dim += d
        # other situation
        for freqx in range(1,max_freq+1):
            for freqy in range(1,max_freq+1):
                for p_fn in [lambda x,y:torch.cos(x*math.pi)*torch.cos(y*math.pi),lambda x,y:torch.cos(x*math.pi)*torch.sin(y*math.pi),
                             lambda x,y:torch.sin(x*math.pi)*torch.cos(y*math.pi),lambda x,y:torch.sin(x*math.pi)*torch.sin(y*math.pi)]:
                    embed_fns.append(lambda x,p_fn=p_fn,freqx=freqx,freqy=freqy:p_fn(x[:,:,0]*freqx,x[:,:,1]*freqy).reshape(x.shape[0],x.shape[1],1))

        embed_fns.append(lambda x:torch.ones((x.shape[0],x.shape[1],1)))
        out_dim+=4*max_freq*max_freq+1
        self.embed_fns = embed_fns  # sin(2^0 x) cos(2^0 x) sin(2^1 x) cos(2^1 x) ...
        self.out_dim = out_dim

    def embed(self, inputs):
        channel=torch.cat([fn(inputs).to(inputs.device) for fn in self.embed_fns], -1)
        return channel.repeat(1,1,3)  # 按照最后一维拼接，拼接后维度不变


def get_embedder(multires, if_embed=True):
    """
    获取位置编码函数与位置编码后的维度,使用pi
    :param multires: sinx sin2x sin3x ... sin nx
    :param if_embed: if use embed
    :return: example: x,y ---> 1,cosy,siny,cos2y,sin2y,... outdim=4*n*n+2*n*2+1
    """
    if not if_embed:
        return nn.Identity(), 2

    embed_kwargs = {
        'include_input': False,  # 输出的编码是否包含输入的特征本身
        'input_dims': 2,  # 位置坐标数量，二维
        'multires': multires,
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


if __name__ == '__main__':
    # features = np.arange(10).reshape(1,5,2)
    # features_ten1 = torch.Tensor(features).expand(2,5,2)
    # print(features_ten1.shape)
    # print(features_ten1)
    # features_ten1=features_ten1/torch.pi
    # emb_fn, out_dims = get_embedder(8, if_embed=True)
    # print(emb_fn(features_ten1), out_dims)
    # print(emb_fn(features_ten1).shape)

    features1 = np.arange(10).reshape(5, 2)
    features2 = np.arange(10,20).reshape(5, 2)
    features_ten1 = torch.Tensor(features1)
    features_ten2 = torch.Tensor(features2)
    features_ten=torch.stack((features_ten1,features_ten2))
    print(features_ten)
    print(features_ten.shape)

    features_ten = features_ten / math.pi
    emb_fn, out_dims = get_embedder(8, if_embed=True)
    print(emb_fn(features_ten), out_dims*3)


