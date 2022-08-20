import numpy as np
import torch
import torch.nn as nn
import math


def p_n(x, freq):
    """
    legendre position encoding
    get N,bsize,1
    return N,bsize,(freq+1)^2
    """
    n, b, _ = x.shape
    res = [(0.5**0.5)*torch.ones(n, b).unsqueeze(-1).to(x.device), (1.5**0.5)*x]
    for i in range(2, freq + 1):
        res.append((res[i - 1] * x * (2 - 1 / i) + (1 / i - 1) * res[i - 2])*(i+0.5)**0.5)
    return res


def p_n2D(coord, freq):
    """
    get N,bsize,2
    return N,bsize,(freq+1)^2
    """
    n, b, c = coord.shape
    li_x = p_n(coord[:, :, 0].unsqueeze(-1), freq)
    li_y = p_n(coord[:, :, 1].unsqueeze(-1), freq)
    res = [0.5*torch.ones(n, b).unsqueeze(-1).to(coord.device)]
    for tmp_f in range(freq + 1):
        for i in range(tmp_f):
            res.append(li_x[i] * li_y[tmp_f])
            res.append(li_x[tmp_f] * li_y[i])
        res.append(li_x[tmp_f] * li_y[tmp_f])
    return torch.cat(res, dim=-1)


if __name__ == "__main__":
    a = torch.arange(10).resize(1, 5, 2)
    b = torch.arange(10).resize(1, 5, 2)
    ten = torch.cat([a, b], dim=0)
    print(ten)
    c = p_n2D(ten, 3)
    print(c)
