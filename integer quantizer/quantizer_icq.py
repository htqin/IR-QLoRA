from logging import getLogger

import torch
import torch.nn as nn


tau_range = 0.1
tau_n = 50
logger = getLogger(__name__)


def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):

        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        # our ICQ start from here
        def evaluate_entropy(weight_q: torch.Tensor, maxq: torch.Tensor):
            blocksize = weight_q.shape[-1]
            weight_q_reshape = weight_q.reshape(1, weight_q.shape[0], blocksize)
            weight_q_repeat = weight_q_reshape.repeat(maxq+1, 1, 1).cuda()
            values = torch.tensor(range(maxq+1)).reshape(maxq+1, 1, 1).cuda()
            freqs = (weight_q_repeat==values).sum(dim=-1) / blocksize
            _entropy = -freqs*torch.log2(freqs)
            _entropy = torch.where(torch.isnan(_entropy), 0, _entropy)
            return _entropy.sum(dim=0)
        best = torch.full([x.shape[0]], float('inf'), device=dev)
        _p = torch.ones([x.shape[0]])
        p_left = 1 - tau_range
        p_right = 1 + tau_range
        for p in torch.cat([torch.ones(1),torch.linspace(1.0,p_right,tau_n+1)[1:],torch.linspace(1.0,p_left,tau_n+1)[1:]]):
            xmin1 = p * xmin
            xmax1 = p * xmax
            scale1 = (xmax1 - xmin1) / self.maxq
            zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
            weight_q = torch.clamp(torch.round(x / scale1.unsqueeze(1)) + zero1.unsqueeze(1), 0, self.maxq).char()
            err = -evaluate_entropy(weight_q, self.maxq)
            tmp = err < best
            if torch.any(tmp):
                _p[tmp] = p
                best[tmp] = err[tmp]
                self.scale[tmp] = scale1[tmp]
                self.zero[tmp] = zero1[tmp]
        # the end
                    
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


__all__ = ["Quantizer"]
