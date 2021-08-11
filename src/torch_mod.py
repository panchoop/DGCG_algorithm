import torch as th
import os
import numpy as np
from . import config

# th.cuda.set_device(1)  # Offices Tesla 420c
cuda = th.device('cuda:1')

default_type = th.float64

minus_2_pi = th.tensor(-2*np.pi, device=cuda)

def tensor(val, device=cuda, dtype=default_type):
    return th.tensor(val, device=cuda, dtype=dtype)

def empty(size, device=cuda, dtype=default_type):
    return th.empty(size, device=cuda, dtype=dtype)

def cut_off(s):
    s[s >  0.5] = 1 - s[ s > 0.5]
    s[s <= 0  ] = 0
    s[s >= 0.1] = 1
    idxs = (s>0)&(s<0.1)
    s[idxs] = 10*s[idxs]**3/0.001 - 15*s[idxs]**4/0.0001 \
                + 6*s[idxs]**5/0.00001
    return None

def TEST_FUNC(freqs, x, x_alloc, out_re_alloc, out_im_alloc): # φ_t(x)
    """ t here is a boolean array """
    # freqs = config.frequencies[t, :, :]  # TxKx2 shaped
    th.tensordot(freqs, x, dims=([2],[1]), out=out_re_alloc) # TxKxN shaped
    out_re_alloc.mul_(minus_2_pi)
    th.sin(out_re_alloc, out=out_im_alloc)
    out_re_alloc.cos_()
    x_alloc.copy_(x)  # copy x into x_alloc
    # x_alloc[x_alloc > 0.5] = 1 - x_alloc[x_alloc > 0.5]
    # Here two dimensions are treated simultaneously
    # cut_off(x_alloc)
    # Multiply the two dimensions into one, and store in one column
    x_alloc[:,0].mul_(x_alloc[:,1])
    # Broadcast to multiply to the whole output
    out_re_alloc.mul_(x_alloc[:,0].unsqueeze(0).unsqueeze(0))
    out_im_alloc.mul_(x_alloc[:,0].unsqueeze(0).unsqueeze(0))
    return None

    












