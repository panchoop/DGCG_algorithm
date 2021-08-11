import torch as th
import os
import numpy as np
from . import config

""" To install Torch for my Tesla 40Kc, which has compute capabilities 3.5,
I used these precompiled binaries:
https://github.com/pytorch/pytorch/issues/30532#issuecomment-708032104
"""
th.cuda.set_device(1)  # Office's Tesla 40Kc
cuda = torch.device('cuda:1')  # Office's Tesla 40Kc



ALPHA = 0.1
BETA = 0.1
T = 51
K = 20
assert T <= 256, "T too high, increase the bits of the integer tensor"
TIME_SAMPLES = th.tensor(np.linspace(0,1,T), dtype=th.float32)
FREQ_DIMENSION = th.ones([20], dtype=th.int8)*K

def Archimedian_spiral(t, a, b):
    """ Archimedian spiral to get the frequency measurements"""
    return np.array([(a+b*t)*np.cos(t), (a+b*t)*np.sin(t)])


FREQ_SAMPLES = np.array([Archimedian_spiral(t, 0, 0.2)
                         for t in np.arange(FREQ_DIMENSION[0])])
FREQUENCIES = np.array([FREQ_SAMPLES for t in range(T)])  # at each time sample

# Store this value in memory, as it remains fixed during execution
DGCG.config.frequencies = th.tensor(FREQUENCIES, dtype=th.float64, device=cuda)
# Also store in the config file
DGCG.config.K = K

minus_2_pi = th.tensor(-2*np.pi, device=cuda)

def cut_off(s):
    s[s >  0.5] = 1 - s[ s > 0.5]
    s[s <= 0  ] = 0
    s[s >= 0.1] = 1
    idxs = (s>0)&(s<0.1)
    s[idxs] = 10*s[idxs]**3/0.001 - 15*s[idxs]**4/0.0001 \
                + 6*s[idxs]**5/0.00001
    return None

def TEST_FUNC(t, x, x_alloc, out_alloc): # φ_t(x)
    """ t here is a boolean array """
    freqs = DGCG.config.frequencies[t, :, :]  # TxKx2 shaped
    out = th.tensordot(freqs, x, dims=([2],[1]), out=out_alloc) # TxKxN shaped
    out.mul_(minus_2_pi)
    x_alloc = x.detach().clone()
    print("bin hier")
    import code; code.interact(local=dict(globals(), **locals()))
    # Here two dimensions are treated simultaneously
    cut_off(x_alloc)
    # Multiply the two dimensions into one, and store in one column
    x_alloc[:,0].mul_(x_alloc[:,1])
    # Broadcast to multiply to the whole output
    out.mul_(x_alloc[:,0].unsqueeze(0).unsqueeze(0))
    return None

    









