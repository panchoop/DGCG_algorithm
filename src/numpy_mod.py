""" Accelerated version of the operators """

import numpy as np
from . import config

def cut_off(s):
    s[s >  0.5] = 1 - s[ s > 0.5]
    s[s <= 0  ] = 0
    s[s >= 0.1] = 1
    idxs = (s>0)&(s<0.1)
    s[idxs] = 10*s[idxs]**3/0.001 - 15*s[idxs]**4/0.0001 \
                + 6*s[idxs]**5/0.00001
    return s

def TEST_FUNC(curves):
    """Applies the kernel in the given family of curves, represented by points.

    Parameters
    ----------
    curves : numpy.ndarray
        with shape TxNx2 and type double

    Returns
    -------
    evaluations : numpy.ndarray
        with shape TxNxK and type complex
    """
    curves = curves.copy()
    freqs = config.freqs_np  # TxKx2 shaped array
    top_vals = np.einsum('tni,tki->tnk', curves, freqs)  # TxNxK shaped array
    evals = np.exp(-2j*np.pi*top_vals)  # TxNxK
    cutoffs = cut_off(curves[:,:,0])*cut_off(curves[:,:,1])
    return  evals*np.expand_dims(cutoffs, -1)

