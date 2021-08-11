"""
Checker module.

Checks during execution that everything is working correctly (the outputs of
the functions are mapping correctly and etc). Should be replaced by proper
unit tests.

Undocumented.
"""
# Standard imports
import numpy as np
import sys

# Local imports
from . import config
from . import operators as op

def is_valid_time(t):
    # t should be an integer number in {0,1,..., T-1}
    if isinstance(t, int) or isinstance(t, np.integer):
        if t >= 0 and t <= config.T:
            return True
    return False

def is_in_H_t(t, f_t):
    # H_t should be just a 1 dimensional vector of complex numbers of size
    # H_DIMENSIONS[t], defined on the operators module
    if is_valid_time(t):
        if isinstance(f_t, np.ndarray):
            if len(f_t.shape) == 1:
                if f_t.shape[0] == op.H_DIMENSIONS[t]:
                    return True
    return False

def set_in_H_t(t, f_t):
    # f_t is a matrix whose rows are all elements of H_t, correspond to a 
    # collection of elements in H_t, for fixed t.
    if is_valid_time(t):
        if isinstance(f_t, np.ndarray):
            if len(f_t.shape)==2:
                if f_t.shape[1] == op.H_DIMENSIONS[t]:
                    return True
    return False

def is_in_H(f):
    # f should be a numpy array of  elements in H_t.
    if isinstance(f, np.ndarray):
        if f.shape[0] == config.T:
            for t in range(config.T):
                if is_in_H_t(t,f[t]) == False:
                    return False
                else:
                    return True
    return False

def H_t_product(t, f_t, g_t):
    return is_in_H_t(t, f_t) and is_in_H_t(t,g_t)

def H_t_product_full(f,g):
    return is_in_H(f) and is_in_H(g)

def is_in_space_domain(x):
    # x is a numpy array of numpy D dimensional vectors.
    # i.e. x is a NxD sized vector, with D = 2
    if isinstance(x, np.ndarray):
        if x.shape[1] == 2:
            return True
    return False







