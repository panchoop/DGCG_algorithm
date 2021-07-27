import pyopencl as cl
import pyopencl.array as clarray
import numpy as np
import os 
import time
from . import config

# os.environ['PYOPENCL_CTX'] = '0'  # Personal computer setting
os.environ['PYOPENCL_CTX'] = '0:1'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'  # Option to see compiler warnings

platform = cl.get_platforms()[0]  # Select the first platform [0]
device = platform.get_devices()[0]   # Select the first device on this platform
context = cl.create_some_context([device])  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue

# Load kernel code
with open("kernel_code.c", 'r', encoding='utf-8') as f:
    kernel_code = ''.join(f.readlines())
# Load the fixed (to not modify) kernel code
with open("kernel_code_fixed.c", 'r', encoding='utf-8') as f:
    kernel_code += ''.join(f.readlines())

# Compile the kernel code into an executable OpenCl program
program = cl.Program(context, kernel_code).build()


def clarray_init(values, astype=np.float64):
    return clarray.to_device(queue, np.require(values, astype, 'C'))

def clarray_empty(shape, astype=np.float64):
    return clarray.empty(queue, shape, dtype=astype)

class mem_alloc:
    """ Class to generate and store groups of GPU buffers."""
    def __init__(self, num, shape, astype=np.float64):
        self.arrays = []
        for _ in range(num):
            self.arrays.append(clarray_empty(shape))
        self.num = num
        self.shape = shape
        self.dtype = astype

    def append(self, *openclarrays):
        if self.num == 0:
            self.shape = openclarrays[0].shape
            self.dtype = openclarrays[0].dtype
        for openclarray in openclarrays:
            self.arrays.append(openclarray)
            assert self.shape == openclarray.shape
            assert self.dtype == openclarray.dtype
        self.num += len(openclarrays)

    def amplify_by(self, scalar):
        for openclarray in self.arrays:
            openclarray *= scalar


    def __getitem__(self, arg):
        return self.arrays[arg]

    def __len__(self):
        return self.num

    def __str__(self):
        out = ''
        for idx, openclarray in enumerate(self.arrays):
            out += 'Entry {} \n'.format(idx) + openclarray.__str__() + '\n'
        return out


def TEST_FUNC(x_cl, t_cl, out_alloc): 
    """ Test function, base of the forward operator.

    Parameters
    ----------
    x_cl, t_cl: pyopencl.array
        pyopencl.arrays loaded with the input values.
        x_cl correspond to a Nx2 shaped matrix, representing N points in R^2,
        with dtype np.float64.
        t_cl correspond to a T length array, representing the considered time
            samples with dtype np.int32
    out_alloc : mem_alloc class
        a collection of two pyopencl.arrays of shape TxNxK, representing
        the real and imaginary part of the output of this function.
    Returns
    -------
    None, out_alloc is modified instead.
    """
    assert x_cl.dtype == np.float64
    assert t_cl.dtype == np.int32
    assert isinstance(out_alloc, mem_alloc)
    assert out_alloc.dtype == np.float64
    assert len(out_alloc) == 2
    assert x_cl.shape[0] == out_alloc.shape[1]  # matching N dimension
    assert x_cl.shape[1] == 2  # dimension 2
    assert t_cl.shape[0] == out_alloc.shape[0]  # number of time sample match
    assert out_alloc.shape[2] == config.freq_cl.shape[1]  # hilbert dimension
    K = config.K
    N = x_cl.shape[0]
    T = t_cl.shape[0]
    program.TEST_FUNC(queue, (N,T,K), None,
                      config.freq_cl.data,
                      t_cl.data,
                      x_cl.data,
                      out_alloc[0].data,
                      out_alloc[1].data)
    return None

def GRAD_TEST_FUNC(x_cl, t_cl, out_alloc):
    """ Gradient of the test function, base of the forward operator.

    Parameters
    ----------
    x_cl, t_cl: pyopencl.array
        pyopencl.arrays loaded with the input values.
        x_cl correspond to a Nx2 shaped matrix, representing N points in R^2,
        with dtype np.float64
        t_cl correspond to a T length array, representing the considered time
            samples, with dtype np.int32
    out_alloc: mem_alloc class
        a collection of 4 pyopencl.arrays of shape TxNxK, representing
        the real and imaginary parts of the two partial derivatives of this
        function. In order: out_alloc[0] = real dx, out_alloc[1] = imag dx, 
        out_alloc[2] = real dy, out_alloc[3] = imag dy.
    Returns
    -------
    None, out_alloc is modified instead.
    """
    assert x_cl.dtype == np.float64
    assert t_cl.dtype == np.int32
    assert isinstance(out_alloc, mem_alloc)
    assert out_alloc.dtype == np.float64
    assert len(out_alloc) == 4
    assert x_cl.shape[0] == out_alloc.shape[1]  # matching N dimension
    assert x_cl.shape[1] == 2  # dimension 2
    assert t_cl.shape[0] == out_alloc.shape[0]  # number of time sample match
    assert out_alloc.shape[2] == config.freq_cl.shape[1]  # hilbert dimension
    K = config.K
    N = x_cl.shape[0]
    T = t_cl.shape[0]
    program.GRAD_TEST_FUNC(queue, (N,T,K), None, 
                           config.freq_cl.data,
                           t_cl.data,
                           x_cl.data,
                           out_alloc[0].data, out_alloc[1].data,
                           out_alloc[2].data, out_alloc[3].data)
    return None

def mat_vec_mul(A_cl, b_cl, output_cl):
    N = A_cl.shape[0]
    K = np.int32(A_cl.shape[1])
    program.mat_vec_mul(queue, (N,), None, K,
                        A_cl.data, b_cl.data, output_cl.data)
    return None

def einsum(t_cl, Phi_cl, data_cl, output_cl):
    """ Sum happening when parallelizing the K_t operator.

    Parameters
    ----------
    t_cl: pyopencl.array
        1-dimensional array of ints of length T, representing indexes of
        considered times.
    Phi_cl: pyopencl.array
        3-dimensional array of doubles, of shape (T, N, K), representing the
        output of the function TEST_FUNC. N represents the number o
        2-dimensional evaluation points, and K represents the dimension of the
        image Hilbert space.
    data_cl: pyopencl.array
        2-dimensional array of doubles, representing an element in the H 
        space. It has size (TT, K), with TT the total number of available 
        time samples, and K the dimension of the Hilbert spaces.
    output_cl: pyopencl.array
        2-dimensional buffer to be overwritten by this method, of shape (T,N).
        The output will be added to the current value! i.e. if output_cl has
        already some values, this process will add up new values.
    Returns
    -------
    None
    """
    assert t_cl.dtype == np.int32
    assert Phi_cl.dtype == np.float64
    assert data_cl.dtype == np.float64
    assert output_cl.dtype == np.float64
    assert t_cl.shape[0] == Phi_cl.shape[0] == output_cl.shape[0]  # T matching
    assert data_cl.shape[0] == config.T  # TT matching
    assert Phi_cl.shape[1] == output_cl.shape[1]
    assert Phi_cl.shape[2] == data_cl.shape[1]
    N = Phi_cl.shape[1]
    T = t_cl.shape[0]
    K = np.int32(Phi_cl.shape[2])
    program.einsum(queue, (N,T), None, K, t_cl.data, 
                   Phi_cl.data, data_cl.data, output_cl.data)
    return None
    

# Release data buffer
if __name__ == '__main__':
    pass
