import pyopencl as cl
import pyopencl.array as clarray
import numpy as np
import os 
import time
from . import config

# os.environ['PYOPENCL_CTX'] = '0'  # Personal computer setting
os.environ['PYOPENCL_CTX'] = '0:0'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'  # Option to see compiler warnings

platform = cl.get_platforms()[0]  # Select the first platform [0]
device = platform.get_devices()[0]   # Select the first device on this platform
context = cl.create_some_context([device])  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue

default_type = np.float64
# Load kernel code
with open("kernel_code.c", 'r', encoding='utf-8') as f:
    kernel_code = ''.join(f.readlines())
# Load the fixed (to not modify) kernel code
with open("kernel_code_fixed.c", 'r', encoding='utf-8') as f:
    kernel_code += ''.join(f.readlines())

# Compile the kernel code into an executable OpenCl program
program = cl.Program(context, kernel_code).build()


def clarray_init(values, astype=default_type):
    return clarray.to_device(queue, np.require(values, astype, 'C'))

def clarray_empty(shape, astype=default_type):
    return clarray.empty(queue, shape, dtype=astype)

class mem_alloc:
    """ Class to generate and store groups of GPU buffers."""
    def __init__(self, num=0, shape=0, astype=default_type):
        self.arrays = []
        for _ in range(num):
            self.arrays.append(clarray_empty(shape))
        self.num = num
        self.shape = shape
        self.dtype = astype

    def append(self, openclarrays):
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


    def __setitem__(self, key, newvalue):
        self.arrays[key] = newvalue

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
        with dtype default_type.
        t_cl correspond to a T length array, representing the considered time
            samples with dtype np.int32
    out_alloc : mem_alloc class
        a collection of two pyopencl.arrays of shape TxNxK, representing
        the real and imaginary part of the output of this function.
    Returns
    -------
    None, out_alloc is modified instead.
    """
    assert x_cl.dtype == default_type
    assert t_cl.dtype == np.int32
    assert isinstance(out_alloc, mem_alloc)
    assert out_alloc.dtype == default_type
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

def TEST_FUNC_2(x_cl, t_cl, out_alloc):
    """ Test function, base of the forward operator, version 2.

    In this version, the input x_cl has a time dimension, that must coincide
    with the t_cl dimensions. Therefore the output is not all the combinations
    between x_cl and t_cl, but instead, the x_cl are combined only with the
    t_cl's that are on the same time.

    Parameters
    ----------
    x_cl, t_cl: pyopencl.array
        pyopencl.array loaded with input values.
        x_cl correspond to a TxNx2 shaped matrix, representing TxN points in
        R^2, with dype default_type
        t_cl correspond to a T length array, representing the considered time
        samples with dtype np.int32
    out_alloc : mem_alloc class
        a collection of two pyopencl.arrays of shape TxNxK, representing 
        the real and imaginary part of the output of this function.
    Returns
    -------
    None, out_alloc is modified instead.
    """
    assert x_cl.dtype == default_type
    assert t_cl.dtype == np.int32
    assert isinstance(out_alloc, mem_alloc)
    assert out_alloc.dtype == default_type
    assert len(out_alloc) == 2
    assert x_cl.shape[1] == out_alloc.shape[1]  # matching N dimension
    assert x_cl.shape[2] == 2  # dimension 2
    assert t_cl.shape[0] == x_cl.shape[0] ==  out_alloc.shape[0]  # time match
    assert out_alloc.shape[2] == config.freq_cl.shape[1]  # hilbert dimension
    K = config.K
    N = x_cl.shape[1]
    T = t_cl.shape[0]
    program.TEST_FUNC_2(queue, (N,T,K), None,
                      config.freq_cl.data,
                      t_cl.data,
                      x_cl.data,
                      out_alloc[0].data,
                      out_alloc[1].data)
    return None

def TEST_FUNC_3(x_cl, out_alloc):
    """ Test function, base of the forward operator, version 3.

    In this version, the input x_cl has a time dimension, that must coincide
    with the number of time samples of the problem, to be defined in config.T.
    Therefore, the kernel matches each positions x_cl at their respective time
    with the respective frequencies at that time.

    Parameters
    ----------
    x_cl: pyopencl.array
        pyopencl.array loaded with input values.
        x_cl correspond to a TxNx2 shaped matrix, representing TxN points in
        R^2, with dype default_type with T = config.T
    out_alloc : mem_alloc class
        a collection of two pyopencl.arrays of shape TxNxK, representing 
        the real and imaginary part of the output of this function.
    Returns
    -------
    None, out_alloc is modified instead.
    """
    assert x_cl.dtype == default_type == out_alloc.dtype
    assert isinstance(out_alloc, mem_alloc)
    assert len(out_alloc) == 2
    assert x_cl.shape[1] == out_alloc.shape[1]  # matching N dimension
    assert x_cl.shape[2] == 2  # dimension 2
    assert x_cl.shape[0] ==  out_alloc.shape[0] == config.T  # T match
    assert out_alloc.shape[2] == config.freq_cl.shape[1]  # K match 
    K = config.K
    N = x_cl.shape[1]
    T = config.T
    program.TEST_FUNC_3(queue, (N,T,K), None,
                      config.freq_cl.data,
                      x_cl.data,
                      out_alloc[0].data,
                      out_alloc[1].data)
    return None

def TEST_FUNC_4(x_cl, out_alloc, freq_cl):
    """ Test function, based on the forward operator, version 4.

    In this version, the input x_cl has different dimension order, that must
    coincide with the number of time samples of the problem, to be defined in
    config.T.  Therefore, the kernel matches each positions x_cl at their
    respective time with the respective frequencies at that time.

    This version also requires a different arrangement of the frequency vector.
    Parameters
    ----------
    x_cl: pyopencl.array
        pyopencl.array loaded with input values.
        x_cl correspond to a Nx2xT shaped matrix, representing TxN points in
        R^2, with dype default_type with T = config.T
    out_alloc : mem_alloc class
        a collection of two pyopencl.arrays of shape NxTxK, representing 
        the real and imaginary part of the output of this function.
    freq_cl : pyopencl.array
        Available frequencies, with shape 2xTxK
    Returns
    -------
    None, out_alloc is modified instead.

    Notes
    -----
    Possibly accelerable, by fine tuning and array ordering.
    """
    assert x_cl.dtype == default_type == out_alloc.dtype
    assert isinstance(out_alloc, mem_alloc)
    assert len(out_alloc) == 2
    assert x_cl.shape[0] == out_alloc.shape[0]  # matching N dimension
    assert x_cl.shape[1] == freq_cl.shape[0] == 2  # dimension 2
    assert x_cl.shape[2] ==  out_alloc.shape[1] == freq_cl.shape[1] == config.T  # T match
    assert out_alloc.shape[2] == freq_cl.shape[2]  # K match 
    N, T, K = out_alloc.shape
    program.TEST_FUNC_4(queue, (K,T,N), None,
                      freq_cl.data,
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
        with dtype default_type
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
    assert x_cl.dtype == default_type
    assert t_cl.dtype == np.int32
    assert isinstance(out_alloc, mem_alloc)
    assert out_alloc.dtype == default_type
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

def GRAD_TEST_FUNC_4(x_cl, out_alloc, freq_cl):
    """ Gradient of the test function, base of the forward operator, version 4.

    Parameters
    ----------
    x_cl : pyopencl.array
        pyopencl.array loaded with input values.
        x_cl correspond to a Nx2xT shaped matrix, representing TxN points in
        R^2, with dype default_type with T = config.T
        pyopencl.arrays loaded with the input values.
    out_alloc: mem_alloc class
        a collection of 4 pyopencl.arrays of shape NxTxK, representing
        the real and imaginary parts of the two partial derivatives of this
        function. In order: out_alloc[0] = real dx, out_alloc[1] = imag dx, 
        out_alloc[2] = real dy, out_alloc[3] = imag dy.
    freq_cl : pyopencl.array
        Available frequencies, with shape 2xTxK
    Returns
    -------
    None, out_alloc is modified instead.
    """
    assert x_cl.dtype == default_type == out_alloc.dtype
    assert isinstance(out_alloc, mem_alloc)
    assert len(out_alloc) == 4
    assert x_cl.shape[0] == out_alloc.shape[0]  # matching N dimension
    assert x_cl.shape[1] == freq_cl.shape[0] == 2  # dimension 2
    assert x_cl.shape[2] ==  out_alloc.shape[1] == freq_cl.shape[1] == config.T  # T match
    assert out_alloc.shape[2] == freq_cl.shape[2]  # K match 
    N, T, K = out_alloc.shape
    program.GRAD_TEST_FUNC_4(queue, (K, T, N), None,
                             freq_cl.data,
                             x_cl.data,
                             out_alloc[0].data, out_alloc[1].data,
                             out_alloc[2].data, out_alloc[3].data)
    return None


#def mat_vec_mul(A_cl, b_cl, output_cl):
#    N = A_cl.shape[0]
#    K = np.int32(A_cl.shape[1])
#    program.mat_vec_mul(queue, (N,), None, K,
#                        A_cl.data, b_cl.data, output_cl.data)
#    return None

def einsum(t_cl, Phi_cl, data_cl, output_cl, wait_for=None):
    """ Sum happening when parallelizing the K_t operator.

    Phi_cl of dimension TxNxK, data_cl of TxK. This function multiplies along
    the "t" defined by t_cl, and sums along the K dimension.
    i.e. A_{t,n,k} , g_{t,k} -> O_{t,n} = Σ_{k} A_{t,n,k} g_{t,k}

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

    Notes
    -----
    There is plenty of room to accelerate this function.
    """
    assert t_cl.dtype == np.int32
    assert Phi_cl.dtype == data_cl.dtype == output_cl.dtype ==  default_type
    assert t_cl.shape[0] == Phi_cl.shape[0] == output_cl.shape[0]  # T matching
    assert data_cl.shape[0] == config.T  # TT matching
    assert Phi_cl.shape[1] == output_cl.shape[1]
    assert Phi_cl.shape[2] == data_cl.shape[1]
    N = Phi_cl.shape[1]
    T = t_cl.shape[0]
    K = np.int32(Phi_cl.shape[2])
    program.einsum(queue, (N,T), None, K, t_cl.data, 
                   Phi_cl.data, data_cl.data, output_cl.data,
                   wait_for=wait_for)
    return None

def mat_vec_mul(weights_cl, Phi_cl, output_cl):
    """ Sum reduction happening when parallelizing the K_t^* operator.

    Phi_cl is of dimension TxNxK, weights of Nx1, we want to compute
    an output is an array of dimension TxK, where we do a matrix like
    multiplication along the dimension of size N

    Parameters
    ----------
    weights: pyopencl.array
        1-dimensional array of doubles, representing the weights of the
        atoms of a measure. It has size (N,)
    Phi_cl: pyopencl.array
        3-dimensional array of doubles, of shape (T, N, K), representing the
        output of the function TEST_FUNC. N represents the number o
        2-dimensional evaluation points, and K represents the dimension of the
        image Hilbert space.
    output_cl: pyopencl.array
        2-dimensional buffer to be overwritten by this method, of shape (T,K).
        The output will be added to the current value! i.e. if output_cl has
        already some values, this process will add up new values.
    Returns
    -------
    None

    Notes
    -----
    There is plenty of room to accelerate this function.
    """
    assert Phi_cl.dtype == weights_cl.dtype == output_cl.dtype == default_type
    assert Phi_cl.shape[0] == output_cl.shape[0]  # T matching
    assert Phi_cl.shape[2] == output_cl.shape[1]  # K matching
    assert weights_cl.shape[0] == Phi_cl.shape[1]  # N matching
    T, N, K = Phi_cl.shape
    N = np.int32(N)
    program.mat_vec_mul(queue, (T,K), None, N,
                   Phi_cl.data, weights_cl.data, output_cl.data)
    return None
    
def slice_mat(index_cl, mat_cl, axis=0):
    """ Slice a matrix mat_cl, along the given axis at the given indexes

    Parameters
    ----------
    index_cl : pyopencl.array
        1 dimensional array of indexes of dtype np.int32
    mat_cl : pyopencl.array
        d dimensional array of indexes of dtype default_type
    axis : int
    
    Returns
    -------
    pyopencl.array


    Notes: Appears to be a useles function for the moment. It will stay idle.
    (it was being used by K_t_star, but it is not required for K_t_star_full)
    """
    assert index_cl.dtype == np.int32
    dims = mat_cl.shape
    # 
    pre_axis = dims[:axis]
    pos_axis = dims[axis+1:]
    # For index calculation, we require their product
    prod_pre = np.prod(pre_axis)
    prod_pos = np.prod(post_axis)
    # indexes are
    pass


def reduce_last_dim(mat_cl, out_cl):
    """ Takes a high dimensional array and sums along the last dimension.
    """
    T = np.int32(mat_cl.shape[-1]) 
    N = np.int32(np.prod(mat_cl.shape[:-1]))
    program.reduce_last_dim(queue, (N,), None,
                            T, mat_cl.data, out_cl.data)
    pass

def sumGPU(array_cl, out_cl, work_group=None):
    """ Sums the elements of the input array"""
    if work_group is None:
        work_group = device.max_work_group_size  #1024 in my Tesla 40c
    elif work_group > device.max_work_group_size:
        raise Exception("The input work_group is too big")
    unit_bytes = np.array(0).astype(default_type).nbytes
   # max_local_mem_size = device.local_mem_size/1024*1000 #bytes
   # # We gather the number of used bytes per value
   # unit_bytes = np.array(0).astype(default_type).nbytes
   # # Maximual local array size
   # max_local_size = max_local_mem_size/unit_bytes
   # # Safe max local array size (if there is memory used without noticing)
   # safe_max_local_size = max_local_size/2

    real_length = np.int32(len(array_cl))
    number_work_groups = np.int32(np.ceil(real_length/work_group))
    global_size = number_work_groups*work_group
    #
    program.sumGPU(queue, (global_size, ), (work_group, ),
                   real_length,
                   array_cl.data, out_cl.data,
                   cl.LocalMemory(work_group*unit_bytes))
    # 
    real_length = number_work_groups
    while real_length > 1:
        number_work_groups = np.int32(np.ceil(real_length/work_group))
        global_size = number_work_groups*work_group
        # 
        program.sumGPU(queue, (global_size, ), (work_group, ),
                       real_length,
                       out_cl.data, out_cl.data,
                       cl.LocalMemory(work_group*unit_bytes))
        real_length = number_work_groups
    return None

def sumGPUb(array_cl, out_cl, work_group=None):
    """ Sums the elements of the input array"""
    if work_group is None:
        work_group = device.max_work_group_size  #1024 in my Tesla 40c
    elif work_group > device.max_work_group_size:
        raise Exception("The input work_group is too big")
    unit_bytes = np.array(0).astype(default_type).nbytes
   # max_local_mem_size = device.local_mem_size/1024*1000 #bytes
   # # We gather the number of used bytes per value
   # unit_bytes = np.array(0).astype(default_type).nbytes
   # # Maximual local array size
   # max_local_size = max_local_mem_size/unit_bytes
   # # Safe max local array size (if there is memory used without noticing)
   # safe_max_local_size = max_local_size/2

    real_length = np.int32(len(array_cl))
    number_work_groups = np.int32(np.ceil(real_length/work_group))
    global_size = number_work_groups*work_group
    #
    program.sumGPUb(queue, (global_size, ), (work_group, ),
                   real_length,
                   array_cl.data, out_cl.data,
                   cl.LocalMemory(work_group*unit_bytes))
    # 
    real_length = number_work_groups
    while real_length > 1:
        number_work_groups = np.int32(np.ceil(real_length/work_group))
        global_size = number_work_groups*work_group
        # 
        program.sumGPUb(queue, (global_size, ), (work_group, ),
                       real_length,
                       out_cl.data, out_cl.data,
                       cl.LocalMemory(work_group*unit_bytes))
        real_length = number_work_groups
    return None

def sumGPUb_2D(array_cl, out_cl, work_group=None):
    """ Sums the elements of the given 2D matrix along axis=1.

    It works by iterating the  reduction kernel. The sums are stored in the
    first column of the out_cl buffer, all the other values should be ignored.

    Parameters
    ----------
    array_cl: pyopencl.array
        2 dimensional array to be summed.
    out_cl : pyopencl.array
        2-dimensional buffer to store the sums.
    work_group : int
        Size of each group acting in each row. The bigger, the better.
    """
    if work_group is None:
        work_group = device.max_work_group_size

    assert 1 < work_group, "The work group has to be bigger than 1"
    assert work_group <= device.max_work_group_size,\
            "The input work_group is bigger than the size allowed by the GPU"

    # To allocate local memory
    unit_bytes = np.array(0).astype(default_type).nbytes

    #
    data_width = np.prod(array_cl.shape[1:]).astype(np.int32)
    interest_width = data_width
    height = np.int32(array_cl.shape[0])
    
    work_groups_per_row = np.int32(np.ceil(interest_width/work_group))
    row_global_size = work_groups_per_row*work_group
    #
    program.sumGPUb_2D(queue, (row_global_size, height), (work_group, 1),
                   data_width, interest_width,
                   array_cl.data, out_cl.data,
                   cl.LocalMemory(work_group*unit_bytes))
    #
    interest_width = work_groups_per_row
    while interest_width > 1:
        work_groups_per_row = np.int32(np.ceil(interest_width/work_group))
        row_global_size = work_groups_per_row*work_group
        # 
        program.sumGPUb_2D(queue, (row_global_size, height), (work_group, 1),
                       data_width, interest_width,
                       out_cl.data, out_cl.data,
                       cl.LocalMemory(work_group*unit_bytes))
        interest_width = work_groups_per_row
    return None

def broadcasted_multiplication(array1_cl, array2_cl, out_cl,
                               dims_broadcast, dims_multiply):
    """ Multiplication with broadcasting along the first coordinate.

    Parameters
    ----------
    array1_cl : pyopencl.clarray
        array of shape dims_broadcast x dims_multiply
    array2_cl : pyopencl.clarray
        array of shape dims_multiply
    out_cl : pyopencl.buffer
        with shape dims_broadcast x dims_multiply
    dims_broadcast : tuple(int)
        tuple indicating the dimensions to broadcast from
    dims_multiply: tuple(int)
        tuple indicating the dimensions to multiply from
    """
    assert array1_cl.shape == out_cl.shape == dims_broadcast + dims_multiply
    assert array2_cl.shape == dims_multiply
    program.broadcast_multiplication(queue, (np.prod(dims_multiply),
                                             np.prod(dims_broadcast)), None,
                                     array1_cl.data, array2_cl.data,
                                     out_cl.data)
    return None


def W_operator(evaluations_cl, data_cl, memory_cl=[mem_alloc()]):
    """ The full H product, in both time and frequencies.

    Parameters
    ----------
    evaluations_cl : mem_alloc
        with length 2 and shape NxTxK, representing the evaluation of the
        kernel in a family of N curves, with the real and complex parts.
    data_cl : mem_alloc
        with length 2 and shape TxK, representing the current data in H, with
        real and complex parts.
    memory_cl : mem_alloc
        with length 2 and shape NxTxK, memory used for computations.

    Returns
    -------
    numpy 1-dimensional array of length N.
    """
    assert evaluations_cl.shape[1:] == data_cl.shape
    N, T, K = evaluations_cl.shape
    assert T == config.T and K == config.K
    # setting the statically allocated memory with the correct shape
    if memory_cl[0].shape != evaluations_cl.shape:
        memory_cl[0] = mem_alloc(num=2, shape=evaluations_cl.shape)
    # multiplying real part
    broadcasted_multiplication(evaluations_cl[0], data_cl[0], memory_cl[0][0],
                               (N,), (T,K))
    # multiplying imaginary part
    broadcasted_multiplication(evaluations_cl[1], data_cl[1], memory_cl[0][1],
                               (N,), (T,K))
    # summing real part with complex part
    memory_cl[0][0] = memory_cl[0][0] + memory_cl[0][1]
    # Reduction along dimensions KxT. memory_cl[1] will be used as computation
    # buffer.
    sumGPUb_2D(memory_cl[0][0], memory_cl[0][1], work_group=K*T)
    # The sums are stored in the first column of memory_cl
    return take_column(memory_cl[0][1])/K/T


def take_column(array_cl, idx_buff=[None], out_buff=[None]):
    """ Function to get the first column of a pyopencl.array object in CPU

    This function uses as static variables idx_buff and out_buff, these
    are remembered between executions of this function.
    """

    width = np.prod(array_cl.shape[1:]).astype(np.int32)
    height = array_cl.shape[0]
    if idx_buff[0] is None or idx_buff[0].shape[0] != height:
        idx_buff[0] = clarray_init(width*np.arange(height),
                                   astype=np.int32)
        out_buff[0] = clarray_empty((height,))
    clarray.take(array_cl, idx_buff[0], out_buff[0])
    return out_buff[0].get()

def H1_seminorm_squared(curves_cl, buff=[None]):
    """ Computes the H1 seminorm squared of each element of the given family of curves

    Parameters
    ----------
    curves_cl : pyopencl.array
        Nx2xT shaped array, representing N 2-dimensional curves in time T.
    Returns
    -------
    (N,) shaped numpy array.
    """
    N, _, T = curves_cl.shape
    if buff[0] is None or buff[0].shape[0] != N:
        buff[0] = clarray_empty((N,))
    workgroup = T
    assert workgroup <= device.max_work_group_size,\
            "T is too big, subdivide the work-groups"
    unit_bytes = np.array(0).astype(default_type).nbytes
    program.H1_seminorm(queue, (workgroup, 2, N), (workgroup, 2, 1),
                        curves_cl.data, buff[0].data,
                        cl.LocalMemory(2*workgroup*unit_bytes))
    return buff[0].get()*(T-1)

def F(curves_cl, data_cl, eval_buff = [None]):
    """ Computes the F value on a group of curves 

    Parameters
    ----------
    curves_cl : pyopencl.array
        (N,2,T) shaped array representing N 2-dimensional curves on T samples
    data_cl : mem_alloc
        length 2, (T,K) shaped memory allocation

    Returns
    -------
    (N,) shaped numpy array.
    """
    N, _, T = curves_cl.shape
    K = data_cl.shape[1]
    # Making sure the allocated space is adequate
    if eval_buff[0] is None or eval_buff[0].shape != (N, T, K):
        eval_buff[0] = mem_alloc(2, (N, T, K))
    # Evaluating kernel and storing in the evaluation buffer
    TEST_FUNC_4(curves_cl, eval_buff[0], config.freq_cl)
    return W_operator(eval_buff[0], data_cl)\
        / (config.beta/2*H1_seminorm_squared(curves_cl) + config.alpha)

def grad_L(curves_cl, out_cl):
    """ Computes the gradient of the L function

    Parameters
    ----------
    curves_cl : pyopencl.array
        (N, 2, T) shaped array representing N 2-dimensional curves on T samples
    out_cl : pyopencl.array
        (N, 2, T) shaped array representing the gradient. 
    """
    assert curves_cl.dtype == out_cl.dtype == default_type
    assert curves_cl.shape == out_cl.shape
    N, _, T = curves_cl.shape

    workgroup = T
    assert workgroup <= device.max_work_group_size, \
            "T is too big, subdivide the work-groups"
    unit_bytes = np.array(0).astype(default_type).nbytes
    beta = np.array([config.beta]).astype(default_type)
    program.grad_L(queue, (T, 2, N), (T, 1, 1),
                   beta, curves_cl.data,
                   out_cl.data,
                   cl.LocalMemory(workgroup*unit_bytes))
    return None

def grad_W(curves_cl, data_cl, out_cl, grad_eval_buff=[None]):
    """ Computes the gradient of the W function

    Parameters
    ----------
    curves_cl : pyopencl.array
        (N, 2, T) shaped array representing N 2-dimensional curves on T samples
    data_cl : mem_alloc
        length 2 (T,K)-shaped arrays representing a complex input data. 
    out_cl : mem_alloc
        (N, 2, T) shaped array, to store the output
    grad_eval_buff : static to not use

    Returns
    -------
    None
    """
    assert curves_cl.dtype == data_cl.dtype == out_cl.dtype == default_type
    assert curves_cl.shape == out_cl.shape  # (N, 2, T) match
    assert data_cl.shape[0] == curves_cl.shape[2]  # T match
    assert len(data_cl) == 2 # real/imag parts
    #
    T, K = data_cl.shape
    N = curves_cl.shape[0]
    if grad_eval_buff[0] is None or grad_eval_buff[0].shape != (N, T, K):
        grad_eval_buff[0] = mem_alloc(4, (N, T, K))

    GRAD_TEST_FUNC_4(curves_cl, grad_eval_buff[0], config.freq_cl)

    workgroup = K
    assert workgroup <= device.max_work_group_size, \
            "K is too big, subdivide the work-groups"
    unit_bytes = np.array(0).astype(default_type).nbytes
    program.grad_W(queue, (K, T, N*2), (K, 1, 1),
                   data_cl[0].data, data_cl[1].data,
                   grad_eval_buff[0][0].data, grad_eval_buff[0][1].data,
                   grad_eval_buff[0][2].data, grad_eval_buff[0][3].data,
                   out_cl.data,
                   cl.LocalMemory(workgroup*unit_bytes))


# Release data buffer
if __name__ == '__main__':
    pass
