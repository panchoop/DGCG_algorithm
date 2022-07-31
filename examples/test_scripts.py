# Standard imports
import sys
import os
import numpy as np
import time
import timeit
import pyopencl.array as clarray
import torch as th

# Import package from sibling folder
sys.path.insert(0, os.path.abspath('..'))
from src import DGCG

# Shared code on the test files

T = 51
TIME_SAMPLES = np.linspace(0, 1, T)
FREQ_DIMENSION = np.ones(T, dtype=int)*20

def Archimedian_spiral(t, a, b):
    """ Archimedian spiral to get the frequency measurements"""
    return np.array([(a+b*t)*np.cos(t), (a+b*t)*np.sin(t)])


FREQ_SAMPLES = np.array([Archimedian_spiral(t, 0, 0.2)
                         for t in np.arange(FREQ_DIMENSION[0])])
FREQUENCIES = np.array([FREQ_SAMPLES + t for t in range(T)])  # at each time sample
#  Store in the config file
DGCG.config.K = FREQ_DIMENSION[0]
DGCG.config.freqs_np = FREQUENCIES
DGCG.config.freqs_cl = FREQUENCIES

# DGCG.config.freq_cp = cp.array(FREQUENCIES)

K = FREQ_DIMENSION[0]
def cut_off(s):
    cutoff_threshold = 0.1
    transition = lambda s: 10*s**3 - 15*s**4 + 6*s**5
    val = np.zeros(s.shape)
    for idx, s_i in enumerate(s):
        if cutoff_threshold <= s_i <= 1-cutoff_threshold:
            val[idx] = 1
        elif 0 <= s_i < cutoff_threshold:
            val[idx] = transition(s_i/cutoff_threshold)
        elif 1-cutoff_threshold < s_i <= 1:
            val[idx] = transition(((1-s_i))/cutoff_threshold)
        else:
            val[idx] = 0
    return val

def D_cut_off(s):
    cutoff_threshold = 0.1
    D_transition = lambda s: 30*s**2 - 60*s**3 + 30*s**4
    val = np.zeros(s.shape)
    for idx, s_i in enumerate(s):
        if 0 <= s_i < cutoff_threshold:
            val[idx] = D_transition(s_i/cutoff_threshold)/cutoff_threshold
        elif 1-cutoff_threshold < s_i <= 1:
            val[idx] = -D_transition((1-s_i)/cutoff_threshold)/cutoff_threshold
        else:
            val[idx] = 0
    return val

def TEST_FUNC(t, x):  # φ_t(x)
    input_fourier = x@FREQUENCIES[t].T  # size (N,K)
    fourier_evals = np.exp(-2*np.pi*1j*input_fourier)
    cutoff = cut_off(x[:, 0:1])*cut_off(x[:, 1:2])
    return fourier_evals*cutoff

def GRAD_TEST_FUNC(t, x):  # ∇φ_t(x)
    freq_t = FREQUENCIES[t]
    input_fourier = x@freq_t.T  # size (N,K)
    fourier_evals = np.exp(-2*np.pi*1j*input_fourier)
    # cutoffs
    cutoff_1 = cut_off(x[:, 0:1])
    cutoff_2 = cut_off(x[:, 1:2])
    D_cutoff_1 = D_cut_off(x[:, 0:1])
    D_cutoff_2 = D_cut_off(x[:, 1:2])
    D_constant_1 = -2*np.pi*1j*cutoff_1@freq_t[:, 0:1].T + D_cutoff_1
    D_constant_2 = -2*np.pi*1j*cutoff_2@freq_t[:, 1:2].T + D_cutoff_2
    # wrap it up
    output = np.zeros((2, x.shape[0], FREQ_DIMENSION[t]), dtype='complex')
    output[0] = fourier_evals*cutoff_2*D_constant_1
    output[1] = fourier_evals*cutoff_1*D_constant_2
    return output

DGCG.operators.TEST_FUNC = TEST_FUNC
DGCG.operators.GRAD_TEST_FUNC = GRAD_TEST_FUNC
DGCG.operators.H_DIMENSIONS = FREQ_DIMENSION




# for timeits
setup_0 = """
import numpy as np
from src import DGCG

T = 51
TIME_SAMPLES = np.linspace(0, 1, T)
FREQ_DIMENSION = np.ones(T, dtype=int)*20

def Archimedian_spiral(t, a, b):
    return np.array([(a+b*t)*np.cos(t), (a+b*t)*np.sin(t)])


FREQ_SAMPLES = np.array([Archimedian_spiral(t, 0, 0.2)
                         for t in np.arange(FREQ_DIMENSION[0])])
FREQUENCIES = np.array([FREQ_SAMPLES + t for t in range(T)])  # at each time sample
#  Store in the config file
DGCG.config.K = FREQ_DIMENSION[0]
K = FREQ_DIMENSION[0]
def cut_off(s):
    cutoff_threshold = 0.1
    transition = lambda s: 10*s**3 - 15*s**4 + 6*s**5
    val = np.zeros(s.shape)
    for idx, s_i in enumerate(s):
        if cutoff_threshold <= s_i <= 1-cutoff_threshold:
            val[idx] = 1
        elif 0 <= s_i < cutoff_threshold:
            val[idx] = transition(s_i/cutoff_threshold)
        elif 1-cutoff_threshold < s_i <= 1:
            val[idx] = transition(((1-s_i))/cutoff_threshold)
        else:
            val[idx] = 0
    return val

def D_cut_off(s):
    cutoff_threshold = 0.1
    D_transition = lambda s: 30*s**2 - 60*s**3 + 30*s**4
    val = np.zeros(s.shape)
    for idx, s_i in enumerate(s):
        if 0 <= s_i < cutoff_threshold:
            val[idx] = D_transition(s_i/cutoff_threshold)/cutoff_threshold
        elif 1-cutoff_threshold < s_i <= 1:
            val[idx] = -D_transition((1-s_i)/cutoff_threshold)/cutoff_threshold
        else:
            val[idx] = 0
    return val

def TEST_FUNC(t, x):  # φ_t(x)
    input_fourier = x@FREQUENCIES[t].T  # size (N,K)
    fourier_evals = np.exp(-2*np.pi*1j*input_fourier)
    cutoff = cut_off(x[:, 0:1])*cut_off(x[:, 1:2])
    return fourier_evals*cutoff

def GRAD_TEST_FUNC(t, x):  # ∇φ_t(x)
    freq_t = FREQUENCIES[t]
    input_fourier = x@freq_t.T  # size (N,K)
    fourier_evals = np.exp(-2*np.pi*1j*input_fourier)
    # cutoffs
    cutoff_1 = cut_off(x[:, 0:1])
    cutoff_2 = cut_off(x[:, 1:2])
    D_cutoff_1 = D_cut_off(x[:, 0:1])
    D_cutoff_2 = D_cut_off(x[:, 1:2])
    D_constant_1 = -2*np.pi*1j*cutoff_1@freq_t[:, 0:1].T + D_cutoff_1
    D_constant_2 = -2*np.pi*1j*cutoff_2@freq_t[:, 1:2].T + D_cutoff_2
    # wrap it up
    output = np.zeros((2, x.shape[0], FREQ_DIMENSION[t]), dtype='complex')
    output[0] = fourier_evals*cutoff_2*D_constant_1
    output[1] = fourier_evals*cutoff_1*D_constant_2
    return output
"""
# Tests


def test_0():
    """ Hardware specs"""
    import pyopencl as cl

    print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
    # Print each platform on this computer
    for platform in cl.get_platforms():
        print('=' * 60)
        print('Platform - Name:  ' + platform.name)
        print('Platform - Vendor:  ' + platform.vendor)
        print('Platform - Version:  ' + platform.version)
        print('Platform - Profile:  ' + platform.profile)
        # Print each device per-platform
        for device in platform.get_devices():
            print('    ' + '-' * 56)
            print('    Device - Name:  ' + device.name)
            print('    Device - Type:  ' + cl.device_type.to_string(device.type))
            print('    Device - Max Clock Speed:  {0} Mhz'.format(device.max_clock_frequency))
            print('    Device - Compute Units:  {0}'.format(device.max_compute_units))
            print('    Device - Local Memory:  {0:.0f} KB'.format(device.local_mem_size/1024.0))
            print('    Device - Constant Memory:  {0:.0f} KB'.format(device.max_constant_buffer_size/1024.0))
            print('    Device - Global Memory: {0:.0f} GB'.format(device.global_mem_size/1073741824.0))
            print('    Device - Max Buffer/Image Size: {0:.0f} MB'.format(device.max_mem_alloc_size/1048576.0))
            print('    Device - Max Work Group Size: {0:.0f}'.format(device.max_work_group_size))
    print('\n')


def test_1():
    """ Testing that the implementation works"""
    # 
    N = 1
    # x = np.array([[0.93697702, 0.47514297]])
    for _ in range(100):
        x = np.random.rand(N,2)

        x_cl = DGCG.opencl_mod.clarray_init(x)

        freq_cl = DGCG.opencl_mod.clarray_init(FREQUENCIES)
        DGCG.config.freq_cl = freq_cl
        
        times = np.array([7,4,10])
        T = len(times)
        t_cl =  DGCG.opencl_mod.clarray_init(times, astype=np.int32)
         
        # initialize output buffers
        out_alloc = DGCG.opencl_mod.mem_alloc(2, (T,N,K))
        
        DGCG.opencl_mod.TEST_FUNC(x_cl, t_cl, out_alloc)

        out_alloc_grad = DGCG.opencl_mod.mem_alloc(4, (T,N,K))

        DGCG.opencl_mod.GRAD_TEST_FUNC(x_cl, t_cl, out_alloc_grad)
        
        res2 = []
        for t in times:
            res2.append(GRAD_TEST_FUNC(t,x))

        # Comparisons
        diff = []
        for t in range(T):
            diff.append([out_alloc_grad[0].get()[t] - np.real(res2[t][0]),
                         out_alloc_grad[1].get()[t] - np.imag(res2[t][0]),
                         out_alloc_grad[2].get()[t] - np.real(res2[t][1]),
                         out_alloc_grad[3].get()[t] - np.imag(res2[t][1])
                        ])

        diff_norm = []
        for t in range(T):
            diff_norm.append([np.linalg.norm(diff[t][0]),
                              np.linalg.norm(diff[t][1]),
                              np.linalg.norm(diff[t][2]),
                              np.linalg.norm(diff[t][3]) ])
        if np.max(diff_norm) > 0.001:
            print("x ist :", x_cl.get())
            for t in range(T):
                print(diff_norm[t])
            import code; code.interact(local=dict(globals(), **locals()))

def test_1_2():
    """ Testing that the implementation works"""
    # 
    N = 1
    # x = np.array([[0.93697702, 0.47514297]])
    for _ in range(100):
        x = np.random.rand(3, N, 2)

        x_cl = DGCG.opencl_mod.clarray_init(x)

        freq_cl = DGCG.opencl_mod.clarray_init(FREQUENCIES)
        DGCG.config.freq_cl = freq_cl
        
        times = np.array([4,  7, 1])
        T = len(times)
        t_cl =  DGCG.opencl_mod.clarray_init(times, astype=np.int32)
         
        # initialize output buffers
        out_alloc = DGCG.opencl_mod.mem_alloc(2, (T,N,K))
        
        DGCG.opencl_mod.TEST_FUNC_2(x_cl, t_cl, out_alloc)
        
        # Compare to the actual implementation
        res2 = []
        for idx, t in enumerate(times):
            res2.append(TEST_FUNC(t,x[idx]))

        # Comparisons
        diff = []
        for t in range(T):
            diff.append([out_alloc[0].get()[t] - np.real(res2[t][0]),
                         out_alloc[1].get()[t] - np.imag(res2[t][0])
                        ])

        diff_norm = []
        for t in range(T):
            diff_norm.append([np.linalg.norm(diff[t][0]),
                              np.linalg.norm(diff[t][1])
                              ])
        if np.max(diff_norm) > 0.001:
            print("x ist :", x_cl.get())
            for t in range(T):
                print(diff_norm[t])
            import code; code.interact(local=dict(globals(), **locals()))



def test_2():
    """Testing the einsum function"""
    T = 51
    N = 3
    K = 4
    t = np.array([1])
    t_cl = DGCG.opencl_mod.clarray_init(t , astype=np.int32)
    Phi_cl = DGCG.opencl_mod.clarray_init(np.random.rand(len(t),N,K))
    data_cl = DGCG.opencl_mod.clarray_init(np.random.rand(T, K))
    output_cl = DGCG.opencl_mod.clarray_empty((len(t),N))
    DGCG.opencl_mod.einsum(t_cl, Phi_cl, data_cl, output_cl)

    print(output_cl)

    # Comparing with a numpy execution
    Phi = Phi_cl.get()
    data = data_cl.get()
    output = np.zeros((len(t),N))
    for tt in range(len(t)):
        ttt = t[tt]
        for n in range(N):
            for k in range(K):
                output[tt,n] += Phi[tt,n,k]*data[ttt,k]
    print(output)

    import code; code.interact(local=dict(globals(), **locals()))

def test_3():
    """Testing that the implemented K_t functions coincide"""
    # Compare to the actual implementation

    DGCG.operators.TEST_FUNC = TEST_FUNC
    DGCG.operators.H_DIMENSIONS = FREQ_DIMENSION

    # Generating random data
    f_t_real = np.random.rand(T,K)
    f_t_imag = np.random.rand(T,K)
    # f_t_real = np.ones((T,K))
    # f_t_imag = np.zeros((T,K))
    f_t = f_t_real + 1j*f_t_imag
    # 
    tt = np.array([1,6])
    x = np.random.rand(1,2)
    #  Compute the original implementation
    res_orig = []
    for t in tt:
        res_orig.append(DGCG.operators.K_t(t,f_t)(x))

    print(res_orig)

    #  Testing with pyopencl implementation
    #  first initialize the memory space
    freq_cl = DGCG.opencl_mod.clarray_init(FREQUENCIES)
    DGCG.config.freq_cl = freq_cl
    
    
    t_cl = DGCG.opencl_mod.clarray_init(tt, astype=np.int32)
    x_cl = DGCG.opencl_mod.clarray_init(x)
    
    data_real_cl = DGCG.opencl_mod.clarray_init(f_t_real)
    data_imag_cl = DGCG.opencl_mod.clarray_init(f_t_imag)
    data_alloc = DGCG.opencl_mod.mem_alloc(0,0)
    data_alloc.append(data_real_cl, data_imag_cl)
    # 
    dummy_alloc = DGCG.opencl_mod.mem_alloc(2, (t_cl.shape[0], x_cl.shape[0], K))
    #
    out_cl = DGCG.opencl_mod.clarray_empty((t_cl.shape[0], x_cl.shape[0]))
    # Now apply the function

    funct = DGCG.operators.K_t_cl(data_alloc, dummy_alloc)
    funct(t_cl, x_cl, out_cl)
    print(out_cl)
    import code; code.interact(local=dict(globals(), **locals()))

    
    import code; code.interact(local=dict(globals(), **locals()))

def test_3_2():
    """ Comparing execution speeds of the K_t functions.
    In the original implementation, K_t is used along t = 0,... 50
    in curves.integrate_against. Therefore we test in this case.
    """

    setup = """if 1:
    DGCG.operators.TEST_FUNC = TEST_FUNC
    DGCG.operators.H_DIMENSIONS = FREQ_DIMENSION

    # Generating random data
    f_t_real = np.random.rand(T,K)
    f_t_imag = np.random.rand(T,K)
    # f_t_real = np.ones((T,K))
    # f_t_imag = np.zeros((T,K))
    f_t = f_t_real + 1j*f_t_imag
    # 
    tt = np.arange(51)
    x = np.random.rand(100,2)
    # 
    #  Testing with pyopencl implementation
    #  first initialize the memory space
    freq_cl = DGCG.opencl_mod.clarray_init(FREQUENCIES)
    DGCG.config.freq_cl = freq_cl
    
    
    t_cl = DGCG.opencl_mod.clarray_init(tt, astype=np.int32)
    x_cl = DGCG.opencl_mod.clarray_init(x)
    
    data_real_cl = DGCG.opencl_mod.clarray_init(f_t_real)
    data_imag_cl = DGCG.opencl_mod.clarray_init(f_t_imag)
    data_alloc = DGCG.opencl_mod.mem_alloc(0,0)
    data_alloc.append(data_real_cl, data_imag_cl)
    # 
    dummy_alloc = DGCG.opencl_mod.mem_alloc(2, (t_cl.shape[0], x_cl.shape[0], K))
    #
    out_cl = DGCG.opencl_mod.clarray_empty((t_cl.shape[0], x_cl.shape[0]))
    # define functions
    def np_style():
        for t in tt:
            DGCG.operators.K_t(t, f_t)(x)
    
    def cl_style():
        funct = DGCG.operators.K_t_cl(data_alloc, dummy_alloc)
        funct(t_cl, x_cl, out_cl)
    """
    import timeit
    cpu = timeit.timeit('np_style()', setup=setup_0 + '\n' + setup, number=1000)
    gpu = timeit.timeit('cl_style()', setup=setup_0 + '\n' + setup, number=1000)

    print("Cpu time: ", cpu)
    print("Gpu time: ", gpu)
    print("speedup: {}".format(cpu/gpu))
    import code; code.interact(local=dict(globals(), **locals()))
    """ Results: for N = 1,   ~6     times faster,
                     N = 10,  ~20    times faster,
                     N = 100, ~215   times faster!"""

def test_4():
    """Testing that the implemented ∇K_t functions coincide"""
    # Compare to the actual implementation

    DGCG.operators.TEST_FUNC = TEST_FUNC
    DGCG.operators.GRAD_TEST_FUNC = GRAD_TEST_FUNC
    DGCG.operators.H_DIMENSIONS = FREQ_DIMENSION

    # Generating random data
    f_t_real = np.random.rand(T,K)
    f_t_imag = np.random.rand(T,K)
    # f_t_real = np.ones((T,K))
    # f_t_imag = np.zeros((T,K))
    f_t = f_t_real + 1j*f_t_imag
    # 
    tt = np.array([1,6,4,8])
    x = np.random.rand(3,2)
    # 
    res_orig = []
    for t in tt:
        res_orig.append(DGCG.operators.grad_K_t(t,f_t)(x))

    print(res_orig)

    #  Testing with pyopencl implementation
    #  first initialize the memory space
    freq_cl = DGCG.opencl_mod.clarray_init(FREQUENCIES)
    DGCG.config.freq_cl = freq_cl
    
    
    t_cl = DGCG.opencl_mod.clarray_init(tt, astype=np.int32)
    x_cl = DGCG.opencl_mod.clarray_init(x)
    
    data_real_cl = DGCG.opencl_mod.clarray_init(f_t_real)
    data_imag_cl = DGCG.opencl_mod.clarray_init(f_t_imag)
    data_alloc = DGCG.opencl_mod.mem_alloc(0,0)
    data_alloc.append(data_real_cl, data_imag_cl)
    # 
    dummy_alloc = DGCG.opencl_mod.mem_alloc(4, (t_cl.shape[0], x_cl.shape[0], K))
    #
    out_buff = DGCG.opencl_mod.mem_alloc(2, (t_cl.shape[0], x_cl.shape[0]))
    
    # Now apply the function
    funct = DGCG.operators.grad_K_t_cl(data_alloc, dummy_alloc)
    funct(t_cl, x_cl, out_buff)
    print(out_buff)

    # To compare, the indexes are a bit... changed
    t = 0
    n = 0
    derivative = 0
    for t in range(len(tt)):
        for n in range(x.shape[0]):
            for derivative in [0,1]:
                print("t = {}, n = {}, derivative = {}".format(t, n, derivative))
                print(res_orig[t][derivative][n][0])
                print(out_buff[derivative][t][n])

    import code; code.interact(local=dict(globals(), **locals()))

def test_4_2():
    """Comparing execution times of the old and new ∇K_t functions """
    # Compare to the actual implementation
    setup = """if 1:
    DGCG.operators.TEST_FUNC = TEST_FUNC
    DGCG.operators.GRAD_TEST_FUNC = GRAD_TEST_FUNC
    DGCG.operators.H_DIMENSIONS = FREQ_DIMENSION
    f_t_real = np.random.rand(T,K)
    f_t_imag = np.random.rand(T,K)
    f_t = f_t_real + 1j*f_t_imag
    tt = np.arange(51)
    N = 100
    x = np.random.rand(N,2)
    freq_cl = DGCG.opencl_mod.clarray_init(FREQUENCIES)
    DGCG.config.freq_cl = freq_cl
    t_cl = DGCG.opencl_mod.clarray_init(tt, astype=np.int32)
    x_cl = DGCG.opencl_mod.clarray_init(x)
    data_real_cl = DGCG.opencl_mod.clarray_init(f_t_real)
    data_imag_cl = DGCG.opencl_mod.clarray_init(f_t_imag)
    data_alloc = DGCG.opencl_mod.mem_alloc(0,0)
    data_alloc.append(data_real_cl, data_imag_cl)
    dummy_alloc = DGCG.opencl_mod.mem_alloc(4, (t_cl.shape[0], x_cl.shape[0], K))
    out_buff = DGCG.opencl_mod.mem_alloc(2, (t_cl.shape[0], x_cl.shape[0]))

    def np_style():
        res_orig = []
        for t in tt:
            res_orig.append( DGCG.operators.grad_K_t(t, f_t)(x))

    def cl_style():
        funct = DGCG.operators.grad_K_t_cl(data_alloc, dummy_alloc)
        funct(t_cl, x_cl, out_buff)
    """
    import timeit
    cpu = timeit.timeit('np_style()', setup=setup_0 + '\n' + setup, number=1000)
    gpu = timeit.timeit('cl_style()', setup=setup_0 + '\n' + setup, number=1000)

    print("Cpu time: ", cpu)
    print("Gpu time: ", gpu)
    print("speedup: {}".format(cpu/gpu))
    import code; code.interact(local=dict(globals(), **locals()))
    """ Results: for N = 1,   ~6.5     times faster,
                     N = 10,  ~18    times faster,
                     N = 100, ~160   times faster!"""
    
def test_6():
    """ Testing mat_vec_mul method """
    T = 2
    N = 3
    K = 4

    weights = np.array([1,2,3])
    phi = np.arange(T*N*K).reshape((T,N,K))
    weights_cl = DGCG.opencl_mod.clarray_init(weights)
    Phi_cl = DGCG.opencl_mod.clarray_init(phi)
    output_cl = DGCG.opencl_mod.clarray_empty((T,K))

    print(Phi_cl)
    print(weights)

    DGCG.opencl_mod.mat_vec_mul(weights_cl, Phi_cl, output_cl)

    print(output_cl)
    import code; code.interact(local=dict(globals(), **locals()))

def test_7():
    """ Testing that TEST_FUNC_3 is correctly implemented """
    N = 2
    x = np.random.rand(T, N, 2)
    x_cl = DGCG.opencl_mod.clarray_init(x)

    freq_cl = DGCG.opencl_mod.clarray_init(FREQUENCIES)
    DGCG.config.freq_cl = freq_cl
    
    out_alloc = DGCG.opencl_mod.mem_alloc(2, (T,N,K))
    DGCG.opencl_mod.TEST_FUNC_3(x_cl, out_alloc)
    # Testing with the real value
    val = []
    for t in range(DGCG.config.T):
        for idx in range(N):
            xx = x[t][idx].reshape(1,2)
            val.append(TEST_FUNC(t, xx))
    val = np.array(val).reshape(T,N,K)

    diff1 = out_alloc[0].get() - np.real(val)
    diff2 = out_alloc[1].get() - np.imag(val)

    print(np.linalg.norm(diff1))
    print(np.linalg.norm(diff2))

    import code; code.interact(local=dict(globals(), **locals()))

def test_8():
    """ Testing that K_t_star_full_cl is correctly implemented """
    DGCG.operators.TEST_FUNC = TEST_FUNC
    DGCG.operators.GRAD_TEST_FUNC = GRAD_TEST_FUNC
    DGCG.operators.H_DIMENSIONS = FREQ_DIMENSION
    freq_cl = DGCG.opencl_mod.clarray_init(FREQUENCIES)
    DGCG.config.freq_cl = freq_cl
    #
    N = 2   # 2 curves
    curves = np.random.rand(T,N,2)
    curves_cl = DGCG.opencl_mod.clarray_init(curves)
    weights = np.array([1, 1])
    weights_cl = DGCG.opencl_mod.clarray_init(weights)
    rho_cl = DGCG.classes.measure_cl()
    rho_cl.curves = curves_cl
    rho_cl.weights = weights_cl
    # Allocating buffers
    buff_alloc = DGCG.opencl_mod.mem_alloc(2, (T,N,K) )
    output_alloc = DGCG.opencl_mod.mem_alloc(2, (T,K) )
    # 
    DGCG.operators.K_t_star_full_cl(rho_cl, buff_alloc, output_alloc)

    # Comparing with original implmenetation
    rho = DGCG.classes.measure()
    curve1 = DGCG.classes.curve(curves[0])
    curve2 = DGCG.classes.curve(curves[1])
    rho.add(curve1, weights[0])
    rho.add(curve2, weights[1])

    orig_res = DGCG.operators.K_t_star_full(rho)

    t = 0
    k = 0

    print("Original value")
    print(orig_res[t][k])

    print("New value")
    print(output_alloc[0].get()[t][k], output_alloc[1].get()[t][k])

    import code; code.interact(local=dict(globals(), **locals()))

def test_8_2():
    """ Simple execution test to see if K_t_star_full is faster in GPU"""
    setup = """if 1:
    DGCG.operators.TEST_FUNC = TEST_FUNC
    DGCG.operators.GRAD_TEST_FUNC = GRAD_TEST_FUNC
    DGCG.operators.H_DIMENSIONS = FREQ_DIMENSION
    freq_cl = DGCG.opencl_mod.clarray_init(FREQUENCIES)
    DGCG.config.freq_cl = freq_cl
    #
    N = 50   # 2 curves
    curves = np.random.rand(T,N,2)
    curves_cl = DGCG.opencl_mod.clarray_init(curves)
    weights = np.random.rand(N)
    weights_cl = DGCG.opencl_mod.clarray_init(weights)
    rho_cl = DGCG.classes.measure_cl()
    rho_cl.curves = curves_cl
    rho_cl.weights = weights_cl
    # Allocating buffers
    buff_alloc = DGCG.opencl_mod.mem_alloc(2, (T,N,K) )
    output_alloc = DGCG.opencl_mod.mem_alloc(2, (T,K) )
    # 
    def cl_style():
        DGCG.operators.K_t_star_full_cl(rho_cl, buff_alloc, output_alloc)

    # Comparing with original implmenetation
    rho = DGCG.classes.measure()
    curve1 = DGCG.classes.curve(curves[0])
    curve2 = DGCG.classes.curve(curves[1])
    rho.add(curve1, weights[0])
    rho.add(curve2, weights[1])

    def np_style():
        DGCG.operators.K_t_star_full(rho)"""
    import timeit
    cpu = timeit.timeit('np_style()', setup=setup_0 + '\n' + setup, number=1000)
    gpu = timeit.timeit('cl_style()', setup=setup_0 + '\n' + setup, number=1000)
    import code; code.interact(local=dict(globals(), **locals()))
    """ Results: gpu is ~11x faster than cpu -> success! """



def test_9():
    """Test 1 dimensional sum Richard style"""
    T = 51
    N = 10

    x = np.random.rand(N,T)
    x_cl = DGCG.opencl_mod.clarray_init(x)
    out_cl = DGCG.opencl_mod.clarray_empty((N))
    setup = """if 1:
    T = 51
    N = 10
    import numpy as np
    from src import DGCG
    x = np.random.rand(N,T)
    x_cl = DGCG.opencl_mod.clarray_init(x)
    out_cl = DGCG.opencl_mod.clarray_empty((N))
    def np_style():
        return np.sum(x,1)

    def cl_style():
        DGCG.opencl_mod.reduce_last_dim(x_cl, out_cl)
    """
    
    def np_style():
        return np.sum(x,1)

    def cl_style():
        DGCG.opencl_mod.reduce_last_dim(x_cl, out_cl)

    import timeit
    cpu = timeit.timeit('np_style()', setup=setup)
    gpu = timeit.timeit('cl_style()', setup=setup)

    import code; code.interact(local=dict(globals(), **locals()))


def test_10():
    """ Testing the curves_cl implementation """
    T = 51
    N = 2 

    x = np.random.rand(T, N, 2)
    x1 = x[:,0, :].squeeze()
    x2 = x[:,1, :].squeeze()
    curves = DGCG.classes.curves_cl(x)

    curve1 = DGCG.classes.curve(x1)  # old curve implementation
    curve2 = DGCG.classes.curve(x2)  # old curve implementation
    
    """ Testing the H1 seminorm """
    assert np.linalg.norm(curves.H1_seminorm() - 
                  np.array([curve1.H1_seminorm(), curve2.H1_seminorm()])) < 1e-5
    """ Testing the eval function """
    t = 0.33
    assert np.linalg.norm(curves.eval(t) - 
                          np.vstack((curve1.eval(t), curve2.eval(t)))) < 1e-5
    

def test_11():
    """ Testing the concatenation of clarrays 
    
    A complete disaster. I am now moving to torch, to see if this speedup could
    be achieved in an easier way
    """
    a = np.random.rand(2,3,4)
    b = np.random.rand(2,3,4)
    a_cl = DGCG.opencl_mod.clarray_init(a)
    b_cl = DGCG.opencl_mod.clarray_init(b)
    
    a_t_cl = clarray.transpose(a_cl, axes=[1,0,2])
    b_t_cl = clarray.transpose(b_cl, axes=[1,0,2])

    c_t_cl = clarray.concatenate( (a_t_cl, b_t_cl), axis = 0)

    c_cl = clarray.transpose(c_cl, axes=[0,1])
    
    print(c_cl)
    print("*************************************************")
    print(c_cl.shape)

    import code; code.interact(local=dict(globals(), **locals()))


def test_12():
    """ Comparing the different implementations of the forward operator.
    Comparing here the torch implementation against the OpenCl one.
    """
    DGCG.config.frequencies = DGCG.torch_mod.tensor(FREQUENCIES)

    N = 1
    T = 51

    x = np.random.rand(N,2)
    x_th = DGCG.torch_mod.tensor(x)
    x_alloc = DGCG.torch_mod.empty((N,2))
    
    times = DGCG.torch_mod.tensor([True]*T, dtype=bool)
    
    out_re_alloc = DGCG.torch_mod.empty((T, 20, N))
    out_im_alloc = DGCG.torch_mod.empty((T, 20, N))
    
    if 0:
        DGCG.torch_mod.TEST_FUNC(times, x_th, x_alloc, out_re_alloc, out_im_alloc)
        """ Lets compare that the values are correct"""
        res2 = []
        for t in range(51):
            res2.append(TEST_FUNC(t,x))
        # res2 
        for t in range(51):
            for n in range(N):
                for k in range(20):
                    diff_re = np.real(res2[t][n][k]) - out_re_alloc[t][k][n].cpu().detach().numpy()
                    diff_im = np.imag(res2[t][n][k]) - out_im_alloc[t][k][n].cpu().detach().numpy()
                    if np.abs(diff_re) > 1e-4:
                        print(f"Too high error real for t ={t}, n={n}, k={k}")
                    if np.abs(diff_im) > 1e-4:
                        print(f"Too high error imaginary for t ={t}, n={n}, k={k}")
    # Comparing execution times
    setup_1 = """if 1:
    N = 2
    T = 51
    x = np.random.rand(N,2)
    """
    setup_th = """if 1:
    x_th = DGCG.torch_mod.tensor(x)
    x_alloc = DGCG.torch_mod.empty((N,2))
    
    t_th = DGCG.torch_mod.tensor([True]*T, dtype=bool)
    
    out_re_alloc = DGCG.torch_mod.empty((T, 20, N))
    out_im_alloc = DGCG.torch_mod.empty((T, 20, N))
    freqs = DGCG.config.frequencies[t_th,:,:]
    def th_style():
        DGCG.torch_mod.TEST_FUNC(freqs, x_th, x_alloc, out_re_alloc, out_im_alloc)
    """
    setup_cl = """if 1:
    x_cl = DGCG.opencl_mod.clarray_init(x)
    freq_cl = DGCG.opencl_mod.clarray_init(FREQUENCIES)
    DGCG.config.freq_cl = freq_cl
    t = np.arange(51)
    t_cl = DGCG.opencl_mod.clarray_init(t, astype=np.int32)
    out_alloc = DGCG.opencl_mod.mem_alloc(2, (T,N,K))
    def cl_style():
        DGCG.opencl_mod.TEST_FUNC(x_cl, t_cl, out_alloc)
        DGCG.opencl_mod.queue.finish()
        
    """
    NUM = 5000
    test_th = timeit.timeit('th_style()', setup = setup_0 + '\n' + setup_1 +
                            '\n' + setup_th, number=NUM)
    test_cl = timeit.timeit('cl_style()', setup = setup_0 + '\n' + setup_1 +
                            '\n' + setup_cl, number=NUM)
    print("Torch   : ", test_th/NUM)
    print("Opencl  : ", test_cl/NUM)

def test_13():
    """ Testing how slow is the reduction along a dimension for the H1 norm.
    Also testing the speed.
    Results
    -------
    Numpy is much faster than the opencl implementation. Even at high N
    values, there is no comparison.

    It appears that the worst offenders of these methods are the slicings
    required to compute the difference, as well as the reduction.
    """
    if 0:  # just the matching test
        N = 1
        K = 20
        T = 51

        curves = np.arange(N*2*T).reshape(T,N,2)
        
        # curves = np.random.rand(T,N,2)
        curves_cl = DGCG.classes.curves_cl(curves)

        cl_sol = curves_cl.H1_norm_cl()
        np_sol = curves_cl.H1_seminorm() 

        print(cl_sol)
        print(np_sol)
    # The speed test
    setup_1 = """if 1:
    N = 800
    T = 51
    curves = np.random.rand(T,N,2)
    curves_cl = DGCG.classes.curves_cl(curves)
    def cl_style():
        curves_cl.H1_norm_cl()
        DGCG.opencl_mod.queue.finish()

    def np_style():
        curves_cl.H1_seminorm()
    """

    NUM = 5000
    test_np = timeit.timeit('np_style()', setup = setup_0 + '\n' + setup_1,
                            number=NUM)
    test_cl = timeit.timeit('cl_style()', setup = setup_0 + '\n' + setup_1,
                            number=NUM)
    print("Numpy   : ", test_np/NUM)
    print("Opencl  : ", test_cl/NUM)


    import code; code.interact(local=dict(globals(), **locals()))

def test_14():
    """ numpy parallelized forward operator. Comparing to opencl implementation
    
    The Opencl implementation blows it out. It is better for N = 1, and the
    difference increases inmensely for bigger N. Even if one includes a copy
    back to CPU command .get() to the output.
    """
    if 0: 
        N = 10
        T = 51
        K = 20

        curves = np.random.rand(T, N ,2 )

        evals = DGCG.numpy_mod.TEST_FUNC(curves)


        curves_cl = DGCG.opencl_mod.clarray_init(curves)

        freq_cl = DGCG.opencl_mod.clarray_init(FREQUENCIES)
        DGCG.config.freq_cl = freq_cl
        
        out_alloc = DGCG.opencl_mod.mem_alloc(2, (T,N,K))
        DGCG.opencl_mod.TEST_FUNC_3(curves_cl, out_alloc)
        # Testing with the real value

        diff1 = out_alloc[0].get() - np.real(evals)
        diff2 = out_alloc[1].get() - np.imag(evals)

        print(np.linalg.norm(diff1))
        print(np.linalg.norm(diff2))
        import code; code.interact(local=dict(globals(), **locals()))

    """ Testing speed"""
    setup_1 = """if 1:
    N = 100
    T = 51
    K = 20

    curves = np.random.rand(T, N, 2)
    curves_cl = DGCG.opencl_mod.clarray_init(curves)
    freq_cl = DGCG.opencl_mod.clarray_init(FREQUENCIES)
    DGCG.config.freq_cl = freq_cl
    DGCG.config.freqs_cl = FREQUENCIES

    out_alloc = DGCG.opencl_mod.mem_alloc(2, (T,N,K))

    def cl_style():
        DGCG.opencl_mod.TEST_FUNC_3(curves_cl, out_alloc)
        DGCG.opencl_mod.queue.finish()
        out_alloc[0].get()
        out_alloc[1].get()

    def np_style():
        DGCG.numpy_mod.TEST_FUNC(curves)
    """
    NUM = 5000
    test_np = timeit.timeit('np_style()', setup = setup_0 + '\n' + setup_1,
                            number=NUM)
    test_cl = timeit.timeit('cl_style()', setup = setup_0 + '\n' + setup_1,
                            number=NUM)
    print("Numpy   : ", test_np/NUM)
    print("Opencl  : ", test_cl/NUM)

def test_15():
    """ Testing the sum function """
    if 0:
        """ Testing successfull """
        for i in range(1000):
            length = np.random.randint(10**7)
            # length = 1025  # mersenne prime number
            a = np.random.rand(length)
            a_cl = DGCG.opencl_mod.clarray_init(a)
            out = np.zeros(length)
            out_cl = DGCG.opencl_mod.clarray_init(out)
            #
            DGCG.opencl_mod.sumGPUb(a_cl, out_cl)

            if np.abs(np.sum(a) - out_cl[0]) > 1e-5:
                print("Not good sum")
            import code; code.interact(local=dict(globals(), **locals()))

    """ Let's test speed.
    Results
    -------
    For [1, 100,000] lenght number numpy is definitely the winner, around 10-100 
    times faster. The native pyopencl implementation is for [1,100] around x10
    times faster, but around [1000, 100000],it is around 2x, 3x times faster 
    than my implementation.

    For [100,000 - 1,000,000] the three implementations are in the same order
    with native pyopencl tied with numpy, both faster than mine. Around 3x 
    difference between the best and worse.

    For [1,000,000 - 10,000,000] numpy is the slowest, followed closely
    by mine, and with the fastest being the native implementation.

    Between the two own implementations, it is hard to tell the best one, it 
    strongly variates. In general, the "b" implementation wins. 

    Regarding work_group sizes. In general, the bigger, the better.

    """
    setup_1 = """if 1:
    import pyopencl.array as clarray
    power = {}
    length = np.random.randint(10**power, 10**(power+1))
    a = np.random.rand(length)
    a_cl = DGCG.opencl_mod.clarray_init(a)
    out_cl = DGCG.opencl_mod.clarray_empty((length,))
    #
    def np_style():
        np.sum(a)

    def cl_style():
        DGCG.opencl_mod.sumGPU(a_cl, out_cl, work_group={})
        DGCG.opencl_mod.queue.finish()

    def cl2_style():
        DGCG.opencl_mod.sumGPUb(a_cl, out_cl, work_group={})
        DGCG.opencl_mod.queue.finish()
        
    def pyopencl_style():
        clarray.sum(a_cl)
        DGCG.opencl_mod.queue.finish()
        
    """
    test_np = 0
    test_cl = 0
    test_cl_2 = 0
    test_pyopencl = 0
    for power in [6, 7]:
        for work_group in [64, 126, 250, 256, 500, 512, 1000, 1024]:
            full_setup = setup_0 + '\n' + setup_1.format(power, work_group,
                                                         work_group)
            print("Power: ", power)
            print("work_group: ", work_group)

            N = 1000
            NUM = 100
            test_cl = []
            test_cl_2 = []

            length = 6504424
            a = np.random.rand(length)
            a_cl = DGCG.opencl_mod.clarray_init(a)
            out_cl = DGCG.opencl_mod.clarray_empty((length,))
            DGCG.opencl_mod.sumGPU(a_cl, out_cl, work_group=work_group)
            DGCG.opencl_mod.queue.finish()

            for _ in range(N):
                # test_np += timeit.timeit('np_style()', setup = setup_0 + '\n' + setup_1,
                #                        number=NUM)
                test_cl.append(timeit.timeit('cl_style()', setup=full_setup,
                                             number=NUM))
                test_cl_2.append(timeit.timeit('cl2_style()', setup=full_setup,
                                               number=NUM))
                # test_pyopencl += timeit.timeit('pyopencl_style()', setup = setup_0 + '\n' + setup_1,
                #                        number=NUM)
            # print("Numpy         : ", test_np/NUM/N)
            print("Power : ", power)
            print("work_group : ", work_group)
            print("Opencl  average: ", np.mean(test_cl))
            print("Opencl2 average: ", np.mean(test_cl_2))
            print("Opencl  std :", np.std(test_cl))
            print("Opencl2 std :", np.std(test_cl_2))
            print("Opencl  max :", np.max(test_cl))
            print("Opencl2 max :", np.max(test_cl_2))
            print("Opencl  min :", np.min(test_cl))
            print("Opencl2 min :", np.min(test_cl_2))
            # print("Opencl_native : ", test_pyopencl/NUM/N)


def test_16():
    """ Testing implementation of the 2D sum kernel"""
    def test_implementation(width, height):
        """ Testing successfull """
        work_group = np.random.randint(2, 1024)
        print("width, height, work_group = ({}, {}, {})".format(width, height,
                                                                work_group))
        a = np.random.rand(height, width)
        a_cl = DGCG.opencl_mod.clarray_init(a)
        out_cl = DGCG.opencl_mod.clarray_empty((height, width))
        #
        DGCG.opencl_mod.sumGPUb_2D(a_cl, out_cl, work_group=work_group)
        a_sum = np.sum(a, axis=1)
        out_row = out_cl.get()[:, 0].reshape(-1)

        for i in range(len(a_sum)):
            if abs(a_sum[i] - out_row[i]) > 1e-5:
                print("Something went wrong!")
                import code; code.interact(local=dict(globals(), **locals()))
        DGCG.opencl_mod.queue.finish()

    if 0:
        max_memory_allocation = 2560  # mb
        data_type = DGCG.opencl_mod.default_type
        data_type_size = np.array(0).astype(data_type).nbytes
        max_array_size = max_memory_allocation*10**6/data_type_size
        
        for _ in range(1000):
            max_10_power = int(np.floor(np.log10(max_array_size)))
            width_pow = np.random.randint(1, max_10_power)
            width = np.random.randint(10**width_pow, 10**(width_pow + 1))
            height = np.random.randint(1, max(max_array_size//width, 2))
            test_implementation(width, height)

    """ Testing speeds

    Results
    -------
    For fixed width of 20*51 the numpy execution time is more or less
    linear with the height. Wereas the OpenCl is scalonated.

    The minimum OpenCl speed is 0.0148 
    numpy reaches this speed around [2^8, 2^9) = [256, 512]
    then OpenCl is faster.

    """
    setup_1 = """if 1:
    import pyopencl.array as clarray
    width= {}
    height= {}
    a = np.random.rand(height, width)
    a_cl = DGCG.opencl_mod.clarray_init(a)
    out_cl = DGCG.opencl_mod.clarray_empty((height, width))
    #
    def np_style():
        np.sum(a, axis=1)

    def cl_style():
        DGCG.opencl_mod.sumGPU(a_cl, out_cl, work_group=min(width, 1024))
        DGCG.opencl_mod.queue.finish()
    """

    width_power = 3
    height_power = 11
    #
    test_cl = []
    test_np = []
    NUM = 100
    N = 500
    #
    for _ in range(N):
        # width = np.random.randint(10**width_power, 10**(width_power + 1))
        width = 51*20  # T*K
        height = np.random.randint(2**height_power, 2**(height_power + 1))
        #  print("width, height : ({}, {})".format(width, height))
        full_setup = setup_0 + '\n' + setup_1.format(width, height)
        test_np.append(timeit.timeit('np_style()', setup=full_setup,
                                     number=NUM))
        test_cl.append(timeit.timeit('cl_style()', setup=full_setup,
                                     number=NUM))
    print("Powers : ({}, {})".format(width_power, height_power))
    print("Numpy average : ", np.mean(test_np))
    print("Opencl average : ", np.mean(test_cl))

def test_17():
    """ Verifying that TEST_FUNC_4 is well implemented. Tested against
    TEST_FUNC_3

    Results
    -------
    It works and it is twice faster! 
    """
    if 0:
        # Testing that they coincide
        N = 2
        x = np.random.rand(T, N, 2)
        x_cl = DGCG.opencl_mod.clarray_init(x)
        #
        freq_cl = DGCG.opencl_mod.clarray_init(FREQUENCIES)
        DGCG.config.freq_cl = freq_cl
        #
        out_alloc = DGCG.opencl_mod.mem_alloc(2, (T,N,K))
        DGCG.opencl_mod.TEST_FUNC_3(x_cl, out_alloc)
        #
        x4 = np.swapaxes(np.swapaxes(x, 1,2), 0, 2)
        x4_cl = DGCG.opencl_mod.clarray_init(x4)

        FREQUENCIES4 = np.swapaxes(np.swapaxes(FREQUENCIES, 0,2), 1,2)
        freq4_cl = DGCG.opencl_mod.clarray_init(FREQUENCIES4)
        # freq4_cl 

        out_alloc4 = DGCG.opencl_mod.mem_alloc(2, (N,T,K))
        DGCG.opencl_mod.TEST_FUNC_4(x4_cl, out_alloc4, freq4_cl)
        
        for n in range(N):
            for k in range(20):
                for t in range(T):
                    if np.abs(out_alloc[0].get()[t ,n ,k] -
                              out_alloc4[0].get()[n ,t ,k]) > 1e-5:
                        print("real part double")
                    if np.abs(out_alloc[1].get()[t ,n ,k] -
                              out_alloc4[1].get()[n ,t ,k]) > 1e-5:
                        print("imaginary part double")
        import code; code.interact(local=dict(globals(), **locals()))

    # Testing speed. Checking if speedup
    setup_1 = """if 1:
    N = 5
    x = np.random.rand(T, N, 2)
    x_cl = DGCG.opencl_mod.clarray_init(x)
    freq_cl = DGCG.opencl_mod.clarray_init(FREQUENCIES)
    DGCG.config.freq_cl = freq_cl
    out_alloc = DGCG.opencl_mod.mem_alloc(2, (T,N,K))
    x4 = np.swapaxes(np.swapaxes(x, 1,2), 0, 2)
    x4_cl = DGCG.opencl_mod.clarray_init(x4)
    FREQUENCIES4 = np.swapaxes(np.swapaxes(FREQUENCIES, 0,2),1,2)
    freq4_cl = DGCG.opencl_mod.clarray_init(FREQUENCIES4)
    out_alloc4 = DGCG.opencl_mod.mem_alloc(2, (N,T,K))
    def test_1():
        DGCG.opencl_mod.TEST_FUNC_3(x_cl, out_alloc)
        DGCG.opencl_mod.queue.finish()

    def test_2():
        DGCG.opencl_mod.TEST_FUNC_4(x4_cl, out_alloc4, freq4_cl)
        DGCG.opencl_mod.queue.finish()
    """
    print("TEST_FUNC_3 result")
    print(timeit.timeit('test_1()', setup=setup_0 + '\n' + setup_1, number=1000))
    print("TEST_FUNC_4 result")
    print(timeit.timeit('test_2()', setup=setup_0 + '\n' + setup_1, number=1000))

def test_18():
    """ Testing the broadcasted multiplication

    Results
    -------
    For N small (from 1 to 100), the winner is numpy, at some point by 10 times.
    For N around 100-200, numpy and opencl are matched. Above, wins opencl"""
    if 0:
        N = 100
        array1 = np.random.rand(N,K,T)
        array2 = np.random.rand(K,T)
        array1_cl = DGCG.opencl_mod.clarray_init(array1)
        array2_cl = DGCG.opencl_mod.clarray_init(array2)
        out_cl = DGCG.opencl_mod.clarray_empty((N,K,T))
        out_cl2 = DGCG.opencl_mod.clarray_empty((N,K,T))
        # apply
        DGCG.opencl_mod.broadcasted_multiplication(array1_cl, array2_cl, out_cl,
                                                   (N,), (K,T))
        # testing
        np_result = array1*array2
        print("The error is : ")
        print(np.linalg.norm(np_result - out_cl.get()))
        print("The other error is : ")
        print(np.linalg.norm(np_result - out_cl2.get()))
        import code; code.interact(local=dict(globals(), **locals()))
    setup_1 = """if 1:
    N = 1000
    array1 = np.random.rand(N,K,T)
    array2 = np.random.rand(K,T)
    array1_cl = DGCG.opencl_mod.clarray_init(array1)
    array2_cl = DGCG.opencl_mod.clarray_init(array2)
    out_cl = DGCG.opencl_mod.clarray_empty((N,K,T))
    # apply
    def cl_style():
        DGCG.opencl_mod.broadcasted_multiplication(array1_cl, array2_cl,
                                                   out_cl, (N,), (K,T))
        DGCG.opencl_mod.queue.finish()
        DGCG.opencl_mod.queue.finish()

    # testing
    def np_style():
        array1*array2
    """
    print("Opencl style result")
    print(timeit.timeit('cl_style()', setup=setup_0 + '\n' + setup_1, number=10000))
    print("Numpy result result")
    print(timeit.timeit('np_style()', setup=setup_0 + '\n' + setup_1, number=10000))


def test_19():
    """ Testing taking from a clarray

    Results
    -------
    For K = 20, T = 51, the naive method (copy everything and then extract),
    wins for N <= 40.
    For N >= 40, the method using an idx wins, when idx is pre-allocated. 
    If Idx is not pre-allocated, it takes almost double the time.
    """

    if 0:
        N = 100
        x = np.random.rand(N,K,T)
        x_cl = DGCG.opencl_mod.clarray_init(x)

        idx = K*T*np.arange(N)
        idx_cl = DGCG.opencl_mod.clarray_init(idx, astype=np.int32)

        out_cl = DGCG.opencl_mod.clarray_empty((N,))
        

        def idx_style():
            col = clarray.take(x_cl, idx_cl)
            col.get()
            print(col)

        def naive_style():
            col = x_cl.get()[:,0,0]
            print(col)

        def mixed_style():
            DGCG.opencl_mod.take_column(x_cl, out_cl)
            print(out_cl)

        print("Idx style")
        idx_style()
        print("naive style")
        naive_style()
        print("mixed style")
        mixed_style()
        print("mixed 2 style")
        mixed_style()

        import code; code.interact(local=dict(globals(), **locals()))
    #
    setup_1 = """\nif 1:
    import pyopencl.array as clarray
    N = 20
    x = np.random.rand(N,K,T)
    x_cl = DGCG.opencl_mod.clarray_init(x)

    idx = K*T*np.arange(N)
    idx_cl = DGCG.opencl_mod.clarray_init(idx, astype=np.int32)
    out_cl = DGCG.opencl_mod.clarray_empty((N,))
    

    def idx_style():
        col = clarray.take(x_cl, idx_cl)
        col.get()

    def naive_style():
        col = x_cl.get()[:,0,0]

    def own_style():
        DGCG.opencl_mod.take_column(x_cl, out_cl)
        DGCG.opencl_mod.queue.finish()
    """
    print("Idx style result : ")
    print(timeit.timeit('idx_style()', setup=setup_0 + setup_1, number=10000))
    print("naive style result : ")
    print(timeit.timeit('naive_style()', setup=setup_0 + setup_1, number=10000))
    print("own style result : ")
    print(timeit.timeit('own_style()', setup=setup_0 + setup_1, number=10000))

def test_20():
    """ Testing the full W operator """
    N = 100
    
    swap_freq = np.swapaxes(np.swapaxes(FREQUENCIES, 0, 2), 1, 2)
    DGCG.config.freq_cl = DGCG.opencl_mod.clarray_init(swap_freq)
    curves = np.random.rand(N,2,T)
    curves_cl = DGCG.opencl_mod.clarray_init(curves)

    evaluations_cl = DGCG.opencl_mod.mem_alloc(2, (N, T, K))
    DGCG.opencl_mod.TEST_FUNC_4(curves_cl, evaluations_cl, DGCG.config.freq_cl)
    
    real_part = evaluations_cl[0].get()
    imag_part = evaluations_cl[1].get()
    
    #
    real_data = np.random.rand(T,K)
    imag_data = np.random.rand(T,K)
    real_data_cl = DGCG.opencl_mod.clarray_init(real_data)
    imag_data_cl = DGCG.opencl_mod.clarray_init(imag_data)
    data_cl = DGCG.opencl_mod.mem_alloc()
    data_cl.append([real_data_cl, imag_data_cl])
    #
    out_cl = DGCG.opencl_mod.clarray_empty((N,))
    
    print(DGCG.opencl_mod.W_operator(curves_cl, data_cl, out_cl))

    # comparison to original method
    f_list = []
    for n in range(N):
        f_list.append(real_part[n, :, :] + 1j*imag_part[n, :, :])
    g = real_data + 1j*imag_data

    output = []
    for n in range(N):
        output.append(DGCG.operators.int_time_H_t_product(f_list[n], g))

    print(np.array(output))
    

def test_21():
    """ Testing the implemented GPU L_operator"""
    N = 10
    real_curves = []
    stack_curves = []
    for _ in range(N):
        curve = np.random.rand(T,2)
        real_curves.append(DGCG.classes.curve(curve))
        stack_curves.append(curve)

    stack_curves = np.array(stack_curves)
    stack_curves = np.swapaxes(stack_curves,1,2)

    curves_cl = DGCG.opencl_mod.clarray_init(stack_curves)
    # 
    out_cl = DGCG.opencl_mod.clarray_empty((N,))
    
    print( DGCG.opencl_mod.L_operator(curves_cl, out_cl))

    original_H1 = []
    for n in range(N):
        original_H1.append(DGCG.config.beta/2*real_curves[n].H1_seminorm()**2
                           + DGCG.config.alpha)
    print(np.array(original_H1))
    import code; code.interact(local=dict(globals(), **locals()))

def test_22():
    """Testing the F function.

    Results
    -------
    Works like a charm, it is even faster than the GPU in the 1 curve case
    Then, the CPU case slows down linearly on the number of curves, wereas
    the GPU remains much more stable, up to twice the time only for 
    1000 curves.
    """
    swap_freq = np.swapaxes(np.swapaxes(FREQUENCIES, 0, 2), 1, 2)
    DGCG.config.freq_cl = DGCG.opencl_mod.clarray_init(swap_freq)

    if 1:
        N = 4
        real_curves = []
        stack_curves = []
        for _ in range(N):
            curve = np.random.rand(T,2)
            real_curves.append(DGCG.classes.curve(curve))
            stack_curves.append(curve)

        stack_curves = np.array(stack_curves)
        stack_curves = np.swapaxes(stack_curves,1,2)

        curves_cl = DGCG.opencl_mod.clarray_init(stack_curves)

        real_data = np.random.rand(T, K)
        imag_data = np.random.rand(T, K)
        data_cl = DGCG.opencl_mod.mem_alloc()
        data_cl.append( [DGCG.opencl_mod.clarray_init(real_data),
                         DGCG.opencl_mod.clarray_init(imag_data)])
        # 
        out_cl = DGCG.opencl_mod.clarray_empty((N,))
        

        F_cl = DGCG.opencl_mod.F_operator(curves_cl, data_cl, out_cl)

        print(out_cl)
        zero_measure = DGCG.classes.measure()
        w_t = DGCG.classes.dual_variable(zero_measure)
        w_t._data = real_data + 1j*imag_data
        F = []
        for n in range(N):
            F.append(DGCG.optimization.F(real_curves[n], w_t))
        print(np.array(F))
        import code; code.interact(local=dict(globals(), **locals()))

    setup_1 = """\nif 1:
    N = 100
    real_curves = []
    stack_curves = []
    for _ in range(N):
        curve = np.random.rand(T,2)
        real_curves.append(DGCG.classes.curve(curve))
        stack_curves.append(curve)

    stack_curves = np.array(stack_curves)
    stack_curves = np.swapaxes(stack_curves,1,2)

    curves_cl = DGCG.opencl_mod.clarray_init(stack_curves)

    real_data = np.random.rand(T, K)
    imag_data = np.random.rand(T, K)
    data_cl = DGCG.opencl_mod.mem_alloc()
    data_cl.append( [DGCG.opencl_mod.clarray_init(real_data),
                     DGCG.opencl_mod.clarray_init(imag_data)])
    out_cl = DGCG.opencl_mod.clarray_empty((N,))
    

    def gpu_style():
        F_cl = DGCG.opencl_mod.F_operator(curves_cl, data_cl, out_cl)
        DGCG.opencl_mod.queue.finish()
        

    zero_measure = DGCG.classes.measure()
    w_t = DGCG.classes.dual_variable(zero_measure)
    w_t._data = real_data + 1j*imag_data
    
    def cpu_style():
        F = []
        for n in range(N):
            F.append(DGCG.optimization.F(real_curves[n], w_t))
    """

    print("GPU time: ", timeit.timeit("gpu_style()", setup=setup_0 + setup_1,
                                      number=100))
    print("CPU time: ", timeit.timeit("cpu_style()", setup=setup_0 + setup_1,
                                      number=100))

def test_23():
    """Testing the implemented GRAD_TEST_FUNC_4 against the origina ones"""
    # Compare to the actual implementation
    N = 1
    x = np.random.rand(N,2,T)
    x_cl = DGCG.opencl_mod.clarray_init(x)
    
    out_alloc = DGCG.opencl_mod.mem_alloc(4, (N, T, K))
    swap_FREQ = np.swapaxes(np.swapaxes(FREQUENCIES,0,2),1,2)
    freq_cl = DGCG.opencl_mod.clarray_init(swap_FREQ)

    DGCG.opencl_mod.GRAD_TEST_FUNC_4(x_cl, out_alloc, freq_cl)

    # Actual implementation
    t = 0
    np_output = GRAD_TEST_FUNC(t, x[0,:,t].reshape(1,2))  # 2xNxK size

    # comparison
    for t in range(T):
        np_output = GRAD_TEST_FUNC(t, x[0, :, t].reshape(1, 2)) # 2x1xK shaped
        for derivative in [0,1]:
            for k in range(K):
                if np.abs(np.real(np_output[derivative, 0, k]) 
                          - out_alloc[2*derivative][0,t,k]) > 1e-4:
                    print(" No bueno ")
                if np.abs(np.imag(np_output[derivative, 0, k]) 
                          - out_alloc[2*derivative + 1][0,t,k]) > 1e-4:
                    print(" Also No bueno ")

    import code; code.interact(local=dict(globals(), **locals()))

def test_24():
    """ Testing the implemented gradient of the L(x) funtion, the divisor of F
    """

    N = 1
    x = np.random.rand(N,2,T)
    # x = np.array([np.arange(T), np.arange(T)]).reshape(N,2,T)
    x_cl = DGCG.opencl_mod.clarray_init(x)
    out_cl = DGCG.opencl_mod.clarray_empty((N, 2, T))
    DGCG.opencl_mod.grad_L(x_cl, out_cl)

    def grad_L_np(curves):
        diff_positions = np.diff(curves, axis=0)  # γ_{i+1}-γ_{i} (T-1)x2 array
        diff_times = np.diff(DGCG.config.time)   # t_{i+1}-t{i} 1D array
        diffs = np.diag(1/diff_times)@diff_positions  # diff(γ)/diff(t) (T-1)x2 array
        prepend_zero_diffs = np.insert(diffs, 0, 0, axis=0)
        append_zero_diffs = np.insert(diffs, len(diffs), 0, axis=0)
        grad_L_gamma = DGCG.config.beta*(prepend_zero_diffs - append_zero_diffs)
        return grad_L_gamma

    curves = x.reshape(2,T).transpose()
    out_np = grad_L_np(curves)
    out_np[np.abs(out_np) < 1e-10] = 0


    print(out_cl)
    print(out_np.transpose())

    import code; code.interact(local=dict(globals(), **locals()))

def test_25():
    """ Testing the implemented grad_W function """
    N = 1
    T = 51
    #
    swap_FREQ = np.swapaxes(np.swapaxes(FREQUENCIES,0,2),1,2)
    freq_cl = DGCG.opencl_mod.clarray_init(swap_FREQ)
    DGCG.config.freq_cl = freq_cl
    #
    x = np.random.rand(N,2,T)
    x_cl = DGCG.opencl_mod.clarray_init(x)

    data_real = np.random.rand(T,K)
    data_imag = np.random.rand(T,K)
    data_real_cl = DGCG.opencl_mod.clarray_init(data_real)
    data_imag_cl = DGCG.opencl_mod.clarray_init(data_imag)
    
    data_cl = DGCG.opencl_mod.mem_alloc()
    data_cl.append([data_real_cl, data_imag_cl])
    
    out_cl = DGCG.opencl_mod.clarray_empty((N,2,T))
    DGCG.opencl_mod.grad_W(x_cl, data_cl, out_cl)
    # Original implementation
    t_weigh = DGCG.config.time_weights
    rho_empty = DGCG.classes.measure()
    
    w_t = DGCG.classes.dual_variable(rho_empty)
    w_t._data = data_real + 1j*data_imag
    curve = DGCG.classes.curve(x.reshape(2,T).transpose())
    
    
    w_t_curve = lambda t: w_t.grad_eval(t, curve.eval_discrete(t)).reshape(2)
    grad_W_gamma = -np.array([t_weigh[t]*w_t_curve(t) for t in range(T)])

    print(out_cl)

    print(grad_W_gamma.transpose())
    import code; code.interact(local=dict(globals(), **locals()))
    
def test_26():
    """ Testing the gradient of F function"""
    N = 1
    #
    swap_FREQ = np.swapaxes(np.swapaxes(FREQUENCIES,0,2),1,2)
    freq_cl = DGCG.opencl_mod.clarray_init(swap_FREQ)
    DGCG.config.freq_cl = freq_cl
    #
    curves = np.random.rand(N,2,T)
    curves_cl = DGCG.opencl_mod.clarray_init(curves)
    #
    data_real = np.random.rand(T,K)
    data_imag = np.random.rand(T,K)
    data_real_cl = DGCG.opencl_mod.clarray_init(data_real)
    data_imag_cl = DGCG.opencl_mod.clarray_init(data_imag)
    #
    data_cl = DGCG.opencl_mod.mem_alloc()
    data_cl.append([data_real_cl, data_imag_cl])
    # 
    out_cl = DGCG.opencl_mod.clarray_empty((N,2,T))

    DGCG.opencl_mod.grad_F(curves_cl, data_cl, out_cl)
    print(out_cl)
    # testing with the original F function
    real_curve = DGCG.classes.curve(curves.reshape(2,T).transpose())
    
    rho_empty = DGCG.classes.measure()
    
    w_t = DGCG.classes.dual_variable(rho_empty)
    w_t._data = data_real + 1j*data_imag

    print(DGCG.optimization.grad_F(real_curve, w_t).spatial_points.transpose())
        
def test_27():
    """ Testing the implemented H1_norm """
    N = 10
    real_curves = []
    stack_curves = []
    for _ in range(N):
        curve = np.random.rand(T,2)
        real_curves.append(DGCG.classes.curve(curve))
        stack_curves.append(curve)

    stack_curves = np.array(stack_curves)
    stack_curves = np.swapaxes(stack_curves,1,2)

    curves_cl = DGCG.opencl_mod.clarray_init(stack_curves)
    # 
    out_cl = DGCG.opencl_mod.clarray_empty((N,))
    
    DGCG.opencl_mod.H1_norm(curves_cl, out_cl)
    print(out_cl)

    original_L2 = []
    for n in range(N):
        original_L2.append(real_curves[n].H1_norm())
    print(np.array(original_L2))
    import code; code.interact(local=dict(globals(), **locals()))

def test_28():
    """ Testing the implemented vector norm """
    N = 10
    real_curves = []
    stack_curves = []
    for _ in range(N):
        curve = np.random.rand(T,2)
        real_curves.append(DGCG.classes.curve(curve))
        stack_curves.append(curve)

    stack_curves = np.array(stack_curves)
    stack_curves = np.swapaxes(stack_curves,1,2)

    curves_cl = DGCG.opencl_mod.clarray_init(stack_curves)
    # 
    out_cl = DGCG.opencl_mod.clarray_empty((N,))
    
    DGCG.opencl_mod.vector_norm_squared(curves_cl, out_cl)
    print(out_cl)

    original_norm = []
    for n in range(N):
        original_norm.append(np.linalg.norm(real_curves[n].spatial_points)**2)
    print(np.array(original_norm))
    import code; code.interact(local=dict(globals(), **locals()))

def test_29():
    """ Testing the implemented gradient descent
    Speed results
    -------------
    Doing N simultaneous descents for N =1 to around 256 takes the same time.
    Increasing N above this point also increases the execution time, but it 
    is still convenient, for instance, N=1024 took only x3 execution time
    compared to N=256, despite being 4x bigger.
    """
    if 1:
        N = 1000
        curves = np.random.rand(N, 2, T)
        curves_cl = DGCG.opencl_mod.clarray_init(curves)
        #
        swap_FREQ = np.swapaxes(np.swapaxes(FREQUENCIES, 0, 2), 1, 2)
        freq_cl = DGCG.opencl_mod.clarray_init(swap_FREQ)
        DGCG.config.freq_cl = freq_cl
        #
        data_real = np.random.rand(T, K)
        data_imag = np.random.rand(T, K)
        data_real_cl = DGCG.opencl_mod.clarray_init(data_real)
        data_imag_cl = DGCG.opencl_mod.clarray_init(data_imag)
        #
        data_cl = DGCG.opencl_mod.mem_alloc()
        data_cl.append([data_real_cl, data_imag_cl])
        #  Initialize stepsizes and F_values
        stepsizes = DGCG.opencl_mod.clarray_init(np.ones(N)*1)
        F_vals = DGCG.opencl_mod.clarray_empty((N,))
        DGCG.opencl_mod.F_operator(curves_cl, data_cl, F_vals)
        
        iters = 50000
        # F_statistics = np.zeros((N, iters + 1))
        F_statistics = []
        F_statistics.append(F_vals.get())
        stepsize_statistics = []
        stepsize_statistics.append(stepsizes.get())
        grad_F_statistics = []
        grad_F = DGCG.opencl_mod.gradient_descent(curves_cl, data_cl, F_vals, stepsizes)
        for i in range(iters):
            grad_F = DGCG.opencl_mod.gradient_descent(curves_cl, data_cl, F_vals, stepsizes)
            if i % (iters//100) == 0:
                print("At iter: ", i)
                F_statistics.append(F_vals.get())
                stepsize_statistics.append(stepsizes.get())
                grad_F_statistics.append(grad_F.get())
                new_grad = grad_F.get()
                

        F_statistics = np.array(F_statistics).transpose()
        stepsize_statistics = np.array(stepsize_statistics).transpose()
        grad_F_statistics = np.array(grad_F_statistics).transpose()
        import matplotlib.pyplot as plt
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        for i in range(N):
            ax1.semilogy(F_statistics[i, :] - F_statistics[i, -1])
            ax1.set_title("F values")

        for i in range(N):
            ax2.semilogy(stepsize_statistics[i, :])
            ax2.set_title("Stepsizes")

        for i in range(N):
            ax3.semilogy(grad_F_statistics[i, :])
            ax3.set_title("Gradient values")

        fig.suptitle(str(iters) + " iterations")
        plt.savefig("Convergence_graph.pdf")
        plt.close()
        # plt.show(block=False)
    """ Doing some time comparisons"""
    setup_1 = """\nif 1:
    def test(N, iters):
        curves = np.random.rand(N,2,T)
        curves_cl = DGCG.opencl_mod.clarray_init(curves)
        #
        swap_FREQ = np.swapaxes(np.swapaxes(FREQUENCIES,0,2),1,2)
        freq_cl = DGCG.opencl_mod.clarray_init(swap_FREQ)
        DGCG.config.freq_cl = freq_cl
        #
        data_real = np.random.rand(T,K)
        data_imag = np.random.rand(T,K)
        data_real_cl = DGCG.opencl_mod.clarray_init(data_real)
        data_imag_cl = DGCG.opencl_mod.clarray_init(data_imag)
        #
        data_cl = DGCG.opencl_mod.mem_alloc()
        data_cl.append([data_real_cl, data_imag_cl])
        #  Initialize stepsizes and F_values
        stepsizes = DGCG.opencl_mod.clarray_init(np.ones(N)*1)
        F_vals = DGCG.opencl_mod.clarray_empty((N,))
        DGCG.opencl_mod.F_operator(curves_cl, data_cl, F_vals)
        
        # iters = 50000
        for i in range(iters):
            grad_F = DGCG.opencl_mod.gradient_descent(curves_cl, data_cl, F_vals, stepsizes)
    """
    Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for N in Ns:
        print("Execution time for N = "+str(N)+" : ", end='')
        print(timeit.timeit(f"test({N}, 10000)", setup=setup_0+setup_1,
                            number=1))
    import code; code.interact(local=dict(globals(), **locals()))

def test_30():
    """ This is a non-GPU parallel test. Here we evaluate how reasonable is 
    to try to paralellize one of the insertion step parts, the propose method,
    particularly the random insertion method.

    The method will be copied here (minus clutter), as there are some 
    dependencies that need to be isolated.

    Conclusion: Generating around 1000 random curves takes around 23 seconds.
    Although not too much, it would happen at every fun of the gradient descent.
    It is better to see ways to accelerate this
    """

    setup_1 = """\nif 1:
    rho_empty = DGCG.classes.measure()
    data_real = np.random.rand(T,K)
    data_imag = np.random.rand(T,K)
    w_t = DGCG.classes.dual_variable(rho_empty)
    w_t._data = data_real + 1j*data_imag

    min_segments = 1
    max_segments = T-1

    def sample_random_curve(w_t):
        num_segments = np.random.randint(max_segments-min_segments+1)\
                                         + min_segments
        # preset the intermediate random times
        considered_times = [0, T-1]
        while len(considered_times) <= num_segments:
            new_time = np.random.randint(T)
            if not (new_time in considered_times):
                considered_times.append(new_time)
        considered_times = np.sort(np.array(considered_times), -1)
        # times
        positions = DGCG.insertion_step.insertion_mod.rejection_sampling(0, w_t)
        for t in considered_times[1:]:
            positions = np.append(positions, 
           DGCG.insertion_step.insertion_mod.rejection_sampling(t, w_t), 0)
        rand_curve = DGCG.classes.curve(considered_times/(T - 1), positions)
        # discarding any proposed curve that has too much length
        return rand_curve

    def test(w_t):
        for _ in range(1000):
            sample_random_curve(w_t)
    """
        
    print("Testing execution time")
    print(timeit.timeit("test(w_t)", setup=setup_0+setup_1, 
                        number=1))

def test_31():
    """ Testing a different implementation of the sample_random_curve function """
    # 1) Select random number of segments
    # 2) Select random number of times
    # 3) Select random number of spatial points at these times
    # 4) Build the curves
     
    # 1 and 2 could be done at the same time, taking out randomnes.
    # Given a maximum number of segments, we could just generate each posibility
    # equally weighted

    setup_1 = """\nif 1:
    def test_spatial_generation():
        min_segments = 1
        max_segments = T-1
        times_stack = []
        N_curves = 10000
        for _ in range(N_curves):
            num_segments = np.random.randint(max_segments-min_segments+1) \
                           + min_segments
            considered_times = [0, T-1]
            while len(considered_times) <= num_segments:
                new_time = np.random.randint(T)
                if not (new_time in considered_times):
                    considered_times.append(new_time)
            considered_times = np.sort(np.array(considered_times), -1)
            times_stack.append(considered_times)
        return times_stack
    examples = test_spatial_generation()
    """
    print("Testing execution time")
    print(timeit.timeit("test_spatial_generation()", setup=setup_0+setup_1, number=1))
    # Generating these curves is takes 0.16 seconds for N_curves=1000, and 
    # the time is linear on N_curves. Completely neglegible cost

    # 2) Select random number of spatial 
    # parallel on eachtime sample
    #   
    setup_2 = """\nif 1:
    def test_spatial_generation2():
        rho_empty = DGCG.classes.measure()
        data_real = np.random.rand(T,K)
        data_imag = np.random.rand(T,K)
        w_t = DGCG.classes.dual_variable(rho_empty)
        w_t._data = data_real + 1j*data_imag
        for each_example in examples:
            for each_time in each_example:
                support, density_max = w_t.as_density_get_params(each_time)
    """
    print("Testing execution time")
    print(timeit.timeit("test_spatial_generation2()", 
          setup=setup_0+setup_1+setup_2, number=1))
    # Total of 11.2 seconds. High cost but thansk to caching, this time is 
    # expense happens once in each execution of the insertion step (aka, once
    # iteration of the whole algorithm), therefore it is a very minimal cost. 
    # Nonetheless, this cost can certainly be potentially be accelerated,
    # but probably is not worth the effort. TODO
    setup_3= """\n
                M = support*density_max
                iter_reasonable_threshold = 10000
                iter_index = 0
                boolean = False
                while iter_index < iter_reasonable_threshold:
                    reasonable_threshold = 10000
                    i = 0
                    while i < reasonable_threshold:
                        x = np.random.rand()
                        y = np.random.rand()
                        sample = np.array([[x, y]])
                        #y = w_t.as_density_eval(each_time, sample)
                        y = np.random.rand()-0.5
                        if y > 0:
                            break
                        else:
                            i = i + 1
                    if i == reasonable_threshold:
                        sys.exit('It is not able to sample inside the support of w_t')
                    # sample rejection sampling
                    u = np.random.rand()
                    if u < y/M*support:
                        # accept
                        boolean = True
                        break
                    else:
                        # reject
                        iter_index = iter_index+1
                if (not boolean):
                    sys.exit(('The rejection_sampling algorithm failed to find sample in {} ' +
                         'iterations').format(iter_index))"""
    print("Testing execution time")
    print(timeit.timeit("test_spatial_generation2()",
          setup=setup_0+setup_1+setup_2+setup_3, number=1))
    # Gives an additional 11.3 [s] for N_curves=1000, and 125 [s] for N_curves=10000
    # so linear on N_curves, with a rather high cost.
    # when commenting out line "y = w_t.as_density_eval( ...  )" and uncommenting
    # the bottom one, the additional time are:
    # 3.8 [s] for N_curves=1000, and  39.5 [s] for N_curves=1000
    # Therefore, one posibility to accelerate this is the as_density_eval function
    


    
if __name__ == "__main__":
    test_31()
