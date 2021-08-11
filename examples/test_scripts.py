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
        for work_group in [126, 250, 256, 500, 512, 1000, 1024]:
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
                length = np.random.randint(10**power, 10**(power+1))
                for _ in range(NUM):
                    a = np.random.rand(length)
                    a_cl = DGCG.opencl_mod.clarray_init(a)
                    out_cl = DGCG.opencl_mod.clarray_empty((length,))
                    try:
                        DGCG.opencl_mod.sumGPU(a_cl, out_cl, work_group=work_group)
                        DGCG.opencl_mod.queue.finish()
                    except Exception as e:
                        print(e)
                        import code; code.interact(local=dict(globals(), **locals()))
                    
                for _ in range(NUM):
                    a = np.random.rand(length)
                    a_cl = DGCG.opencl_mod.clarray_init(a)
                    out_cl = DGCG.opencl_mod.clarray_empty((length,))
                    try:
                        DGCG.opencl_mod.sumGPU(a_cl, out_cl, work_group=work_group)
                        DGCG.opencl_mod.queue.finish()
                    except Exception as e:
                        print(e)
                        import code; code.interact(local=dict(globals(), **locals()))

                # test_cl.append(timeit.timeit('cl_style()', setup=full_setup,
                #                              number=NUM))
                # test_cl_2.append(timeit.timeit('cl2_style()', setup=full_setup,
                #                                number=NUM))
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
    if 1:
        """ Testing successfull """
        length = np.random.randint(10)
        # length = 1025  # mersenne prime number
        a = np.random.rand(2, length)
        a_cl = DGCG.opencl_mod.clarray_init(a)
        out_cl = DGCG.opencl_mod.clarray_empty((2, length))
        #
        DGCG.opencl_mod.sumGPUb_2D(a_cl, out_cl, work_group=10)

        import code; code.interact(local=dict(globals(), **locals()))


        
        
    
    
    
if __name__ == "__main__":
    test_15()
