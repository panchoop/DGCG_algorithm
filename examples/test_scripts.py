# Standard imports
import sys
import os
import numpy as np
import time

# Import package from sibling folder
sys.path.insert(0, os.path.abspath('..'))
from src import DGCG

# General simulation parameters
def test_1():
    """ Testing that the implementation works"""
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
    K = FREQ_DIMENSION[0]
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
        
        # Compare to the actual implementation
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


        # Kernels to define the forward measurements
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
    
    def TEST_FUNC(t, x):  # φ_t(x)
        input_fourier = x@FREQUENCIES[t].T  # size (N,K)
        fourier_evals = np.exp(-2*np.pi*1j*input_fourier)
        cutoff = cut_off(x[:, 0:1])*cut_off(x[:, 1:2])
        return fourier_evals*cutoff

    DGCG.operators.TEST_FUNC = TEST_FUNC
    DGCG.operators.H_DIMENSIONS = FREQ_DIMENSION

    # Generating random data
    f_t_real = np.random.rand(T,K)
    f_t_imag = np.random.rand(T,K)
    # f_t_real = np.ones((T,K))
    # f_t_imag = np.zeros((T,K))
    f_t = f_t_real + 1j*f_t_imag
    # 
    tt = np.array([1,6,4,8])
    x = np.random.rand(1,2)
    # 
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

    def cossum():
        out = 0
        for i in range(20):
            val = np.cos(np.pi*(FREQUENCIES[0][i][0] + FREQUENCIES[0][i][1]))
            print(val)
            out += val
        return out/20
    # print(cossum())
    import code; code.interact(local=dict(globals(), **locals()))

def test_4():
    """Testing that the implemented ∇K_t functions coincide"""
    # Compare to the actual implementation
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

    

if __name__ == "__main__":
    test_4()
