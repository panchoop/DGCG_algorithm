import cupy as cp
import numpy as np
from . import config

# Get device properties
# cp.cuda.runtime.getDeviceProperties(n)
# The office Tesla is n = 1.


cupy.cuda.Device(1).use()  # switch to GPU 1

minus_2_pi = cp.array(-2*cp.pi)

kernel = cp.RawKernel(r'''
typedef float2 cfloat;

double cut_off(double s)
{
    // threshold = 0.1
    if (s > 0.5) s = 1-s;
    if (s <= 0) return 0;
    if (s >= 0.1) return 1;
    return 10*s*s*s/0.001 - 15*s*s*s*s/0.0001 + 6*s*s*s*s*s/0.00001;
}

cfloat test_func(double x1, double x2, double freq1, double freq2)
{
    double tot_freq = -2*M_PI*(x1*freq1 + x2*freq2);
    double amp = cut_off(x1)*cut_off(x2);
    cfloat output;
    output.x = cos(tot_freq)*amp;
    output.y = sin(tot_freq)*amp;
    return output;
}

extern "C" void TEST_FUNC(const double* freqs,
                        const int* times,
                        const double* xs,
                        double* real_output,
                        double* imag_output)
{
    size_t x_id = get_global_id(0);
    size_t t = get_global_id(1);
    size_t freq = get_global_id(2);
    int time = times[t];
    size_t N = get_global_size(0);
    size_t K = get_global_size(2);
    //
    double freq1 = freqs[2*K*time + 2*freq];
    double freq2 = freqs[2*K*time + 2*freq + 1];
    double x1 = xs[2*x_id];
    double x2 = xs[2*x_id + 1];
    //
    cfloat out = test_func(x1, x2, freq1, freq2);
    real_output[K*N*t + K*x_id + freq] = out.x;
    imag_output[K*N*t + K*x_id + freq] = out.y;
}
''')



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


