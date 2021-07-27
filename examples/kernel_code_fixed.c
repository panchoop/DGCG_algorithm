__kernel void einsum(const int K,
                     __global const int *times,
                     __global const double *A,
                     __global const double *B,
                     __global double *C)
{
    // TODO: really optimize this function. There is plenty of room for improvement
    size_t x_id = get_global_id(0);
    int t = get_global_id(1);
    size_t N = get_global_size(0);
    int time = times[t];
    double out = 0;
    int i;
    for (i = 0; i<K; i++)
    {
        out += A[t*N*K + x_id*K + i]*B[time*K + i];
    }
    C[t*N + x_id] += out;
    //printf("\nt = %d", t);
    //printf("\nx_id = %d", x_id);
    //printf("\nout = %f", out);
    //printf("\ntime = %d", time);

}

    


__kernel void mat_vec_mul(const int K,
                          __global const double *A,
                          __global const double *b,
                          __global double *c)
{
    int k;
    int i = get_global_id(0);

    float tmp = 0;
    for (k = 0; k<K ; k++)
    {
        // printf("temp = %f, a = %f, b = %f\n", tmp, A[i*K+k], b[k]);
        tmp += A[i*K + k]*b[k];
    }
    c[i] = tmp;
}


__kernel void K_t_operator(__global const double *freqs,
                           const int K,
                           __global const int *times,
                           __global const double *xs,
                           __global const double *f_t,
                           __global double *output)
{
    size_t x_id = get_global_id(0);
    size_t t = get_global_id(1);
    size_t freq = get_global_id(2);
    int time = times[t];
    size_t N = get_global_size(0);
    //size_t K = get_global_size(2);
    //
    double freq1 = freqs[2*K*time + 2*freq];
    double freq2 = freqs[2*K*time + 2*freq + 1];
    double x1 = xs[2*x_id];
    double x2 = xs[2*x_id + 1];
    //
    cfloat out = 0;
    for (int freq=0; freq<K; ++freq)
    {
        freq1 = freqs[2*K*time + 2*freq];
        freq2 = freqs[2*K*time + 2*freq + 1];
        out += test_func(x1, x2, freq1, freq2);
    }

}
