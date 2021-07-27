// Defining complex variables
typedef float2 cfloat;

inline float real(cfloat a){
    return a.x;
}

inline float img(cfloat a){
    return a.y;
}


double cut_off(double s)
{
    // threshold = 0.1
    if (s > 0.5) s = 1-s;
    if (s <= 0) return 0;
    if (s >= 0.1) return 1;
    return 10*s*s*s/0.001 - 15*s*s*s*s/0.0001 + 6*s*s*s*s*s/0.00001;
}

double dx_cut_off(double s)
{
    int factor = 1;
    if (s > 0.5){
         s = 1-s;
        factor = -1;
    }
    if (s <= 0) return 0;
    if (s >= 0.1) return 0;
    return factor*(30*s*s/0.001 - 60*s*s*s/0.0001 + 30*s*s*s*s/0.00001);
}

cfloat test_func(double x1, double x2, double freq1, double freq2)
{
    double tot_freq = -2*M_PI*(x1*freq1 + x2*freq2);
    double amp = cut_off(x1)*cut_off(x2);
    cfloat output;
    output.x = cos(tot_freq)*amp;
    output.y = sin(tot_freq)*amp;
//    printf("\n *******************\n (x1,x2) = (%f, %f) \n (f1,f2) = (%f, %f) \n output.x = %f",
//           x1, x2, freq1, freq2, output.x);
    return output;
}

cfloat grad1_test_func(double x1, double x2, double freq1, double freq2)
{
    double tot_freq = -2*M_PI*(x1*freq1 + x2*freq2);
    double sinf = sin(tot_freq);
    double cosf = cos(tot_freq);
    double out_freq = 2*M_PI*freq1*cut_off(x1);
    double cut_2 = cut_off(x2);
    double dcut_1 = dx_cut_off(x1);
    cfloat output;
    output.x = cut_2*(dcut_1*cosf + out_freq*sinf);
    output.y = cut_2*(dcut_1*sinf - out_freq*cosf);
    return output;
}

cfloat grad2_test_func(double x1, double x2, double freq1, double freq2)
{
    double tot_freq = -2*M_PI*(x1*freq1 + x2*freq2);
    double sinf = sin(tot_freq);
    double cosf = cos(tot_freq);
    double out_freq = 2*M_PI*freq2*cut_off(x2);
    double cut_1 = cut_off(x1);
    double dcut_2 = dx_cut_off(x2);
    cfloat output;
    output.x = cut_1*(dcut_2*cosf + out_freq*sinf);
    output.y = cut_1*(dcut_2*sinf - out_freq*cosf);
    return output;
}

__kernel void TEST_FUNC(__global const double *freqs,
                        __global const int *times,
                        __global const double *xs,
                        __global double *real_output,
                        __global double *imag_output)
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
//    printf("\n *******************\n (x1,x2) = (%f, %f) \n (f1,f2) = (%f, %f) \n t = %d, time=%d, out.x = %f",
//           x1, x2, freq1, freq2, t, time, out.x);

//    if (x_id == 0 && freq == 0)
//    {
//        printf("x1 = %f\n", x1);
//        printf("x2 = %f\n", x2);
//        printf("freq1 = %f\n", freq1);
//        printf("freq2 = %f\n", freq2);
//        printf("K = %d\n", K);
//        printf("real part = %f\n", out.x);
//        printf("imag part = %f\n", out.y);
//    }
}

__kernel void GRAD_TEST_FUNC(__global const double *freqs,
                             __global const int *times,
                             __global const double *xs,
                             __global double *real_output_1,
                             __global double *imag_output_1,
                             __global double *real_output_2,
                             __global double *imag_output_2)
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
    cfloat out_1 = grad1_test_func(x1, x2, freq1, freq2);
    cfloat out_2 = grad2_test_func(x1, x2, freq1, freq2);
    real_output_1[K*N*t + K*x_id + freq] = out_1.x;
    imag_output_1[K*N*t + K*x_id + freq] = out_1.y;
    real_output_2[K*N*t + K*x_id + freq] = out_2.x;
    imag_output_2[K*N*t + K*x_id + freq] = out_2.y;
//    if (x_id == 0 && freq == 0)
//    {
//        printf("x1 = %f\n", x1);
//        printf("x2 = %f\n", x2);
//        printf("freq1 = %f\n", freq1);
//        printf("freq2 = %f\n", freq2);
//        printf("K = %d\n", K);
//        printf("real part = %f\n", out.x);
//        printf("imag part = %f\n", out.y);
//    }
}

