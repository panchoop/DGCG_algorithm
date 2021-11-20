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

    


__kernel void mat_vec_mul(const int N,
                          __global const double *Phi,
                          __global const double *weights,
                          __global double *output)
{
    int n;
    size_t t = get_global_id(0);
    size_t k = get_global_id(1);
    size_t K = get_global_size(1);

    float tmp = 0;
    for (n = 0; n<N ; n++)
    {
        tmp += Phi[t*K*N + n*K + k]*weights[n];
    }
    output[t*K +k] = tmp;
}


__kernel void reduce_last_dim(const int T,
                           __global const double *mat_cl,
                           __global double *out_cl)
{
    size_t i = get_global_id(0);
    size_t I = get_global_size(0);
    double temp = 0;
    for (int j = 0; j < T; j++)
    {
        temp += mat_cl[i*T + j];
    }
    out_cl[i] = temp;
}

__kernel void sumGPU_power2(__global const double *input,
                     __global double *partialSums,
                     __local double *localSums)
{
    // sumGPU function that works only for the case in which the input array
    // is a power of 2
    size_t local_id = get_local_id(0);
    size_t group_size = get_local_size(0);
    if (get_global_id(0) == 0)
    {
    printf("Group size %d \n", group_size);
    }

    // Copy from global to local memory
    localSums[local_id] = input[get_global_id(0)];

    // Loop for computing localSums: divide WorkGroup into 2 parts
    for (size_t stride = group_size/2; stride>0; stride /= 2)
    {
        // Waiting for each 2x2 addition into given workgroup
        barrier(CLK_LOCAL_MEM_FENCE);

        // Add elements 2 by 2 between local_id and local_id + stride
        //if (get_global_id(0) == 0)
        //{
        //printf("Stride %d \n", stride);
        //printf("localSums %f \n", localSums[local_id]);
        //printf("localSums + stride %f \n", localSums[local_id + stride]);
        //}
        if (local_id < stride)
            localSums[local_id] += localSums[local_id + stride];
    }
    // Write result into partialSums[nWorkGroups]
    if (local_id == 0)
        partialSums[get_group_id(0)] = localSums[0];

} 

__kernel void sumGPU(const int realLength,
                     __global const double *input,
                     __global double *partialSums,
                     __local double *localSums)
{
    size_t local_id = get_local_id(0);
    size_t group_size = get_local_size(0);
    size_t global_id = get_global_id(0);
    // Copy from global to local memory
    if (global_id < realLength)
    {
        localSums[local_id] = input[global_id];

       // Loop for computing localSums: divide WorkGroup into 2 parts
       size_t preStride = group_size;
       size_t stride = group_size/2;

       while (stride > 0)
       {
           barrier(CLK_LOCAL_MEM_FENCE);
           if (global_id + stride < realLength)
           {
               if (local_id < stride)
                   localSums[local_id] += localSums[local_id + stride];
               if (preStride % 2 == 1 && local_id == stride-1)
                   localSums[local_id] += localSums[local_id + stride + 1];
           }
           preStride = stride;
           stride /= 2;
       }
       if (local_id == 0)
           partialSums[get_group_id(0)] = localSums[0];
   }
} 

__kernel void sumGPUb(const int realLength,
                     __global const double *input,
                     __global double *partialSums,
                     __local double *localSums)
{
    // Version b of this function.
    size_t local_id = get_local_id(0);
    size_t group_size = get_local_size(0);
    size_t global_id = get_global_id(0);
    // Copy from global to local memory
    if (global_id < realLength)
    {
        localSums[local_id] = input[global_id];

        // Loop for computing localSums: divide WorkGroup into 2 parts
        size_t preStride = group_size;
        size_t stride = group_size/2;

        while (stride > 0)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            if (global_id + stride < realLength)
            {
                if (preStride % 2 == 1)
                {
                    stride += 1;
                    if (local_id < stride-1)
                        localSums[local_id] += localSums[local_id+stride];
                    if (local_id == stride)
                        localSums[local_id] = localSums[local_id + stride];
                }
                else
                {
                    if (local_id < stride)
                        localSums[local_id] += localSums[local_id+stride];
                }
            }
            preStride = stride;
            stride /= 2;
        }
        if (local_id == 0 && global_id < realLength)
            partialSums[get_group_id(0)] = localSums[0];
    }
} 

__kernel void sumGPUb_2D(const int dataWidth,
                         const int interestWidth,
                       __global const double *input,
                       __global double *partialSums,
                       __local double *localSums)
{
    // This method will apply one iteration of the 1-dimensional reduction 
    // along the horizontal axis of a matrix. 
    //
    // An iteration will compute the sum of all the elements of a each work
    // group. Each work-group can work in only one row of the matrix.
    // Each work group will write the sum in an entry of localSums
    // therefore, if there are 2 groups in a row, the sum output will have 
    // two values in the respective row in partialSums
    // Parameters:
    //     dataWidth: True width of input array
    //     interestWidth: The actual width used for summing up
    //     input: the input matrix
    //     partialSums: the output values
    //     localSums: space allocated locally in each group.
    size_t local_id = get_local_id(0);
    size_t group_size = get_local_size(0);
    size_t global_id_i = get_global_id(0);
    size_t global_id_j = get_global_id(1);
    // Copy from global to local memory. We don't mind the values beyond the
    // width of interest, those workers are left unused.
    if (global_id_i < interestWidth){
        localSums[local_id] = input[global_id_j*dataWidth + global_id_i];

        // the preStride indicates the size of the local memory to be used 
        // This minimum is to be taken when the last group of a given row
        //   exceeds the number of remaining elements in the width of interest
        size_t preStride = min(group_size,
                               interestWidth - group_size*get_group_id(0));
        // The stride halves, as every worker will sum up to two values
        size_t stride = preStride/2;

        while(stride > 0)
        {
            // Synchronize the workers of each group.
            barrier(CLK_LOCAL_MEM_FENCE);
            // If the preStride was not divisible by 2, we add one to the stride
            stride += preStride % 2;
            // As long as we do not excede the local memory of interest, sum.
            if (local_id + stride < preStride)
                        localSums[local_id] += localSums[local_id + stride];
            // Shrink the local memory of interest
            preStride = stride;
            stride /= 2;
        }
        if (local_id == 0 && global_id_i < interestWidth)
            partialSums[global_id_j*dataWidth + get_group_id(0)] = localSums[0];
    }
}

__kernel void broadcast_multiplication_in_place(__global double *array1,
                                                __global const double *array2)
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t I = get_global_size(0);
    //
    array1[j*I + i] = array1[j*I + i] * array2[i];
}


__kernel void L_operator(const double beta,
                         const double alpha,
                         __global const double *curves,
                         __global double *out,
                         __local double *localStorage)
{
    // curves is of size Nx2xT and we have 2xT work-groups.
    size_t local_id = get_local_id(0);
    size_t local_id_j = get_local_id(1);
    size_t T = get_local_size(0);
    size_t global_id_j = get_global_id(1);
    size_t n = get_global_id(2);
    // Copy the array in local memory
    double private_value = curves[n*T*2 + T*local_id_j + local_id];
    localStorage[T*local_id_j + local_id] = private_value;
    barrier(CLK_LOCAL_MEM_FENCE);
    // Substract the forward element
    if (local_id < T-1){
       private_value = private_value - localStorage[T*local_id_j +local_id + 1];
       private_value = private_value*private_value;
       barrier(CLK_LOCAL_MEM_FENCE);
       localStorage[T*local_id_j + local_id] = private_value;  // store the squares difference
       barrier(CLK_LOCAL_MEM_FENCE);
       if (local_id_j == 0){
           localStorage[local_id] = localStorage[local_id]
                                    + localStorage[T + local_id];
           // Up to this point, all squared differences are stored in the local
           // Storage.
           // We proceed now with the reduction.
           // T-1 are the elements to be summed
           // For the rest, refer to sumGPUb_2D comments.
           size_t preStride = T-1;
           size_t stride = preStride/2;
    //        if (local_id == 0 && global_id_j == 0){
    //            for (int i = 0; i < T-1 ; ++i)
    //                    printf("local storage i: %f\n", localStorage[i]);
    //        }

           while (stride > 0){
               barrier(CLK_LOCAL_MEM_FENCE);
               stride += preStride % 2;

               if (local_id + stride < preStride)
                   localStorage[local_id] += localStorage[local_id + stride];

               preStride = stride;
               stride /= 2;
           }
           if (local_id == 0)
               out[n] = beta*(T-1)*localStorage[0]/2 + alpha;
       }
    }
}

__kernel void grad_L(const double beta,
                     __global const double *curves,
                     __global double *out,
                     __local double *localStorage)
{
    // curves is of size Nx2xT  and we have 1x1xT sized work-groups
    size_t T = get_local_size(0);
    size_t t  = get_global_id(0);
    size_t i  = get_global_id(1);
    size_t n  = get_global_id(2);
    // Copy array in time in local memory
    double private_memory = curves[n*2*T + i*T + t];
    localStorage[t] = private_memory;
    barrier(CLK_LOCAL_MEM_FENCE);
    // No more copying, we can use these values!
    double term1 = 0;
    double term2 = 0;
    if (t<T-1){
        term1 = private_memory - localStorage[t+1];
    }
    if (t>0){
        term2 = private_memory - localStorage[t-1];
    }
    out[n*2*T + i*T + t] = beta*(T-1)*(term1 + term2);
}


__kernel void grad_W(__global const double *data_real,
                     __global const double *data_imag,
                     __global const double *grad_real_1,
                     __global const double *grad_imag_1,
                     __global const double *grad_real_2,
                     __global const double *grad_imag_2,
                     __global double *out,
                     __local double *localStorage)
{
    size_t k = get_global_id(0);
    size_t t = get_global_id(1);
    size_t ni = get_global_id(2);  // ni are two variables together
    size_t n = ni/2;
    size_t i = ni % 2;
    size_t K = get_global_size(0);
    size_t T = get_global_size(1);

    if (i == 0){
        localStorage[k] = grad_real_1[n*T*K + t*K + k]*data_real[t*K + k] 
                          + grad_imag_1[n*T*K + t*K + k]*data_imag[t*K + k];
    }
    else {
        localStorage[k] = grad_real_2[n*T*K + t*K + k]*data_real[t*K + k] 
                          + grad_imag_2[n*T*K + t*K + k]*data_imag[t*K + k];
    }
    // summing the elements in the localStorage
    size_t preStride = K;
    size_t stride = preStride/2;

    while (stride > 0){
        barrier(CLK_LOCAL_MEM_FENCE);
        stride += preStride % 2;

        if (k + stride < preStride){
            localStorage[k] += localStorage[k + stride];
        }
        preStride = stride;
        stride /= 2;
    }
    if (k == 0){
        out[ni*T + t] = localStorage[0]/T/K;
    }
}

__kernel void assign_division(__global const double *numerator,
                              __global const double *denominator,
                              __global double *result)
{
    size_t i = get_global_id(0);
    result[i] = numerator[i]/denominator[i];
}

__kernel void add_in_place(__global const double *array1,
                           __global const double *array2,
                           __global double *out)
{
    size_t i = get_global_id(0);
    out[i] = array1[i]+array2[i];
}

__kernel void put_grad_F_together(__global const double *W_val,
                                  __global const double *L_val,
                                  __global const double *dW_val,
                                  __global const double *dL_val,
                                  __global double *out)
{
    size_t t = get_global_id(0);
    size_t i = get_global_id(1);  // 
    size_t n = get_global_id(2);
    size_t T = get_global_size(0);
    // size_t I = get_global_id(1); is equal to 2.
    double L = L_val[n];
    out[n*2*T + i*T + t] = (L*dW_val[n*2*T + i*T + t] 
                            - W_val[n]*dL_val[n*2*T + i*T + t])/L/L;
}

__kernel void H1_seminorm_squared(__global const double *curves,
                          __global double *out,
                          __local double *localStorage)
{
    // curves is of size Nx2xT and we have 2xT work-groups.
    size_t local_id = get_local_id(0);
    size_t local_id_j = get_local_id(1);
    size_t T = get_local_size(0);
    size_t global_id_j = get_global_id(1);
    size_t n = get_global_id(2);
    // Copy the array in local memory
    double private_value = curves[n*T*2 + T*local_id_j + local_id];
    localStorage[T*local_id_j + local_id] = private_value;
    barrier(CLK_LOCAL_MEM_FENCE);
    // Substract the forward element
    if (local_id < T-1){
       private_value = private_value - localStorage[T*local_id_j +local_id + 1];
       private_value = private_value*private_value;
       barrier(CLK_LOCAL_MEM_FENCE);
       localStorage[T*local_id_j + local_id] = private_value;  // store the squares difference
       barrier(CLK_LOCAL_MEM_FENCE);
       if (local_id_j == 0){
           localStorage[local_id] = localStorage[local_id]
                                    + localStorage[T + local_id];
           // Up to this point, all squared differences are stored in the local
           // Storage.
           // We proceed now with the reduction.
           // T-1 are the elements to be summed
           // For the rest, refer to sumGPUb_2D comments.
           size_t preStride = T-1;
           size_t stride = preStride/2;
    //        if (local_id == 0 && global_id_j == 0){
    //            for (int i = 0; i < T-1 ; ++i)
    //                    printf("local storage i: %f\n", localStorage[i]);
    //        }

           while (stride > 0){
               barrier(CLK_LOCAL_MEM_FENCE);
               stride += preStride % 2;

               if (local_id + stride < preStride)
                   localStorage[local_id] += localStorage[local_id + stride];

               preStride = stride;
               stride /= 2;
           }
           if (local_id == 0)
               out[n] = (T-1)*localStorage[0];
       }
    }
}

__kernel void L2_norm_squared(__global const double *curves,
                      __global double *out,
                      __local double *localStorage)
{
    // curves is of size Nx2xT and we have 2xT work-groups.
    size_t local_id = get_local_id(0);
    size_t local_id_j = get_local_id(1);
    size_t T = get_local_size(0);
    size_t global_id_j = get_global_id(1);
    size_t n = get_global_id(2);
    // Copy the array in local memory
    double private_value = curves[n*T*2 + T*local_id_j + local_id];
    double private_value2;
    localStorage[T*local_id_j + local_id] = private_value;
    barrier(CLK_LOCAL_MEM_FENCE);
    // Substract the forward element
    if (local_id < T-1){
       private_value2 = localStorage[T*local_id_j + local_id + 1];
       private_value = private_value*private_value 
                       + private_value2*private_value2
                       + private_value*private_value2;
       barrier(CLK_LOCAL_MEM_FENCE);
       localStorage[T*local_id_j + local_id] = private_value;  // store the squares difference
       barrier(CLK_LOCAL_MEM_FENCE);
       if (local_id_j == 0){
           localStorage[local_id] = localStorage[local_id]
                                    + localStorage[T + local_id];
           // Up to this point, all squared differences are stored in the local
           // Storage.
           // We proceed now with the reduction.
           // T-1 are the elements to be summed
           // For the rest, refer to sumGPUb_2D comments.
           size_t preStride = T-1;
           size_t stride = preStride/2;
    //        if (local_id == 0 && global_id_j == 0){
    //            for (int i = 0; i < T-1 ; ++i)
    //                    printf("local storage i: %f\n", localStorage[i]);
    //        }

           while (stride > 0){
               barrier(CLK_LOCAL_MEM_FENCE);
               stride += preStride % 2;

               if (local_id + stride < preStride)
                   localStorage[local_id] += localStorage[local_id + stride];

               preStride = stride;
               stride /= 2;
           }
           if (local_id == 0)
               out[n] = localStorage[0]/(T-1)/3;
       }
    }
}

__kernel void vector_norm_squared(__global const double *curves,
                                  __global double *out,
                                  __local double *localStorage)
{
    // curves is of size Nx2xT and we have 2xT work-groups.
    size_t local_id = get_local_id(0);
    size_t local_id_j = get_local_id(1);
    size_t T = get_local_size(0);
    size_t global_id_j = get_global_id(1);
    size_t n = get_global_id(2);
    // Copy the array in local memory
    double private_value = curves[n*T*2 + T*local_id_j + local_id];
    localStorage[T*local_id_j + local_id] = private_value*private_value;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id_j == 0){
        localStorage[local_id] = localStorage[local_id]
                                    + localStorage[T + local_id];
           // Up to this point, all squared differences are stored in the local
           // Storage.
           // We proceed now with the reduction.
           // T-1 are the elements to be summed
           // For the rest, refer to sumGPUb_2D comments.
        size_t preStride = T;
        size_t stride = preStride/2;

        while (stride > 0){
            barrier(CLK_LOCAL_MEM_FENCE);
            stride += preStride % 2;

            if (local_id + stride < preStride)
                localStorage[local_id] += localStorage[local_id + stride];

            preStride = stride;
            stride /= 2;
        }
        if (local_id == 0)
            out[n] = localStorage[0];
    }
}

__kernel void gradient_update(__global double *curve,
                              __global const double *stepsizes,
                              __global const double *gradient)
{
    // curve, gradient, out_curve with size Nx2xT, stepsizes has size N, 
    size_t t = get_global_id(0);
    size_t i = get_global_id(1);
    size_t n = get_global_id(2);
    size_t T = get_global_size(0);
    // 2 = get_global_size(1);
    curve[2*T*n + T*i + t] = curve[2*T*n + T*i + t] - 
                                 stepsizes[n]*gradient[2*T*n + T*i + t];
}

__kernel void backtracking(__global double *F_curve,
                           __global const double *F_new_curve,
                           __global const double *F_curve_grad,
                           __global double *stepsizes)
{
    size_t n = get_global_id(0);
    double control = 0.5;
    double increase = 1.05;
    double decrease = 0.8;
    double F_new = F_new_curve[n];
    if (F_new < F_curve[n] - control*stepsizes[n]*F_curve_grad[n]){
        stepsizes[n] = stepsizes[n]*increase;
    }
    else {
        stepsizes[n] = stepsizes[n]*decrease;
    }
//    if (F_curve[n] - F_new  <= control*F_curve_grad[n]*stepsizes[n] && stepsizes[n] > 0.05){
//       stepsizes[n] = stepsizes[n]*decrease; 
//    }
//    else {
//        stepsizes[n] = stepsizes[n]*increase;
//    }
    F_curve[n] = F_new;
}
