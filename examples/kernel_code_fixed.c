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

__kernel void sumGPUb_2D(const int realLength,
                       __global const double *input,
                       __global double *partialSums,
                       __local double *localSums)
{
    size_t local_id = get_local_id(0);
    size_t group_size = get_local_size(0);
    size_t global_id_i = get_global_id(0);
    size_t global_id_j = get_global_id(1);
    // Copy from global to local memory. Keep in mind, that for
    // global_id_i >= real_length, we are copying wrong/trash values.
    localSums[local_id] = input[global_id_j*realLength + global_id_i];

    // Loop for computing LocalSums: divide WorkGroup into 2 parts
    size_t preStride = group_size;
    size_t stride = group_size/2;

    while(stride > 0)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (global_id_i + stride < realLength)
        {
            if (preStride % 2 == 1)
            {
                stride += 1;
                if (local_id < stride-1)
                    localSums[local_id] += localSums[local_id + stride];
                if (local_id == stride)
                    localSums[local_id] = localSums[local_id + stride];
            }
            else
            {
                if (local_id < stride)
                    localSums[local_id] += localSums[local_id + stride];
            }
            preStride = stride;
            stride /= 2;
        }
    }
    if (local_id == 0 && global_id_i < realLength)
        partialSums[global_id_j*realLength + get_group_id(0)] = localSums[0];
}
