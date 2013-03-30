#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
//#include <cutil_inline.h>

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define EXTERN 1

#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#define EXPANDED_SIZE(__x) (__x+(__x>>LOG_NUM_BANKS)+(__x>>(2*LOG_NUM_BANKS)))

////////////////////////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void scan_L1_kernel(unsigned int n, unsigned int* data, unsigned int* inter)
{
    __shared__ unsigned int s_data[1104]; // 1088 = 1024 + (1024/16);

    unsigned int thid = threadIdx.x;
    unsigned int g_ai = blockIdx.x*2*blockDim.x + threadIdx.x;
    unsigned int g_bi = g_ai + blockDim.x;

    unsigned int s_ai = thid;
    unsigned int s_bi = thid + blockDim.x;

    s_ai += CONFLICT_FREE_OFFSET(s_ai);
    s_bi += CONFLICT_FREE_OFFSET(s_bi);

    s_data[s_ai] = (g_ai < n) ? data[g_ai] : 0;
    s_data[s_bi] = (g_bi < n) ? data[g_bi] : 0;

    unsigned int stride = 1;
    for (unsigned int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)
        {
            unsigned int i  = 2*stride*thid;
            unsigned int ai = i + stride - 1;
            unsigned int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            s_data[bi] += s_data[ai];
        }

        stride *= 2;
    }

    if (thid == 0){
        unsigned int last = blockDim.x*2 -1;
        last += CONFLICT_FREE_OFFSET(last);
        inter[blockIdx.x] = s_data[last];
        s_data[last] = 0;
    }

    for (unsigned int d = 1; d <= blockDim.x; d *= 2)
    {
        stride >>= 1;

        __syncthreads();

        if (thid < d)
        {
            unsigned int i  = 2*stride*thid;
            unsigned int ai = i + stride - 1;
            unsigned int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            unsigned int t  = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
    __syncthreads();

    if (g_ai < n) { data[g_ai] = s_data[s_ai]; }
    if (g_bi < n) { data[g_bi] = s_data[s_bi]; }
}

__global__ void scan_inter1_kernel(unsigned int* data, unsigned int iter)
{
#if EXTERN
    extern __shared__ unsigned int s_data[]; // 552 = 512 + (512/16);
#else
    __shared__ unsigned int s_data[552];
#endif

    unsigned int thid = threadIdx.x;
    unsigned int gthid = (blockIdx.x*blockDim.x + threadIdx.x);
    unsigned int gi = 2*iter*gthid;
    unsigned int g_ai = gi + iter - 1;
    unsigned int g_bi = g_ai + iter;

    unsigned int s_ai = 2*thid;
    unsigned int s_bi = 2*thid + 1;

    s_ai += CONFLICT_FREE_OFFSET(s_ai);
    s_bi += CONFLICT_FREE_OFFSET(s_bi);

    s_data[s_ai] = data[g_ai];
    s_data[s_bi] = data[g_bi];

    unsigned int stride = 1;
    for (unsigned int d = blockDim.x; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)
        {
            unsigned int i  = 2*stride*thid;
            unsigned int ai = i + stride - 1;
            unsigned int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            s_data[bi] += s_data[ai];
        }

        stride *= 2;
    }

    __syncthreads();

    data[g_ai] = s_data[s_ai];
    data[g_bi] = s_data[s_bi];
}

__global__ void scan_inter2_kernel(unsigned int* data, unsigned int iter)
{
#if EXTERN
    extern __shared__ unsigned int s_data[]; // 552 = 512 + (512/16) + (512/64);
#else
    __shared__ unsigned int s_data[552];
#endif

    unsigned int thid = threadIdx.x;
    unsigned int gthid = (blockIdx.x*blockDim.x + threadIdx.x);
    unsigned int gi = 2*iter*gthid;
    unsigned int g_ai = gi + iter - 1;
    unsigned int g_bi = g_ai + iter;

    unsigned int s_ai = 2*thid;
    unsigned int s_bi = 2*thid + 1;

    s_ai += CONFLICT_FREE_OFFSET(s_ai);
    s_bi += CONFLICT_FREE_OFFSET(s_bi);

    s_data[s_ai] = data[g_ai];
    s_data[s_bi] = data[g_bi];

#if EXTERN
    unsigned int stride = blockDim.x*2;
#else
    unsigned int stride = 512;
#endif
    for (unsigned int d = 1; d <= blockDim.x; d *= 2)
    {
        stride >>= 1;

        __syncthreads();

        if (thid < d)
        {
            unsigned int i  = 2*stride*thid;
            unsigned int ai = i + stride - 1;
            unsigned int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            unsigned int t  = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
    __syncthreads();

    data[g_ai] = s_data[s_ai];
    data[g_bi] = s_data[s_bi];
}

__global__ void uniformAdd(unsigned int n, unsigned int *data, unsigned int *inter)
{

    __shared__ unsigned int uni;
    if (threadIdx.x == 0) { uni = inter[blockIdx.x]; }
    __syncthreads();

    unsigned int g_ai = blockIdx.x*2*blockDim.x + threadIdx.x;
    unsigned int g_bi = g_ai + blockDim.x;

    if (g_ai < n) { data[g_ai] += uni; }
    if (g_bi < n) { data[g_bi] += uni; }
}

void scanLargeArray( unsigned int gridNumElements, unsigned int* data_d) {
    unsigned int gridNumElems = gridNumElements;    

    // allocate device memory input and output arrays
    unsigned int* inter_d = NULL;

//    /*cutilSafeCall( */cudaMalloc( (void**) &inter_d, 512*512*sizeof(unsigned int));//);
//    /*cutilSafeCall( */cudaMemset (inter_d, 0, 512*512*sizeof(unsigned int));//);
    
    // Run the prescan
    unsigned int size = (gridNumElems+1023)/1024;

#if EXTERN
    unsigned int dim_block;
    unsigned int current_max = size*1024;
    for (int block_size = 128; block_size <= 1024; block_size *= 2){
      unsigned int array_size = block_size;
      while(array_size < size){
        array_size *= block_size;
      }
      if (array_size <= current_max){
        current_max = array_size;
        dim_block = block_size;
      }
    }
#else
    unsigned int dim_block = 512;
    unsigned int current_max = 512*512;
#endif

    //printf("dim_block = %d, current_max = %d, expanded_size = %d\n", dim_block, current_max, EXPANDED_SIZE(dim_block));

    cudaMalloc( (void**) &inter_d, current_max*sizeof(unsigned int));
    cudaMemset (inter_d, 0, current_max*sizeof(unsigned int));

    for (unsigned int i=0; i < (size+65534)/65535; i++){
        unsigned int gridSize = ((size-(i*65535)) > 65535) ? 65535 : (size-i*65535);
        unsigned int numElems = ((gridNumElems-(i*65535*1024)) > (65535*1024)) ? (65535*1024) : (gridNumElems-(i*65535*1024));

        dim3 block (512);
        dim3 grid (gridSize);
        scan_L1_kernel<<<grid, block>>>(numElems, data_d+(i*65535*1024), inter_d+(i*65535));
        CUT_CHECK_ERROR("Unable to launch scan_L1_kernel\n");
    }

    //unsigned int inter_size = current_max;
    unsigned int stride = 1;
    for (unsigned int d = current_max; d > 1; d /= dim_block)
    {
        dim3 block (dim_block/2);
        dim3 grid (d/dim_block);
#if EXTERN
        scan_inter1_kernel<<<grid, block, EXPANDED_SIZE(dim_block)*sizeof(unsigned int)>>>(inter_d, stride);
#else
	scan_inter1_kernel<<<grid, block>>>(inter_d, stride);
#endif
        CUT_CHECK_ERROR("Unable to launch scan_inter1_kernel\n");
        stride *= dim_block;
    }

    cudaMemset(&(inter_d[current_max-1]), 0, sizeof(unsigned int));

    for (unsigned int d = dim_block; d <= current_max; d *= dim_block)
    {
        stride /= dim_block;
        dim3 block (dim_block/2);
        dim3 grid (d/dim_block);
#if EXTERN
        scan_inter2_kernel<<<grid, block, EXPANDED_SIZE(dim_block)*sizeof(unsigned int)>>>(inter_d, stride);
#else
	scan_inter2_kernel<<<grid, block>>>(inter_d, stride);
#endif
        CUT_CHECK_ERROR("Unable to launch scan_inter2_kernel\n");
    }

    for (unsigned int i=0; i < (size+65534)/65535; i++){
        unsigned int gridSize = ((size-(i*65535)) > 65535) ? 65535 : (size-i*65535);
        unsigned int numElems = ((gridNumElems-(i*65535*1024)) > (65535*1024)) ? (65535*1024) : (gridNumElems-(i*65535*1024));

        dim3 block (512);
        dim3 grid (gridSize);
        uniformAdd<<<grid, block>>>(numElems, data_d+(i*65535*1024), inter_d+(i*65535));
        CUT_CHECK_ERROR("Unable to launch uniformAdd kernel\n");
    }

    cudaThreadSynchronize();

    // cleanup memory
    cudaFree(inter_d);
}
