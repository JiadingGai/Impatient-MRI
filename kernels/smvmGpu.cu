/*
(C) Copyright 2010 The Board of Trustees of the University of Illinois.
All rights reserved.

Developed by:

                     IMPACT & MRFIL Research Groups
                University of Illinois, Urbana Champaign

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal with the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimers.

Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimers in the documentation
and/or other materials provided with the distribution.

Neither the names of the IMPACT Research Group, MRFIL Research Group, the
University of Illinois, nor the names of its contributors may be used to
endorse or promote products derived from this Software without specific
prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
THE SOFTWARE.
*/

/*****************************************************************************

    File Name   [smvmGpu.cu]

    Synopsis    [Sparse matrix-vector multiplications.]

    Description [This part is mainly from NVIDIA Corporation. Please read
        their license agreement before you use it.]

    Revision    [0.1; Initial build; Xiao-Long Wu, ECE UIUC]
    Revision    [0.1.1; Code cleaning; Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Further optimization, Code cleaning, Adding more comments;
                 Xiao-Long Wu, ECE UIUC]
    Date        [10/27/2010]

 *****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// CUDA libraries
#include <cutil.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>

// Project header files
#include <structures.h>
#include <smvmGpu.cuh>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Function prototypes                                                      */
/*---------------------------------------------------------------------------*/

    void
bind_x(const float *x);

    void
bind_x(const double *x);

    void
unbind_x(const float *x);

    void
unbind_x(const double *x);

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Interface of the GPU kernel of sparse matrix-vector         */
/*      multiplication.]                                                     */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

// Kernel debugging macros
#if DEBUG_KERNEL_MSG
    #define DEBUG_SMVMGPU  true
#else
    #define DEBUG_SMVMGPU  false
#endif

//#define USE_CACHE  true
#define USE_CACHE  false

    void
smvmGpu(
    FLOAT_T *Cf_r_d, FLOAT_T *Cfi_d,             // Output vector
    const FLOAT_T *xr_d, const FLOAT_T *xi_d,   // Input vector
    const int *Ap_d, const int *Aj_d,           // Matrix in CSR format
    const FLOAT_T *Ax_d, const int num_rows)
{
    #if DEBUG_SMVMGPU
    msg(2, "smvmGpu() begin\n");
    #endif
    startMriTimer(getMriTimer()->timer_smvmGpu);

    // Maximum number of co-resident threads for GTX 280
    // FIXME: Should choose the right one for different machines.
    #define MAX_THREADS (30 * 1024)

    const unsigned int block_size = 256;
    const unsigned int num_blocks = MAX_THREADS/block_size;

    #if 0 //DEBUG_SMVMGPU
    msg(3, "Setup smvmGpu execution parameters.\n");
    msg(3, "  Block size : %d\n", block_size);
    msg(3, "  Grid size  : %d\n", num_blocks);
    msg(3, "  num_rows   : %d\n", num_rows);
    #endif

    #if USE_CACHE
    bind_x(xr_d);
    #endif


    // Jiading GAI
    #if 1
    smvmGpuScalarKernel<<<num_blocks, block_size>>>
        (Cf_r_d, xr_d, Ap_d, Aj_d, Ax_d, num_rows);
    #else
    smvmGpu_kernel<<<num_blocks, block_size>>>
        (Cf_r_d, xr_d, Ap_d, Aj_d, Ax_d, num_rows);
    #endif
                      

    #if USE_CACHE
    unbind_x(xr_d);
    #endif

    #if USE_CACHE
    bind_x(xi_d);
    #endif

    // Jiading GAI
    #if 1
    smvmGpuScalarKernel<<<num_blocks, block_size>>>
        (Cfi_d, xi_d, Ap_d, Aj_d, Ax_d, num_rows);
    #else
    smvmGpu_kernel<<<num_blocks, block_size>>>
        (Cfi_d, xi_d, Ap_d, Aj_d, Ax_d, num_rows);
    #endif


    #if USE_CACHE
    unbind_x(xi_d);
    #endif

    cudaThreadSynchronize();

    // check if kernel execution generated an error
    #ifndef emu
    CUT_CHECK_ERROR("Kernel execution failed");
    #endif

    stopMriTimer(getMriTimer()->timer_smvmGpu);
    #if DEBUG_SMVMGPU
    msg(2, "smvmGpu() end\n");
    #endif
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [The GPU kernel of sparse matrix-vector multiplication.]     */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

#if USE_CACHE
  /*
   * These textures are (optionally) used to cache the 'x' vector in y += A*x
   */
  texture<float, 1> tex_x_float;
  texture<int2, 1>  tex_x_double;
  
  // Use int2 to pull doubles through texture cache
      void
  bind_x(const float *x)
  {   cutilSafeCall(cudaBindTexture(NULL, tex_x_float, x));   }
  
      void
  bind_x(const double *x)
  {   cutilSafeCall(cudaBindTexture(NULL, tex_x_double, x));   }
  
      void
  unbind_x(const float *x)
  {   cutilSafeCall(cudaUnbindTexture(tex_x_float)); }
  
      void
  unbind_x(const double *x)
  {   cutilSafeCall(cudaUnbindTexture(tex_x_double)); }
  
  // Note: x is unused, but distinguishes the two functions
      template <bool UseCache>
      __inline__ __device__ float
  fetch_x(const int & i, const float *x)
  {
      if (UseCache) return tex1Dfetch(tex_x_float, i);
      else return x[i];
  }
  
  #if !defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)
      template <bool UseCache>
      __inline__ __device__ double
  fetch_x(const int & i, const double *x)
  {
      if (UseCache) {
          int2 v = tex1Dfetch(tex_x_double, i);
          return __hiloint2double(v.y, v.x);
      } else {
          return x[i];
      }
  }
  #endif // !defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)
#else
  // FIXME(JDG)!!: update texture binding to the latest cuda version
  
  // Note: x is unused, but distinguishes the two functions
      template <bool UseCache>
      __inline__ __device__ float
  fetch_x(const int & i, const float *x)
  {
    return x[i];
  }
  
  #if !defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)
      template <bool UseCache>
      __inline__ __device__ double
  fetch_x(const int & i, const double *x)
  {
    return x[i];
  }
  #endif // !defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)
#endif

#ifdef emu
    #define EMUSYNC __syncthreads()
#else
    #define EMUSYNC
#endif

    __global__ void
smvmGpu_kernel(
    FLOAT_T *y,
    const FLOAT_T *x, const int *Ap, const int *Aj, const FLOAT_T *Ax,
    const int num_rows)
{
    const unsigned int block_size = 256;
    const unsigned int WARP_SIZE = 32;

    #if USE_CACHE
    const bool UseCache = true;
    #else
    const bool UseCache = false;
    #endif

    __shared__ FLOAT_T sdata[block_size];
    __shared__ int ptrs[block_size/WARP_SIZE][2];

    // global thread index
    const int thread_id   = block_size * blockIdx.x + threadIdx.x;
    // thread index within the warp
    const int thread_lane = threadIdx.x & (WARP_SIZE-1);
    // global warp index
    const int warp_id     = thread_id   / WARP_SIZE;
    // warp index within the CTA
    const int warp_lane   = threadIdx.x / WARP_SIZE;
    // total number of active warps
    const int num_warps   = (block_size / WARP_SIZE) * gridDim.x;

    // Categorize threads in the unit of warps to handle the nonzeros of
    // specified rows. For example, tx0 will handle the (nonzeros % 32)th
    // nonzeros of the (row id % 32)th row.
    for(int row = warp_id; row < num_rows; row += num_warps) {
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the more straightforward option
        if(thread_lane < 2)
            ptrs[warp_lane][thread_lane] = Ap[row + thread_lane];
        //same as: row_start = Ap[row];
        const int row_start = ptrs[warp_lane][0];
        //same as: row_end   = Ap[row+1];
        const int row_end   = ptrs[warp_lane][1];

        // Compute local sums of the nonzeros in the specified row.
        sdata[threadIdx.x] = 0;
        for(int jj = row_start + thread_lane; jj < row_end; jj += WARP_SIZE)
            sdata[threadIdx.x] += Ax[jj] * fetch_x<UseCache>(Aj[jj], x);

        // reduce local sums to row sum (ASSUME: warpsize 32)
        if (thread_lane < 16) {
            sdata[threadIdx.x] += sdata[threadIdx.x + 16]; EMUSYNC;
        }
        if (thread_lane <  8) {
            sdata[threadIdx.x] += sdata[threadIdx.x +  8]; EMUSYNC;
        }
        if (thread_lane <  4) {
            sdata[threadIdx.x] += sdata[threadIdx.x +  4]; EMUSYNC;
        }
        if (thread_lane <  2) {
            sdata[threadIdx.x] += sdata[threadIdx.x +  2]; EMUSYNC;
        }
        if (thread_lane <  1) {
            sdata[threadIdx.x] += sdata[threadIdx.x +  1]; EMUSYNC;
        }

        // first thread writes warp result
        if (thread_lane == 0) y[row] += sdata[threadIdx.x];
    }
}

    __global__ void
smvmGpuScalarKernel(
    FLOAT_T *y,
    const FLOAT_T *x, const int *Ap, const int *Aj, const FLOAT_T *Ax,
    const int num_rows)
{
    #if USE_CACHE
    const bool UseCache = true;
    #else
    const bool UseCache = false;
    #endif

    const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    const int grid_size = gridDim.x * blockDim.x;

    for(int row = thread_id;row < num_rows;row += grid_size)
    {
       const int row_start = Ap[row];
       const int row_end   = Ap[row+1];

       FLOAT_T sum = 0.0f;
      
       for(int jj = row_start; jj < row_end; jj++)
          sum += Ax[jj] * fetch_x<UseCache>(Aj[jj],x);

       y[row] = sum; 
    }
}
/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

