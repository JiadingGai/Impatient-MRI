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
prior writimeen permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
THE SOFTWARE.
*/

/*****************************************************************************

    File Name   [ftGpu.cu]

    Synopsis    [GPU version of FT and IFT implementation.]

    Description []

    Revision    [0.1; Initial build; Xiao-Long Wu, ECE UIUC]
    Revision    [0.1.1; Calculating FLOPS, Code cleaning;
                 Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Further optimization, Code cleaning, Adding comments;
                 Xiao-Long Wu, ECE UIUC, Jiading Gai, Beckman Institute]
    Date        [10/27/2010]

*****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>
#include <stdio.h>
#include <math.h>

// CUDA libraries
#include <cutil.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>

// Project header files
#include <xcpplib_process.h>
#include <xcpplib_typesGpu.cuh>
#include <tools.h>
#include <structures.h>

#include <ftGpu.cuh>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Macro definitions                                                        */
/*---------------------------------------------------------------------------*/

#if ENABLE_DOUBLE_PRECISION
    #define DATATRAJ_NO_Z_DIM           true

    // Used constant memory size for tiled computation.
    // Since constant memory is only 64KB. Using DataTraj(x,y) * 8 * 2, where
    // the number of DataTraj(x,y) is 4096, fits 64KB exactly. But compiler
    // may use some. So 4050 is the final number.
    #if DATATRAJ_NO_Z_DIM
        #define TILED_TRAJ_SIZE         2048 // Better to be power of 2
    #else
        #define TILED_TRAJ_SIZE         1024
    #endif
#else // Single precision
    #define DATATRAJ_NO_Z_DIM           false

    // Used constant memory size for computation per kernel computation.
    #define TILED_TRAJ_SIZE             4096 // Better to be power of 2
#endif

// Both must be the same with the # of data elements in constant memory.
#define K_ELEMS_PER_TILE                TILED_TRAJ_SIZE
#define I_ELEMS_PER_TILE                TILED_TRAJ_SIZE
#define JOBS_PER_THREAD                 1

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [GPU version of the sin function.]                           */
/*                                                                           */
/*  Description [This function is used to avoid additional computations when */
/*      the data values are too small.]                                      */
/*                                                                           */
/*===========================================================================*/

#if COMPUTE_FLOPS
__device__ FLOAT_T sinc_gpu(FLOAT_T x, int *flop)
#else
__device__ FLOAT_T sinc_gpu(FLOAT_T x)
#endif
{
    // This section seems uncessary for GPU but it causes more inaccuracy
    // for some reason.
    #ifdef ENABLE_DOUBLE_PRECISION
    if (fabs(x) < 0.0001) {
    #else
    if (fabs(x) < 0.0001f) {
    #endif
        return 1.0;
    }

    #if COMPUTE_FLOPS
    *flop += 13 + 1 + 1 + 1;
    #endif
    return sinf(MRI_PI * x) / (MRI_PI * x);
}

// Constant memory tile. 64KB total.
// This is used to store the itraj (FT) and ktraj (IFT) tiled data.
// So watch out the indexing when data is tiled-copied to this place and
// accessed later.
__device__ __constant__ DataTraj Trajc[TILED_TRAJ_SIZE];

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [GPU kernel interface of the Fourier Transformation (FT).]   */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

// Kernel debugging macros
#if DEBUG_KERNEL_MSG
    #define DEBUG_FTGPU     true
#else
    #define DEBUG_FTGPU     false
#endif

// Optimization strategies. Note: These are exclusive.
#define FT_NO_SPEEDUP       false    // Useful for small data set
#define FT_LOOP_UNROLLING   true    // Useful for big data set
#define FT_DATA_PREFETCH    false   // bug inside?

// Performance tuning
#if FT_NO_SPEEDUP
    #define FT_BLOCK_SIZE   256
#elif FT_LOOP_UNROLLING
    #define FT_BLOCK_SIZE   256
#elif FT_DATA_PREFETCH
    #define FT_BLOCK_SIZE   256 // Better: 256 = 512 = 64 >> 128 
#endif

// Error checking scheme
#if (FT_NO_SPEEDUP && FT_LOOP_UNROLLING)
    #error Conflict on FT optimization flags.
#endif

    __global__ void
ftGpuKernel(
    FLOAT_T *kdata_r, FLOAT_T *kdata_i,
    const FLOAT_T *idata_r, const FLOAT_T *idata_i,
    const DataTraj *ktraj, int processed,
    const FLOAT_T *fm, const FLOAT_T *time_,
    const int num_k, const int num_k_cpu, const int left_num
    #if COMPUTE_FLOPS
    , int *flop_array
    #endif
    );

    void
ftGpu(FLOAT_T *kdata_r, FLOAT_T *kdata_i,
      const FLOAT_T *idata_r, const FLOAT_T *idata_i,
      const DataTraj *ktraj, const DataTraj *itraj,
      const FLOAT_T *fm, const FLOAT_T *time,
      const int num_k, const int num_k_cpu, const int num_i
      )
{
    #if DEBUG_FTGPU
    msg(2, "ftGpu() begin\n");
    #endif
    startMriTimer(getMriTimer()->timer_ftGpu);

    // FIXME: Need to check input first.
    // kdata_r/i (num_k) must be padded.

    int num_grid = num_i / I_ELEMS_PER_TILE;
    if (num_i % I_ELEMS_PER_TILE) num_grid++;
    int num_block = num_k / (FT_BLOCK_SIZE * JOBS_PER_THREAD);
    if (num_k % (FT_BLOCK_SIZE * JOBS_PER_THREAD)) num_block++;

    #if DEBUG_FTGPU
    msg(3, "ftGpu:\n");
    msg(3, "  num_grid #: %d\n", num_grid);
    msg(3, "  Block #  : %d\n", num_block);
    msg(3, "  Thread # : %d\n", FT_BLOCK_SIZE);
    #endif

    dim3 dim_grid(num_block);
    dim3 dim_block(FT_BLOCK_SIZE * JOBS_PER_THREAD);

    #if COMPUTE_FLOPS
    const int flop_array_num = num_block * FT_BLOCK_SIZE;
    int *flop_array = mriNewCpu<int>(flop_array_num);
    int *flop_array_d = mriNewGpu<int>(flop_array_num);
    #endif

    #if DEBUG_FTGPU
    msg(3, "Launch %d kernels (%dx%d) on %d elements/kernel.\n",
        num_grid, dim_grid.x, dim_block.x, I_ELEMS_PER_TILE);
    #endif

    for (int i_grid = 0; i_grid < num_grid; i_grid++) {
        // Put the tile of K values into constant mem
        const int processed = i_grid * I_ELEMS_PER_TILE;
        const DataTraj * itraj_tile = itraj + processed;
        const int left_num = MIN(I_ELEMS_PER_TILE, num_i - processed);

        #if DEBUG_FTGPU
        msg(4, "Kernel %d processed: %d, left_num: %d\n",
            i_grid, processed, left_num);
        #endif

        // Number of ktraj in global memory to copy to constant memory
        cudaError e = cudaMemcpyToSymbol(Trajc, itraj_tile, left_num *
                      sizeof(DataTraj), 0, cudaMemcpyDeviceToDevice);
        if(e!=cudaSuccess)
        {
          printf("Cuda error in file '%s' in line %i : %s.\n",
                  __FILE__,__LINE__, cudaGetErrorString(e));
          exit(EXIT_FAILURE);
        }
        makeSure(e == cudaSuccess,
            "CUDA runtime failure on cudaMemcpyToSymbol.");

        // Must clear the memory for kdata_r/k smaller than I_ELEMS_PER_TILE.
        if (num_i < I_ELEMS_PER_TILE) {
            clearCuda<FLOAT_T>(kdata_r, num_k);
            clearCuda<FLOAT_T>(kdata_i, num_k);
        }
        ftGpuKernel<<< dim_grid, dim_block >>>
                      (kdata_r, kdata_i, idata_r, idata_i,
                       ktraj, processed, fm, time,
                       num_k, num_k_cpu, left_num
                       #if COMPUTE_FLOPS
                       , flop_array_d
                       #endif
                       );
    }
    cudaThreadSynchronize();

    #if COMPUTE_FLOPS
    mriCopyDeviceToHost<int>(flop_array, flop_array_d, flop_array_num);
    for (int i = 0; i < flop_array_num; i++) {
        getMriFlop()->flop_ftGpu += (double) flop_array[i];
    }
    //printf("*** # of threads: %d (%d * %d)\n", flop_array_num, num_block,
    //    FT_BLOCK_SIZE);
    //printf("*** # of accumulated flop: %.f\n", getMriFlop()->flop_ftGpu);

    mriDeleteGpu(flop_array_d);
    mriDeleteCpu(flop_array);
    #endif // COMPUTE_FLOPS

    stopMriTimer(getMriTimer()->timer_ftGpu);
    #if DEBUG_FTGPU
    msg(2, "ftGpu() end\n");
    #endif
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [GPU kernel of the Fourier Transformation (FT).]             */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    __global__ void
ftGpuKernel(
    FLOAT_T *kdata_r, FLOAT_T *kdata_i,
    const FLOAT_T *idata_r, const FLOAT_T *idata_i,
    const DataTraj *ktraj, int processed,
    const FLOAT_T *fm, const FLOAT_T *time_,
    const int num_k, const int num_k_cpu, const int left_num
    #if COMPUTE_FLOPS
    , int *flop_array
    #endif
    )
{
    // Determine the element of the X arrays computed by this thread
    const int k_idx = blockIdx.x * FT_BLOCK_SIZE + threadIdx.x;

    // This is useful when the data is not padded to power of 2.
    if (k_idx >= num_k) { return; }

    FLOAT_T sumr = MRI_ZERO, sumi = MRI_ZERO;
    const FLOAT_T ktraj_z = MRI_ZERO;
    const FLOAT_T itraj_z = MRI_ZERO;
    const FLOAT_T tpi = MRI_PI * 2.0;       // done at compile time
    FLOAT_T kziztpi = ktraj_z * itraj_z * tpi;
    FLOAT_T time = time_[k_idx];
    FLOAT_T kx = ktraj[k_idx].x;
    FLOAT_T ky = ktraj[k_idx].y;
    // Division by constants is replaced with shift operations by compilers.
    FLOAT_T kx_tpi = kx * tpi;        // flop: 1
    FLOAT_T ky_tpi = ky * tpi;        // flop: 1

    #if COMPUTE_FLOPS
    flop_array[k_idx] += 1 + 1;
    #endif

    // FIXME: Future optimizations
    // 1) Move some data into shared memory
    // 2) Use texture memory

    // Optimizations
    // When enable flag COMPUTE_FLOPS, it uses the slowest version. Otherwise,
    // it picks the best one.

    // Loop over all elements of K to compute a partial value for X.
    #if FT_NO_SPEEDUP // =====================================================

    for (int i_idx = 0; i_idx < left_num; i_idx++, processed++) {
        FLOAT_T expr, cosexpr, sinexpr, idatac_r, idatac_i;
        expr = (kx_tpi * Trajc[i_idx].x + ky_tpi * Trajc[i_idx].y +
                kziztpi) + (fm[processed] * time);
        #if 1
            #if ENABLE_DOUBLE_PRECISION
                cosexpr = cos(expr); sinexpr = sin(expr);
            #else
                cosexpr = cosf(expr); sinexpr = sinf(expr);
            #endif
        #else // No speedup
            cosexpr = MRI_ZERO; sinexpr = MRI_ZERO;
            sincosf(expr, &sinexpr, &cosexpr);
        #endif

        idatac_r = idata_r[processed]; idatac_i = idata_i[processed];
        sumr += ( cosexpr*idatac_r) + (sinexpr*idatac_i);
        sumi += (-sinexpr*idatac_r) + (cosexpr*idatac_i);
    }

    #elif COMPUTE_FLOPS
    // Note: This is defined in the Makefile.

    for (int i_idx = 0; i_idx < left_num; i_idx++, processed++) {
        FLOAT_T expr, cosexpr, sinexpr, idatac_r, idatac_i;
        expr = (kx_tpi * Trajc[i_idx].x + ky_tpi * Trajc[i_idx].y +
                kziztpi) + (fm[processed] * time);
        #if 1
        cosexpr = cosf(expr); sinexpr = sinf(expr);
        #else // No speedup
        cosexpr = MRI_ZERO; sinexpr = MRI_ZERO;
        sincosf(expr, &sinexpr, &cosexpr);
        #endif

        idatac_r = idata_r[processed]; idatac_i = idata_i[processed];
        sumr += ( cosexpr*idatac_r) + (sinexpr*idatac_i);
        sumi += (-sinexpr*idatac_r) + (cosexpr*idatac_i);

        #if COMPUTE_FLOPS
        // software sin: 13 flop; cos: 12 flop; +,-,*,/: 1 flop
        // hardware sin/cos: 1 flop; +,-,*,/: 1 flop
        // Total operations are 17. Total memory accesses are 3 global memory
        // and 2 constant memory accesses.
        flop_array[k_idx] += 6 + (1 + 1) + (4 + 5);
        #endif
    }
    #elif FT_LOOP_UNROLLING // ===============================================
    // Note: The idata must be padded for this optimization.

        #if 0 // unrolling version 1
        #define FT_UNROLL_FACTOR  8 // 2, 4, 8
        FLOAT_T expr[FT_UNROLL_FACTOR], cosexpr[FT_UNROLL_FACTOR],
                sinexpr[FT_UNROLL_FACTOR],
                sum_r[FT_UNROLL_FACTOR], sum_i[FT_UNROLL_FACTOR];
        FLOAT_T idatac_r, idatac_i;
        for (int i_idx=0; i_idx < left_num;
            i_idx += FT_UNROLL_FACTOR, processed += FT_UNROLL_FACTOR) {
            expr[0] = (kx_tpi * Trajc[i_idx].x + ky_tpi * Trajc[i_idx].y +
                       kziztpi) + (fm[processed+0] * time);
            cosexpr[0] = cosf(expr[0]); sinexpr[0] = sinf(expr[0]);
            idatac_r = idata_r[processed+0]; idatac_i = idata_i[processed+0];
            sum_r[0] = ( cosexpr[0]*idatac_r) + (sinexpr[0]*idatac_i);
            sum_i[0] = (-sinexpr[0]*idatac_r) + (cosexpr[0]*idatac_i);
    
            expr[1] = (kx_tpi * Trajc[i_idx+1].x + ky_tpi * Trajc[i_idx+1].y +
                       kziztpi) + (fm[processed+1] * time);
            cosexpr[1] = cosf(expr[1]); sinexpr[1] = sinf(expr[1]);
            idatac_r = idata_r[processed+1]; idatac_i = idata_i[processed+1];
            sum_r[1] = ( cosexpr[1]*idatac_r) + (sinexpr[1]*idatac_i);
            sum_i[1] = (-sinexpr[1]*idatac_r) + (cosexpr[1]*idatac_i);
    
            #if FT_UNROLL_FACTOR == 2
            sumr += sum_r[0] + sum_r[1];
            sumi += sum_i[0] + sum_i[1];
            #endif
    
            #if FT_UNROLL_FACTOR > 2
            expr[2] = (kx_tpi * Trajc[i_idx+2].x + ky_tpi * Trajc[i_idx+2].y +
                       kziztpi) + (fm[processed+2] * time);
            cosexpr[2] = cosf(expr[2]); sinexpr[2] = sinf(expr[2]);
            idatac_r = idata_r[processed+2]; idatac_i = idata_i[processed+2];
            sum_r[2] = ( cosexpr[2]*idatac_r) + (sinexpr[2]*idatac_i);
            sum_i[2] = (-sinexpr[2]*idatac_r) + (cosexpr[2]*idatac_i);
    
            expr[3] = (kx_tpi * Trajc[i_idx+3].x + ky_tpi * Trajc[i_idx+3].y +
                       kziztpi) + (fm[processed+3] * time);
            cosexpr[3] = cosf(expr[3]); sinexpr[3] = sinf(expr[3]);
            idatac_r = idata_r[processed+3]; idatac_i = idata_i[processed+3];
            sum_r[3] = ( cosexpr[3]*idatac_r) + (sinexpr[3]*idatac_i);
            sum_i[3] = (-sinexpr[3]*idatac_r) + (cosexpr[3]*idatac_i);
            #endif
    
            #if FT_UNROLL_FACTOR == 4
            sumr += sum_r[0] + sum_r[1] + sum_r[2] + sum_r[3];
            sumi += sum_i[0] + sum_i[1] + sum_i[2] + sum_i[3];
            #endif
    
            #if FT_UNROLL_FACTOR > 4
            expr[4] = (kx_tpi * Trajc[i_idx+4].x + ky_tpi * Trajc[i_idx+4].y +
                       kziztpi) + (fm[processed+4] * time);
            cosexpr[4] = cosf(expr[4]); sinexpr[4] = sinf(expr[4]);
            idatac_r = idata_r[processed+4]; idatac_i = idata_i[processed+4];
            sum_r[4] = ( cosexpr[4]*idatac_r) + (sinexpr[4]*idatac_i);
            sum_i[4] = (-sinexpr[4]*idatac_r) + (cosexpr[4]*idatac_i);
    
            expr[5] = (kx_tpi * Trajc[i_idx+5].x + ky_tpi * Trajc[i_idx+5].y +
                       kziztpi) + (fm[processed+5] * time);
            cosexpr[5] = cosf(expr[5]); sinexpr[5] = sinf(expr[5]);
            idatac_r = idata_r[processed+5]; idatac_i = idata_i[processed+5];
            sum_r[5] = ( cosexpr[5]*idatac_r) + (sinexpr[5]*idatac_i);
            sum_i[5] = (-sinexpr[5]*idatac_r) + (cosexpr[5]*idatac_i);
    
            expr[6] = (kx_tpi * Trajc[i_idx+6].x + ky_tpi * Trajc[i_idx+6].y +
                       kziztpi) + (fm[processed+6] * time);
            cosexpr[6] = cosf(expr[6]); sinexpr[6] = sinf(expr[6]);
            idatac_r = idata_r[processed+6]; idatac_i = idata_i[processed+6];
            sum_r[6] = ( cosexpr[6]*idatac_r) + (sinexpr[6]*idatac_i);
            sum_i[6] = (-sinexpr[6]*idatac_r) + (cosexpr[6]*idatac_i);
    
            expr[7] = (kx_tpi * Trajc[i_idx+7].x + ky_tpi * Trajc[i_idx+7].y +
                       kziztpi) + (fm[processed+7] * time);
            cosexpr[7] = cosf(expr[7]); sinexpr[7] = sinf(expr[7]);
            idatac_r = idata_r[processed+7]; idatac_i = idata_i[processed+7];
            sum_r[7] = ( cosexpr[7]*idatac_r) + (sinexpr[7]*idatac_i);
            sum_i[7] = (-sinexpr[7]*idatac_r) + (cosexpr[7]*idatac_i);
            #endif
    
            #if FT_UNROLL_FACTOR == 8
            sumr += sum_r[0] + sum_r[1] + sum_r[2] + sum_r[3] +
                    sum_r[4] + sum_r[5] + sum_r[6] + sum_r[7];
            sumi += sum_i[0] + sum_i[1] + sum_i[2] + sum_i[3] +
                    sum_i[4] + sum_i[5] + sum_i[6] + sum_i[7];
            #endif
        }

        #else // Unrolling version 2 =========================================
        #define FT_UNROLL_FACTOR  8 // 2, 4, 8
        FLOAT_T expr[FT_UNROLL_FACTOR], cosexpr[FT_UNROLL_FACTOR],
                sinexpr[FT_UNROLL_FACTOR];
        FLOAT_T idatac_r, idatac_i;
        for (int i_idx=0; i_idx < left_num;
            i_idx += FT_UNROLL_FACTOR, processed += FT_UNROLL_FACTOR) {
            expr[0] = (kx_tpi * Trajc[i_idx].x + ky_tpi * Trajc[i_idx].y +
                       kziztpi) + (fm[processed+0] * time);
            cosexpr[0] = cosf(expr[0]); sinexpr[0] = sinf(expr[0]);
            idatac_r = idata_r[processed+0]; idatac_i = idata_i[processed+0];
            sumr += ( cosexpr[0]*idatac_r) + (sinexpr[0]*idatac_i);
            sumi += (-sinexpr[0]*idatac_r) + (cosexpr[0]*idatac_i);
    
            expr[1] = (kx_tpi * Trajc[i_idx+1].x + ky_tpi * Trajc[i_idx+1].y +
                       kziztpi) + (fm[processed+1] * time);
            cosexpr[1] = cosf(expr[1]); sinexpr[1] = sinf(expr[1]);
            idatac_r = idata_r[processed+1]; idatac_i = idata_i[processed+1];
            sumr += ( cosexpr[1]*idatac_r) + (sinexpr[1]*idatac_i);
            sumi += (-sinexpr[1]*idatac_r) + (cosexpr[1]*idatac_i);
    
            #if FT_UNROLL_FACTOR > 2
            expr[2] = (kx_tpi * Trajc[i_idx+2].x + ky_tpi * Trajc[i_idx+2].y +
                       kziztpi) + (fm[processed+2] * time);
            cosexpr[2] = cosf(expr[2]); sinexpr[2] = sinf(expr[2]);
            idatac_r = idata_r[processed+2]; idatac_i = idata_i[processed+2];
            sumr += ( cosexpr[2]*idatac_r) + (sinexpr[2]*idatac_i);
            sumi += (-sinexpr[2]*idatac_r) + (cosexpr[2]*idatac_i);
    
            expr[3] = (kx_tpi * Trajc[i_idx+3].x + ky_tpi * Trajc[i_idx+3].y +
                       kziztpi) + (fm[processed+3] * time);
            cosexpr[3] = cosf(expr[3]); sinexpr[3] = sinf(expr[3]);
            idatac_r = idata_r[processed+3]; idatac_i = idata_i[processed+3];
            sumr += ( cosexpr[3]*idatac_r) + (sinexpr[3]*idatac_i);
            sumi += (-sinexpr[3]*idatac_r) + (cosexpr[3]*idatac_i);
            #endif
    
            #if FT_UNROLL_FACTOR > 4
            expr[4] = (kx_tpi * Trajc[i_idx+4].x + ky_tpi * Trajc[i_idx+4].y +
                       kziztpi) + (fm[processed+4] * time);
            cosexpr[4] = cosf(expr[4]); sinexpr[4] = sinf(expr[4]);
            idatac_r = idata_r[processed+4]; idatac_i = idata_i[processed+4];
            sumr += ( cosexpr[4]*idatac_r) + (sinexpr[4]*idatac_i);
            sumi += (-sinexpr[4]*idatac_r) + (cosexpr[4]*idatac_i);
    
            expr[5] = (kx_tpi * Trajc[i_idx+5].x + ky_tpi * Trajc[i_idx+5].y +
                       kziztpi) + (fm[processed+5] * time);
            cosexpr[5] = cosf(expr[5]); sinexpr[5] = sinf(expr[5]);
            idatac_r = idata_r[processed+5]; idatac_i = idata_i[processed+5];
            sumr += ( cosexpr[5]*idatac_r) + (sinexpr[5]*idatac_i);
            sumi += (-sinexpr[5]*idatac_r) + (cosexpr[5]*idatac_i);
    
            expr[6] = (kx_tpi * Trajc[i_idx+6].x + ky_tpi * Trajc[i_idx+6].y +
                       kziztpi) + (fm[processed+6] * time);
            cosexpr[6] = cosf(expr[6]); sinexpr[6] = sinf(expr[6]);
            idatac_r = idata_r[processed+6]; idatac_i = idata_i[processed+6];
            sumr += ( cosexpr[6]*idatac_r) + (sinexpr[6]*idatac_i);
            sumi += (-sinexpr[6]*idatac_r) + (cosexpr[6]*idatac_i);
    
            expr[7] = (kx_tpi * Trajc[i_idx+7].x + ky_tpi * Trajc[i_idx+7].y +
                       kziztpi) + (fm[processed+7] * time);
            cosexpr[7] = cosf(expr[7]); sinexpr[7] = sinf(expr[7]);
            idatac_r = idata_r[processed+7]; idatac_i = idata_i[processed+7];
            sumr += ( cosexpr[7]*idatac_r) + (sinexpr[7]*idatac_i);
            sumi += (-sinexpr[7]*idatac_r) + (cosexpr[7]*idatac_i);
            #endif
        }
        #endif

    #elif FT_DATA_PREFETCH // ================================================

    #define FT_PREFETCH_FACTOR  2
    FLOAT_T fm_p[FT_PREFETCH_FACTOR], idatac_r[FT_PREFETCH_FACTOR],
            idatac_i[FT_PREFETCH_FACTOR];

    fm_p[0]     = fm[processed+0];
    idatac_r[0] = idata_r[processed+0];
    idatac_i[0] = idata_i[processed+0];
    for (int i_idx=0; i_idx < left_num; i_idx += 1, processed += 1) {
        fm_p[1]     = fm[processed+1];
        idatac_r[1] = idata_r[processed+1];
        idatac_i[1] = idata_i[processed+1];

        FLOAT_T expr = (kx_tpi * Trajc[i_idx].x + ky_tpi * Trajc[i_idx].y +
                        kziztpi) + (fm_p[0] * time);
        FLOAT_T cosexpr = cosf(expr);
        FLOAT_T sinexpr = sinf(expr);
        sumr += ( cosexpr * idatac_r[0]) + (sinexpr * idatac_i[0]);
        sumi += (-sinexpr * idatac_r[0]) + (cosexpr * idatac_i[0]);

        fm_p[0]       = fm_p[1];
        idatac_r[0] = idatac_r[1];
        idatac_i[0] = idatac_i[1];
    }
    #endif

    // Only collect the original data points.
    if (k_idx < num_k_cpu) {
        // For first kernel launch, assign them directly.
        if (processed == I_ELEMS_PER_TILE) {
            kdata_r[k_idx] = sumr;
            kdata_i[k_idx] = sumi;
        } else { // For the rest kernel launches, accumulate them.
            kdata_r[k_idx] += sumr;
            kdata_i[k_idx] += sumi;

            #if COMPUTE_FLOPS
            flop_array[k_idx] += 2;
            #endif
        }
    } else { // k_idx >= num_k_cpu: Set zero to padded kdata elements
        kdata_r[k_idx] = MRI_ZERO;
        kdata_i[k_idx] = MRI_ZERO;
    }

    // Debugging the flop count. The collected flop_array[] elements should be
    // the total number of valid threads.
    //flop_array[k_idx] = 1;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [GPU kernel interface of the Inverse Fourier Transformation  */
/*      (IFT).]                                                              */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

// Kernel debugging macros
#if DEBUG_KERNEL_MSG
    #define DEBUG_IFTGPU    true
#else
    #define DEBUG_IFTGPU    false
#endif

// Optimization strategies. Note: These are exclusive.
#define IFT_NO_SPEEDUP      false    // Useful for small data sets
#define IFT_LOOP_UNROLLING  true   // Useful for big data sets
//#define IFT_DATA_PREFETCH   false // not done yet.

// Performance tuning
#if IFT_NO_SPEEDUP
    #define IFT_BLOCK_SIZE  256
#elif IFT_LOOP_UNROLLING
    #define IFT_BLOCK_SIZE  256
#elif IFT_DATA_PREFETCH
    #define IFT_BLOCK_SIZE  256
#endif

// Error checking scheme
#if IFT_NO_SPEEDUP && IFT_LOOP_UNROLLING
    #error Conflict on IFT optimization flags.
#endif

    __global__ void
iftGpuKernel(
    FLOAT_T *idata_r, FLOAT_T *idata_i,
    const FLOAT_T *kdata_r, const FLOAT_T *kdata_i,
    const DataTraj *itraj, int processed,
    const FLOAT_T *fm, const FLOAT_T *time,
    const int num_i, const int left_num
    #if COMPUTE_FLOPS
    , int *flop_array
    #endif
    );

    void
iftGpu(FLOAT_T *idata_r, FLOAT_T *idata_i,
       const FLOAT_T *kdata_r, const FLOAT_T *kdata_i,
       const DataTraj *ktraj, const DataTraj *itraj,
       const FLOAT_T *fm, const FLOAT_T *time,
       const int num_k, const int num_k_cpu, const int num_i
       )
{
    #if DEBUG_IFTGPU
    msg(2, "iftGpu() begin\n");
    #endif
    startMriTimer(getMriTimer()->timer_iftGpu);

    #if IFT_NO_SPEEDUP || COMPUTE_FLOPS
    const int num_k_processed = num_k_cpu;
    #else
    const int num_k_processed = num_k; // Padded kdata is required.
    #endif

    int num_grid = num_k_processed / K_ELEMS_PER_TILE;
    if (num_k_processed % K_ELEMS_PER_TILE) num_grid++;
    int num_block = num_i / (IFT_BLOCK_SIZE * JOBS_PER_THREAD);
    if (num_i % (IFT_BLOCK_SIZE * JOBS_PER_THREAD)) num_block++;

    #if DEBUG_IFTGPU
    msg(3, "iftGpu:\n");
    msg(3, "  num_grid #: %d\n", num_grid);
    msg(3, "  Block  #: %d\n", num_block);
    msg(3, "  Thread #: %d\n", IFT_BLOCK_SIZE);
    msg(3, "  num_k_processed: %d, num_i: %d\n", num_k_processed, num_i);
    #endif

    dim3 dim_grid(num_block);
    dim3 dim_block(IFT_BLOCK_SIZE * JOBS_PER_THREAD);

    #if COMPUTE_FLOPS
    const int flop_array_num = num_block * IFT_BLOCK_SIZE;
    int *flop_array = mriNewCpu<int>(flop_array_num);
    int *flop_array_d = mriNewGpu<int>(flop_array_num);
    #endif

    #if DEBUG_IFTGPU
    msg(3, "Launch %d kernels (%dx%d) on %d elements/kernel.\n",
        num_grid, dim_grid.x, dim_block.x, K_ELEMS_PER_TILE);
    #endif

    for (int i_grid = 0; i_grid < num_grid; i_grid++) {
        // Put the tile of K values into constant mem
        const int processed = i_grid * K_ELEMS_PER_TILE;
        const DataTraj * ktraj_tile = ktraj + processed;
        const int left_num = MIN(K_ELEMS_PER_TILE, num_k_processed - processed);
        #if DEBUG_IFTGPU
        msg(4, "Kernel %d processed: %d, left_num: %d\n",
            i_grid, processed, left_num);
        #endif

        // Number of ktraj to copy to constant memory
        cudaError e = cudaMemcpyToSymbol(Trajc, ktraj_tile, left_num *
                      sizeof(DataTraj), 0, cudaMemcpyDeviceToDevice);
        makeSure(e == cudaSuccess,
            "CUDA runtime failure on cudaMemcpyToSymbol.");

        // Must clear the memory for idata_r/k smaller than K_ELEMS_PER_TILE.
        if (num_k_processed < K_ELEMS_PER_TILE) {
            clearCuda<FLOAT_T>(idata_r, num_i);
            clearCuda<FLOAT_T>(idata_i, num_i);
        }
        iftGpuKernel<<< dim_grid, dim_block >>>
                       (idata_r, idata_i, kdata_r, kdata_i,
                        itraj, processed, fm, time, num_i, left_num
                        #if COMPUTE_FLOPS
                        , flop_array_d
                        #endif
                        );
    }
    cudaThreadSynchronize();

    #if COMPUTE_FLOPS
    mriCopyDeviceToHost<int>(flop_array, flop_array_d, flop_array_num);
    for (int i = 0; i < flop_array_num; i++) {
        getMriFlop()->flop_iftGpu += (double) flop_array[i];
    }
    //printf("*** # of threads: %d (%d * %d)\n", flop_array_num, num_block,
    //    IFT_BLOCK_SIZE);
    //printf("*** # of accumulated flop: %.f\n", getMriFlop()->flop_iftGpu);
    mriDeleteGpu(flop_array_d);
    mriDeleteCpu(flop_array);
    #endif // COMPUTE_FLOPS

    stopMriTimer(getMriTimer()->timer_iftGpu);
    #if DEBUG_IFTGPU
    msg(2, "iftGpu() end\n");
    #endif
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [GPU kernel of the Inverse Fourier Transformation (IFT).]    */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    __global__ void
iftGpuKernel(
    FLOAT_T *idata_r, FLOAT_T *idata_i,
    const FLOAT_T *kdata_r, const FLOAT_T *kdata_i,
    const DataTraj *itraj, int processed,
    const FLOAT_T *fm_, const FLOAT_T *time,
    const int num_i, const int left_num
    #if COMPUTE_FLOPS
    , int *flop_array
    #endif
    )
{
    // Determine the element of the X arrays computed by this thread
    const int i_idx = blockIdx.x * IFT_BLOCK_SIZE + threadIdx.x;

    // This is useful when the data is not padded to power of 2.
    if (i_idx >= num_i) { return; }

    FLOAT_T sumr = MRI_ZERO, sumi = MRI_ZERO;
    const FLOAT_T ktraj_z = MRI_ZERO;
    const FLOAT_T itraj_z = MRI_ZERO;
    const FLOAT_T tpi = MRI_PI * 2.0;                // done at compile time
    FLOAT_T kziztpi = ktraj_z * itraj_z * tpi;
    FLOAT_T fm = fm_[i_idx];
    FLOAT_T itraj_x_tpi = itraj[i_idx].x * tpi;  // flop: 1
    FLOAT_T itraj_y_tpi = itraj[i_idx].y * tpi;  // flop: 1

    #if COMPUTE_FLOPS
    flop_array[i_idx] += 1 + 1;
    #endif

    // Optimizations
    // Loop over all elements of K to compute a partial value for X.
    #if IFT_NO_SPEEDUP // ====================================================

    FLOAT_T expr = MRI_ZERO, cosexpr = MRI_ZERO, sinexpr = MRI_ZERO,
            kdatac_r = MRI_ZERO, kdatac_i = MRI_ZERO;
    for (int k_idx = 0; k_idx < left_num; k_idx++, processed++) {
        expr = (Trajc[k_idx].x * itraj_x_tpi +
                Trajc[k_idx].y * itraj_y_tpi + kziztpi) +
               (fm * time[processed]);
        #if 1
            #if ENABLE_DOUBLE_PRECISION
                cosexpr = cos(expr); sinexpr = sin(expr);
            #else
                cosexpr = cosf(expr); sinexpr = sinf(expr);
            #endif
        #else // No speedup
            cosexpr = MRI_ZERO; sinexpr = MRI_ZERO;
            sincosf(expr, &sinexpr, &cosexpr);
        #endif

        kdatac_r = kdata_r[processed]; kdatac_i = kdata_i[processed];
        sumr += (cosexpr * kdatac_r) - (sinexpr * kdatac_i);
        sumi += (sinexpr * kdatac_r) + (cosexpr * kdatac_i);
    }
    // COMPUTE_FLOPS is used to calculate the flops, not for estimating time.
    #elif COMPUTE_FLOPS
    // Note: This is defined in the Makefile.

    FLOAT_T expr, cosexpr, sinexpr, kdatac_r, kdatac_i;
    for (int k_idx = 0; k_idx < left_num; k_idx++, processed++) {
        expr = (Trajc[k_idx].x * itraj_x_tpi + Trajc[k_idx].y * itraj_y_tpi +
                kziztpi) + (fm * time[processed]);
        cosexpr = cosf(expr); sinexpr = sinf(expr);

        kdatac_r = kdata_r[processed]; kdatac_i = kdata_i[processed];
        sumr += (cosexpr*kdatac_r) - (sinexpr*kdatac_i);
        sumi += (sinexpr*kdatac_r) + (cosexpr*kdatac_i);

        #if COMPUTE_FLOPS
        // sin: 13 flop; cos: 12 flop; +,-,*,/: 1 flop
        flop_array[i_idx] += (6 + 12 + 13) + (5 + 5);
        #endif
    }
    #elif IFT_LOOP_UNROLLING // ==============================================
    // Note: The kdata must be padded for this optimization.

        #if 0 // unrolling version 1
        #define IFT_UNROLL_FACTOR  4 // 2, 4, 8
        FLOAT_T expr[IFT_UNROLL_FACTOR], cosexpr[IFT_UNROLL_FACTOR],
                sinexpr[IFT_UNROLL_FACTOR],
                sum_r[IFT_UNROLL_FACTOR], sum_i[IFT_UNROLL_FACTOR];
        FLOAT_T kdatac_r, kdatac_i;
        for (int k_idx = 0; k_idx < left_num;
            k_idx+=IFT_UNROLL_FACTOR, processed+=IFT_UNROLL_FACTOR) {
            expr[0] = (Trajc[k_idx+0].x * itraj_x_tpi + Trajc[k_idx+0].y *
                       itraj_y_tpi + kziztpi) + (fm * time[processed+0]);
            cosexpr[0] = cosf(expr[0]); sinexpr[0] = sinf(expr[0]);
            kdatac_r = kdata_r[processed+0]; kdatac_i = kdata_i[processed+0];
            sum_r[0] = (cosexpr[0]*kdatac_r) - (sinexpr[0]*kdatac_i);
            sum_i[0] = (sinexpr[0]*kdatac_r) + (cosexpr[0]*kdatac_i);
    
            expr[1] = (Trajc[k_idx+1].x * itraj_x_tpi + Trajc[k_idx+1].y *
                       itraj_y_tpi + kziztpi) + (fm * time[processed+1]);
            cosexpr[1] = cosf(expr[1]); sinexpr[1] = sinf(expr[1]);
            kdatac_r = kdata_r[processed+1]; kdatac_i = kdata_i[processed+1];
            sum_r[1] = (cosexpr[1]*kdatac_r) - (sinexpr[1]*kdatac_i);
            sum_i[1] = (sinexpr[1]*kdatac_r) + (cosexpr[1]*kdatac_i);
    
            #if IFT_UNROLL_FACTOR == 2
            sumr += sum_r[0] + sum_r[1];
            sumi += sum_i[0] + sum_i[1];
            #endif
    
            #if IFT_UNROLL_FACTOR > 2
            expr[2] = (Trajc[k_idx+2].x * itraj_x_tpi + Trajc[k_idx+2].y *
                       itraj_y_tpi + kziztpi) + (fm * time[processed+2]);
            cosexpr[2] = cosf(expr[2]); sinexpr[2] = sinf(expr[2]);
            kdatac_r = kdata_r[processed+2]; kdatac_i = kdata_i[processed+2];
            sum_r[2] = (cosexpr[2]*kdatac_r) - (sinexpr[2]*kdatac_i);
            sum_i[2] = (sinexpr[2]*kdatac_r) + (cosexpr[2]*kdatac_i);
    
            expr[3] = (Trajc[k_idx+3].x * itraj_x_tpi + Trajc[k_idx+3].y *
                       itraj_y_tpi + kziztpi) + (fm * time[processed+3]);
            cosexpr[3] = cosf(expr[3]); sinexpr[3] = sinf(expr[3]);
            kdatac_r = kdata_r[processed+3]; kdatac_i = kdata_i[processed+3];
            sum_r[3] = (cosexpr[3]*kdatac_r) - (sinexpr[3]*kdatac_i);
            sum_i[3] = (sinexpr[3]*kdatac_r) + (cosexpr[3]*kdatac_i);
            #endif
    
            #if IFT_UNROLL_FACTOR == 4
            sumr += sum_r[0] + sum_r[1] + sum_r[2] + sum_r[3];
            sumi += sum_i[0] + sum_i[1] + sum_i[2] + sum_i[3];
            #endif
    
            #if IFT_UNROLL_FACTOR > 4
            expr[4] = (Trajc[k_idx+4].x * itraj_x_tpi + Trajc[k_idx+4].y *
                       itraj_y_tpi + kziztpi) + (fm * time[processed+4]);
            cosexpr[4] = cosf(expr[4]); sinexpr[4] = sinf(expr[4]);
            kdatac_r = kdata_r[processed+4]; kdatac_i = kdata_i[processed+4];
            sum_r[4] = (cosexpr[4]*kdatac_r) - (sinexpr[4]*kdatac_i);
            sum_i[4] = (sinexpr[4]*kdatac_r) + (cosexpr[4]*kdatac_i);
    
            expr[5] = (Trajc[k_idx+5].x * itraj_x_tpi + Trajc[k_idx+5].y *
                       itraj_y_tpi + kziztpi) + (fm * time[processed+5]);
            cosexpr[5] = cosf(expr[5]); sinexpr[5] = sinf(expr[5]);
            kdatac_r = kdata_r[processed+5]; kdatac_i = kdata_i[processed+5];
            sum_r[5] = (cosexpr[5]*kdatac_r) - (sinexpr[5]*kdatac_i);
            sum_i[5] = (sinexpr[5]*kdatac_r) + (cosexpr[5]*kdatac_i);
    
            expr[6] = (Trajc[k_idx+6].x * itraj_x_tpi + Trajc[k_idx+6].y *
                       itraj_y_tpi + kziztpi) + (fm * time[processed+6]);
            cosexpr[6] = cosf(expr[6]); sinexpr[6] = sinf(expr[6]);
            kdatac_r = kdata_r[processed+6]; kdatac_i = kdata_i[processed+6];
            sum_r[6] = (cosexpr[6]*kdatac_r) - (sinexpr[6]*kdatac_i);
            sum_i[6] = (sinexpr[6]*kdatac_r) + (cosexpr[6]*kdatac_i);
    
            expr[7] = (Trajc[k_idx+7].x * itraj_x_tpi + Trajc[k_idx+7].y *
                       itraj_y_tpi + kziztpi) + (fm * time[processed+7]);
            cosexpr[7] = cosf(expr[7]); sinexpr[7] = sinf(expr[7]);
            kdatac_r = kdata_r[processed+7]; kdatac_i = kdata_i[processed+7];
            sum_r[7] = (cosexpr[7]*kdatac_r) - (sinexpr[7]*kdatac_i);
            sum_i[7] = (sinexpr[7]*kdatac_r) + (cosexpr[7]*kdatac_i);
            #endif
    
            #if IFT_UNROLL_FACTOR == 8
            sumr += sum_r[0] + sum_r[1] + sum_r[2] + sum_r[3] +
                    sum_r[4] + sum_r[5] + sum_r[6] + sum_r[7];
            sumi += sum_i[0] + sum_i[1] + sum_i[2] + sum_i[3] +
                    sum_i[4] + sum_i[5] + sum_i[6] + sum_i[7];
            #endif
        }
        
        #else // Unrolling version 2 =========================================
        #define IFT_UNROLL_FACTOR  8 // 2, 4, 8
        FLOAT_T expr[IFT_UNROLL_FACTOR], cosexpr[IFT_UNROLL_FACTOR],
                sinexpr[IFT_UNROLL_FACTOR];
        FLOAT_T kdatac_r, kdatac_i;
        for (int k_idx = 0; k_idx < left_num;
            k_idx+=IFT_UNROLL_FACTOR, processed+=IFT_UNROLL_FACTOR) {
            expr[0] = (Trajc[k_idx+0].x * itraj_x_tpi + Trajc[k_idx+0].y *
                       itraj_y_tpi + kziztpi) + (fm * time[processed+0]);
            cosexpr[0] = cosf(expr[0]); sinexpr[0] = sinf(expr[0]);
            kdatac_r = kdata_r[processed+0]; kdatac_i = kdata_i[processed+0];
            sumr += (cosexpr[0]*kdatac_r) - (sinexpr[0]*kdatac_i);
            sumi += (sinexpr[0]*kdatac_r) + (cosexpr[0]*kdatac_i);
    
            expr[1] = (Trajc[k_idx+1].x * itraj_x_tpi + Trajc[k_idx+1].y *
                       itraj_y_tpi + kziztpi) + (fm * time[processed+1]);
            cosexpr[1] = cosf(expr[1]); sinexpr[1] = sinf(expr[1]);
            kdatac_r = kdata_r[processed+1]; kdatac_i = kdata_i[processed+1];
            sumr += (cosexpr[1]*kdatac_r) - (sinexpr[1]*kdatac_i);
            sumi += (sinexpr[1]*kdatac_r) + (cosexpr[1]*kdatac_i);

            #if IFT_UNROLL_FACTOR > 2
            expr[2] = (Trajc[k_idx+2].x * itraj_x_tpi + Trajc[k_idx+2].y *
                       itraj_y_tpi + kziztpi) + (fm * time[processed+2]);
            cosexpr[2] = cosf(expr[2]); sinexpr[2] = sinf(expr[2]);
            kdatac_r = kdata_r[processed+2]; kdatac_i = kdata_i[processed+2];
            sumr += (cosexpr[2]*kdatac_r) - (sinexpr[2]*kdatac_i);
            sumi += (sinexpr[2]*kdatac_r) + (cosexpr[2]*kdatac_i);
    
            expr[3] = (Trajc[k_idx+3].x * itraj_x_tpi + Trajc[k_idx+3].y *
                       itraj_y_tpi + kziztpi) + (fm * time[processed+3]);
            cosexpr[3] = cosf(expr[3]); sinexpr[3] = sinf(expr[3]);
            kdatac_r = kdata_r[processed+3]; kdatac_i = kdata_i[processed+3];
            sumr += (cosexpr[3]*kdatac_r) - (sinexpr[3]*kdatac_i);
            sumi += (sinexpr[3]*kdatac_r) + (cosexpr[3]*kdatac_i);
            #endif

            #if IFT_UNROLL_FACTOR > 4
            expr[4] = (Trajc[k_idx+4].x * itraj_x_tpi + Trajc[k_idx+4].y *
                       itraj_y_tpi + kziztpi) + (fm * time[processed+4]);
            cosexpr[4] = cosf(expr[4]); sinexpr[4] = sinf(expr[4]);
            kdatac_r = kdata_r[processed+4]; kdatac_i = kdata_i[processed+4];
            sumr += (cosexpr[4]*kdatac_r) - (sinexpr[4]*kdatac_i);
            sumi += (sinexpr[4]*kdatac_r) + (cosexpr[4]*kdatac_i);
    
            expr[5] = (Trajc[k_idx+5].x * itraj_x_tpi + Trajc[k_idx+5].y *
                       itraj_y_tpi + kziztpi) + (fm * time[processed+5]);
            cosexpr[5] = cosf(expr[5]); sinexpr[5] = sinf(expr[5]);
            kdatac_r = kdata_r[processed+5]; kdatac_i = kdata_i[processed+5];
            sumr += (cosexpr[5]*kdatac_r) - (sinexpr[5]*kdatac_i);
            sumi += (sinexpr[5]*kdatac_r) + (cosexpr[5]*kdatac_i);
    
            expr[6] = (Trajc[k_idx+6].x * itraj_x_tpi + Trajc[k_idx+6].y *
                       itraj_y_tpi + kziztpi) + (fm * time[processed+6]);
            cosexpr[6] = cosf(expr[6]); sinexpr[6] = sinf(expr[6]);
            kdatac_r = kdata_r[processed+6]; kdatac_i = kdata_i[processed+6];
            sumr += (cosexpr[6]*kdatac_r) - (sinexpr[6]*kdatac_i);
            sumi += (sinexpr[6]*kdatac_r) + (cosexpr[6]*kdatac_i);
    
            expr[7] = (Trajc[k_idx+7].x * itraj_x_tpi + Trajc[k_idx+7].y *
                       itraj_y_tpi + kziztpi) + (fm * time[processed+7]);
            cosexpr[7] = cosf(expr[7]); sinexpr[7] = sinf(expr[7]);
            kdatac_r = kdata_r[processed+7]; kdatac_i = kdata_i[processed+7];
            sumr += (cosexpr[7]*kdatac_r) - (sinexpr[7]*kdatac_i);
            sumi += (sinexpr[7]*kdatac_r) + (cosexpr[7]*kdatac_i);
            #endif
        }
        #endif
    #endif

    if (i_idx < num_i) {
        // For first kernel launch, assign them directly.
        // Warning: You must make sure the idata_r/i size is larger than
        //          K_ELEMS_PER_TILE or you have to clear idata_r/i beforehand.
        if (processed == K_ELEMS_PER_TILE) {
            idata_r[i_idx] = sumr;
            idata_i[i_idx] = sumi;
        } else { // For the rest kernel launches, accumulate them.
            idata_r[i_idx] += sumr;
            idata_i[i_idx] += sumi;

            #if COMPUTE_FLOPS
            flop_array[i_idx] += 2;
            #endif
        }
    } else { // When idata is not padded.
        idata_r[i_idx] = MRI_ZERO;
        idata_i[i_idx] = MRI_ZERO;
    }

    // Debugging the flop count. The collected flop_array[] elements should be
    // the total number of valid threads.
    //flop_array[i_idx] = 1;
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

