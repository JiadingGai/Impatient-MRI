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

    File Name   [dotProductGpu.cu]

    Synopsis    [GPU version of the complex number dot product.]

    Description []

    Revision    [0.1; Initial build; Xiao-Long Wu, ECE UIUC]
    Revision    [0.1.1; Calculating FLOPS, Code cleaning;
                 Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Further optimization, Code cleaning, Adding more comments;
                 Xiao-Long Wu, ECE UIUC]
    Date        [10/27/2010]

 *****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// CUDA libraries
#include <cutil_inline.h>

// Project header files
#include <xcpplib_process.h>
#include <xcpplib_typesGpu.cuh>
#include <structures.h>

#include <dotProductGpu.cuh>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [GPU entry of the complex number dot product.]               */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

// Kernel debugging macros
#if DEBUG_KERNEL_MSG
    #define DEBUG_DOTPRODUCTGPU  true
#else
    #define DEBUG_DOTPRODUCTGPU  false
#endif

// dotProductGpu kernel performance tuning factor
#define dotProductGpu_BLOCK_SIZE 256

    __global__ void
dotProductGpuKernel(
    FLOAT_T *output,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i, const int num_elements);

    void
dotProductGpu(
    FLOAT_T *output,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i,
    const int num_elements)
{
    #if DEBUG_DOTPRODUCTGPU
    msg(2, "dotProductGpu() begin\n");
    #endif
    startMriTimer(getMriTimer()->timer_dotProductGpu);

    makeSure(num_elements <= dotProductGpu_BLOCK_SIZE * 65535,
        "Maximum supported vector size is 64K * 256 or 256^3.");

    // dotProductGpuKernel() =================================================
    // Perform real part inner product of all vector pairs.

    #if DEBUG_DOTPRODUCTGPU
    msg(3, "dotProductGpuKernel()\n");
    #endif

    const int num_blocks = ceil((FLOAT_T) num_elements /
                                (FLOAT_T) dotProductGpu_BLOCK_SIZE);

    FLOAT_T *dot_product = mriNewGpu<FLOAT_T>(num_elements);

    dim3 dimBlock(dotProductGpu_BLOCK_SIZE, 1);
    dim3 dimGrid(num_blocks, 1);

    #if COMPUTE_FLOPS
    // 3 operations per pair.
    getMriFlop()->flop_dotProductGpu += (num_elements * 3);
    #endif
    dotProductGpuKernel <<< dimGrid, dimBlock >>>
                            (dot_product, a_r, a_i, b_r, b_i, num_elements);

    // reductionArray() ======================================================
    // Perform reductions.

    #if DEBUG_DOTPRODUCTGPU
    msg(3, "reductionArray()\n");
    #endif

    reductionArray(dot_product, dot_product, num_elements,
        &getMriFlop()->flop_dotProductGpu);

    mriCopyDeviceToHost<FLOAT_T>(output, dot_product, 1);
    mriDeleteGpu(dot_product);

    stopMriTimer(getMriTimer()->timer_dotProductGpu);
    #if DEBUG_DOTPRODUCTGPU
    msg(2, "dotProductGpu() end\n");
    #endif
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [GPU kernel of the complex number dot product.]              */
/*                                                                           */
/*  Description [Perform real part inner product of all vector pairs.]       */
/*                                                                           */
/*===========================================================================*/

    __global__ void
dotProductGpuKernel(
    FLOAT_T *output,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i, const int num_elements)
{
    const int idx = dotProductGpu_BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (idx < num_elements) {
        output[idx] = a_r[idx] * b_r[idx] + a_i[idx] * b_i[idx];
    }
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [GPU reduction entry for unlimited elements.]                */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

#define reductionArray_DEBUG  false

#define reduction_BLOCK_SIZE  256

    __global__ void 
reductionStep(
    FLOAT_T *output, const FLOAT_T *input, const int num
    #if COMPUTE_FLOPS
    , int *flop_array
    #endif
    );

    void
reductionArray(
    FLOAT_T *outArray, const FLOAT_T *inArray, const int num_elements,
    unsigned int *flops)
{
    dim3 grid((num_elements + reduction_BLOCK_SIZE * 2 - 1) /
              (reduction_BLOCK_SIZE *2), 1, 1);
    dim3 block(reduction_BLOCK_SIZE, 1, 1);

    #if COMPUTE_FLOPS
    const int flop_array_num = reduction_BLOCK_SIZE * 2;
    int *flop_array = mriNewCpu<int>(flop_array_num);
    int *flop_array_d = mriNewGpu<int>(flop_array_num);
    #endif

    #if reductionArray_DEBUG
    printf("\nreductionArray(): gridDim.x: %d, blockDim.x: %d\n",
        grid.x, block.x);
    #endif
    for (int i = num_elements; i > 1; i = grid.x) {
        grid.x = (i + reduction_BLOCK_SIZE*2 - 1) / (reduction_BLOCK_SIZE*2);
        #if reductionArray_DEBUG
        printf("  i: %d, num_elements: %d, grid.x: %d\n",
            i, num_elements, grid.x);
        #endif
        reductionStep <<< grid, block >>>
            (outArray, i == num_elements ? inArray : outArray, i
             #if COMPUTE_FLOPS
             , flop_array_d
             #endif
            );
        cutilCheckMsg("reductionStep() execution failed\n");
    }

    #if COMPUTE_FLOPS
    mriCopyDeviceToHost<int>(flop_array, flop_array_d, flop_array_num);
    mriDeleteGpu(flop_array_d);
    for (int i = 0; i < flop_array_num; i++) {
        *flops += (double) flop_array[i];
    }
    mriDeleteCpu(flop_array);
    #endif
}

    __global__ void 
reductionStep(
    FLOAT_T *output, const FLOAT_T *input, const int num
    #if COMPUTE_FLOPS
    , int *flop_array
    #endif
    )
{
    __shared__ FLOAT_T scratch[reduction_BLOCK_SIZE * 2];

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;

    // Coalesced loads to shared memory.
    int gx = (bx * reduction_BLOCK_SIZE * 2 + tx);

    scratch[tx] = gx < num ? input[gx] : MRI_ZERO;
    scratch[tx + reduction_BLOCK_SIZE] = (gx + reduction_BLOCK_SIZE) < num ?
                                input[gx + reduction_BLOCK_SIZE] : MRI_ZERO;
    __syncthreads();

    for (int stride = reduction_BLOCK_SIZE; stride > 0; stride >>= 1) {
        if (tx < stride) {
            scratch[tx] += scratch[tx + stride];

            #if COMPUTE_FLOPS
            flop_array[tx] += 1;
            #endif
        }
        __syncthreads();
    }

    // write results to global memory
    if (tx == 0) output[bx] = scratch[0];
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

