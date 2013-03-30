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

    File Name   [multiplyGpu.cu]

    Synopsis    [GPU version of the complex number vector multiplication.]

    Description []

    Revision    [1.0a; Initial build; Xiao-Long Wu, ECE UIUC]
    Date        [10/25/2010]

 *****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <stdio.h>
#include <string.h>
#include <math.h>

// CUDA libraries
#include <cutil.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>

// Project header files
#include <tools.h>
#include <structures.h>

#include <multiplyGpu.cuh>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    []                                                           */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

// Kernel debugging macros
#if DEBUG_KERNEL_MSG
    #define DEBUG_MULTIPLYGPU  true
#else
    #define DEBUG_MULTIPLYGPU  false
#endif

#define multiplyGpu_BLOCK_SIZE  256

    __global__ void
multiplyGpuKernel(
    FLOAT_T *output_r,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i,
    const int num_elements);

    void
multiplyGpu(
    FLOAT_T *output_r,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i,
    const int num_elements
    )
{
    #if DEBUG_MULTIPLYGPU
    msg(2, "multiplyGpu() begin\n");
    #endif
    startMriTimer(getMriTimer()->timer_multiplyGpu);

    makeSure(num_elements <= multiplyGpu_BLOCK_SIZE * 65535,
        "Maximum supported vector size is 64K * 256 or 256^3.");

    int num_Blocks = ceil((FLOAT_T) num_elements /
                          (FLOAT_T) multiplyGpu_BLOCK_SIZE);

    // Setup the execution configuration
    dim3 dimBlock(multiplyGpu_BLOCK_SIZE, 1);
    dim3 dimGrid(num_Blocks, 1);

    // Launch the device computation threads
    multiplyGpuKernel <<< dimGrid, dimBlock >>>
                      (output_r, a_r, a_i, b_r, b_i, num_elements);
    cudaThreadSynchronize();

    #if COMPUTE_FLOPS // sqrt: 13 flops, +,-,*,/: 1 flop
    getMriFlop()->flop_multiplyGpu += num_elements * (1 + 13 + 8);
    #endif

    stopMriTimer(getMriTimer()->timer_multiplyGpu);
    #if DEBUG_MULTIPLYGPU
    msg(2, "multiplyGpu() end\n");
    #endif
}

    __global__ void
multiplyGpuKernel(
    FLOAT_T *output_r,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i,
    const int num_elements)
{
    int i = blockIdx.x * multiplyGpu_BLOCK_SIZE + threadIdx.x;

    if (i < num_elements) {
        const FLOAT_T a_r_tmp = a_r[i], a_i_tmp = a_i[i];
        const FLOAT_T b_r_tmp = b_r[i], b_i_tmp = b_i[i];
        output_r[i] = MRI_POINT_FIVE /
                      sqrt(a_r_tmp*a_r_tmp + a_i_tmp*a_i_tmp +
                           b_r_tmp*b_r_tmp + b_i_tmp*b_i_tmp +
                           MRI_SMOOTH_FACTOR);
    }
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

