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

    File Name   [addGpu.cu]

    Synopsis    [GPU version of the complex number vector addition.]

    Description []

    Revision    [0.1; Initial build; Xiao-Long Wu, ECE UIUC]
    Revision    [0.1.1; Calculating FLOPS, Code cleaning;
                 Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Further optimization, Code cleaning, Adding comments;
                 Xiao-Long Wu, ECE UIUC]
    Date        [10/27/2010]

*****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>
#include <stdio.h>

// CUDA libraries
#include <cutil.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>

// Project header files
#include <structures.h>
#include <addGpu.cuh>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Interface to the vector addition GPU kernel.]               */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

// Kernel debugging macros
#if DEBUG_KERNEL_MSG
    #define DEBUG_ADDGPU  true
#else
    #define DEBUG_ADDGPU  false
#endif

#define addGpu_BLOCK_SIZE  256

    __global__ void
addGpuKernel(
    FLOAT_T *output_r, FLOAT_T *output_i,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i,
    const FLOAT_T alpha, const int num_elements);

    void
addGpu(
    FLOAT_T *output_r, FLOAT_T *output_i,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i,
    const FLOAT_T alpha, const int num_elements
    )
{
    #if DEBUG_ADDGPU
    msg(2, "addGpu() begin\n");
    #endif
    startMriTimer(getMriTimer()->timer_addGpu);

    makeSure(num_elements <= addGpu_BLOCK_SIZE * 65535,
        "Maximum supported vector size is 64K * 256 or 256^3.");

    int num_Blocks = ceil((FLOAT_T) num_elements /
                          (FLOAT_T) addGpu_BLOCK_SIZE);

    // Launch the device computation threads
    addGpuKernel <<< dim3(num_Blocks), dim3(addGpu_BLOCK_SIZE) >>>
                   (output_r, output_i, a_r, a_i, b_r, b_i, alpha, 
                    num_elements);

    #if COMPUTE_FLOPS
    getMriFlop()->flop_addGpu += num_elements * (2 + 2);
    #endif

    stopMriTimer(getMriTimer()->timer_addGpu);
    #if DEBUG_ADDGPU
    msg(2, "addGpu() end\n");
    #endif
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Vector addition GPU kernel.]                                */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    __global__ void
addGpuKernel(
    FLOAT_T *output_r, FLOAT_T *output_i,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i,
    const FLOAT_T alpha, const int num_elements)
{
    int i = blockIdx.x * addGpu_BLOCK_SIZE + threadIdx.x;

    if (i < num_elements) {
        output_r[i] = a_r[i] + alpha * b_r[i];
        output_i[i] = a_i[i] + alpha * b_i[i];
    }
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

