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

    File Name   [pointMultGpu.cu]

    Synopsis    [GPU version of the element-wise complex vector multiply.]

    Description []

    Revision    [1.0a; Initial build; Jiading Gai, Beckman Institute,
                 Xiao-Long Wu, ECE UIUC]
    Date        [10/24/2010]

 *****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <stdio.h>
#include <string.h>

// CUDA libraries
#include <cutil.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>

// Project header files
#include <structures.h>

#include <pointMultGpu.cuh>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Macro definitions                                                        */
/*---------------------------------------------------------------------------*/

#define pointMult_BLOCK_SIZE  256

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [GPU entry of the element-wise product (complex number       */
/*      vectors)]                                                            */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

// Kernel debugging macros
#if DEBUG_KERNEL_MSG
    #define DEBUG_POINTMULTGPU  true
#else
    #define DEBUG_POINTMULTGPU  false
#endif

    __global__ void
pointMultGpu_kernel(
    FLOAT_T *result_r, FLOAT_T *result_i,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i, const int num_elements);

    void
pointMultGpu(
    FLOAT_T *result_r, FLOAT_T *result_i,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i, const int num_elements)
{
    #if DEBUG_POINTMULTGPU
    msg(2, "pointMultGpu() begin\n");
    #endif
    startMriTimer(getMriTimer()->timer_pointMultGpu);

    makeSure(num_elements <= pointMult_BLOCK_SIZE * 65535,
        "Maximum supported vector size is 64K * 256 or 256^3.");

    const int num_Blocks = ceil((FLOAT_T) num_elements /
                                (FLOAT_T) pointMult_BLOCK_SIZE);
    dim3 dimBlock(pointMult_BLOCK_SIZE, 1);
    dim3 dimGrid(num_Blocks, 1);

    pointMultGpu_kernel <<< dimGrid, dimBlock >>>
                        (result_r, result_i, a_r, a_i, b_r, b_i,
                         num_elements);
    cudaThreadSynchronize();

    stopMriTimer(getMriTimer()->timer_pointMultGpu);
    #if DEBUG_POINTMULTGPU
    msg(2, "pointMultGpu() end\n");
    #endif
}

    __global__ void
pointMultGpu_kernel(
    FLOAT_T *result_r, FLOAT_T *result_i,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i, const int num_elements)
{
    const int i = threadIdx.x + blockIdx.x * pointMult_BLOCK_SIZE;

    if (i < num_elements) {
        const FLOAT_T a_r_tmp = a_r[i], a_i_tmp = a_i[i];
        const FLOAT_T b_r_tmp = b_r[i], b_i_tmp = b_i[i];
        result_r[i] = a_r_tmp * b_r_tmp - a_i_tmp * b_i_tmp;
        result_i[i] = a_r_tmp * b_i_tmp + a_i_tmp * b_r_tmp;
    }
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [GPU entry of the element-wise conjugate product (complex    */
/*      number vectors: [sen^H * data])]                                     */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    __global__ void
pointMult_conjGpu_kernel(
    FLOAT_T *result_r, FLOAT_T *result_i,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i, const int num_elements);

    void
pointMult_conjGpu(
    FLOAT_T *result_r, FLOAT_T *result_i,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i, const int num_elements)
{
    #if DEBUG_POINTMULTGPU
    msg(2, "pointMult_conjGpu() begin\n");
    #endif

    makeSure(num_elements <= pointMult_BLOCK_SIZE * 65535,
        "Maximum supported vector size is 64K * 256 or 256^3.");

    const int num_Blocks = ceil((FLOAT_T) num_elements /
                                (FLOAT_T) pointMult_BLOCK_SIZE);
    dim3 dimBlock(pointMult_BLOCK_SIZE, 1);
    dim3 dimGrid(num_Blocks, 1);

    pointMult_conjGpu_kernel <<< dimGrid, dimBlock >>>
                             (result_r, result_i, a_r, a_i, b_r, b_i,
                              num_elements);

    cudaThreadSynchronize();

    #if DEBUG_POINTMULTGPU
    msg(2, "pointMult_conjGpu() end\n");
    #endif
}

    __global__ void
pointMult_conjGpu_kernel(
    FLOAT_T *result_r, FLOAT_T *result_i,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i, const int num_elements)
{
    const int i = threadIdx.x + blockIdx.x * pointMult_BLOCK_SIZE;

    if (i < num_elements) {
        const FLOAT_T a_r_tmp = a_r[i], a_i_tmp = a_i[i];
        const FLOAT_T b_r_tmp = b_r[i], b_i_tmp = b_i[i];
        result_r[i] = a_r_tmp * b_r_tmp + a_i_tmp * b_i_tmp;
        result_i[i] = a_r_tmp * b_i_tmp - a_i_tmp * b_r_tmp;
    }
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

