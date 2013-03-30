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
#include <assert.h>

// Project header files
#include <cufft.h>
#include "multiplyGpu_toeplitz.cuh"

#define multiplyGpu_BLOCK_SIZE  256

__global__ void
multiplyGpuKernel2d(
    float *output_r,
    cufftComplex *a, cufftComplex *b,
    const int num_elements);

__global__ void
multiplyGpuKernel3d(
    float *output_r,
    cufftComplex *a, cufftComplex *b, cufftComplex *c,
    const int num_elements);


void
multiply2dGpu(float *output_r,
            matrix_t&a, matrix_t&b)
{
    const unsigned int num_elements = a.elems;

    assert(num_elements <= multiplyGpu_BLOCK_SIZE * 65535);

    int num_Blocks = ceil((float) num_elements /
                          (float) multiplyGpu_BLOCK_SIZE);

    // Setup the execution configuration
    dim3 dimBlock(multiplyGpu_BLOCK_SIZE, 1);
    dim3 dimGrid(num_Blocks, 1);

    // Launch the device computation threads
    multiplyGpuKernel2d <<< dimGrid, dimBlock >>>
    (output_r, a.device, b.device, num_elements);
    cudaThreadSynchronize();
}

void
multiply3dGpu(float *output_r,
            matrix_t&a, matrix_t&b, matrix_t&c)
{
    const unsigned int num_elements = a.elems;

    int num_Blocks_x = ceil((float) num_elements /
                            (float) multiplyGpu_BLOCK_SIZE);
    int num_Blocks_y = 1;

    while(num_Blocks_x>32768) {
         num_Blocks_y *= 2;
         if(num_Blocks_x%2) {
            num_Blocks_x /= 2;
            num_Blocks_x++;
         }
         else {
            num_Blocks_x /= 2;
         } 
    }

    // Setup the execution configuration
    dim3 dimBlock(multiplyGpu_BLOCK_SIZE, 1);
    dim3 dimGrid(num_Blocks_x, num_Blocks_y);

    // Launch the device computation threads
    multiplyGpuKernel3d <<< dimGrid, dimBlock >>>
    (output_r, a.device, b.device, c.device, num_elements);
    cudaThreadSynchronize();
}



__global__ void
multiplyGpuKernel2d(
    float *output_r,
    cufftComplex *a, cufftComplex *b,
    const int num_elements)
{
    int i = blockIdx.x * multiplyGpu_BLOCK_SIZE + threadIdx.x;

    if (i < num_elements) {
        const float a_r_tmp = a[i] REAL, a_i_tmp = a[i] IMAG;
        const float b_r_tmp = b[i] REAL, b_i_tmp = b[i] IMAG;
        output_r[i] = 0.5f /
                      sqrt(a_r_tmp * a_r_tmp + a_i_tmp * a_i_tmp +
                           b_r_tmp * b_r_tmp + b_i_tmp * b_i_tmp +
                           0.000001f);
    }
}


__global__ void
multiplyGpuKernel3d(
    float *output_r,
    cufftComplex *a, cufftComplex *b, cufftComplex *c,
    const int num_elements)
{
    int i = blockIdx.y * (gridDim.x * multiplyGpu_BLOCK_SIZE)
          + blockIdx.x * multiplyGpu_BLOCK_SIZE + threadIdx.x;

    if (i < num_elements) {
        const float a_r_tmp = a[i] REAL, a_i_tmp = a[i] IMAG;
        const float b_r_tmp = b[i] REAL, b_i_tmp = b[i] IMAG;
        const float c_r_tmp = c[i] REAL, c_i_tmp = c[i] IMAG;
        output_r[i] = 0.5f /
                      sqrt(a_r_tmp * a_r_tmp + a_i_tmp * a_i_tmp +
                           b_r_tmp * b_r_tmp + b_i_tmp * b_i_tmp +
                           c_r_tmp * c_r_tmp + c_i_tmp * c_i_tmp +
                           0.000001f);
    }
}
