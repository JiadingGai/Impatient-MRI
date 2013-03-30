/**************************************************************************
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
 *****************************************************************************/

/*****************************************************************************

    File Name   [Dhori2dGpu.cu]

    Revision    [0.1; Initial build; Fan Lam, Mao-Jing Fu, ECE UIUC]
    Date        [10/25/2010]

*****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <stdio.h>
#include <string.h>
#include <assert.h>

// Project header files
#include <cufft.h>
#include "Dhori2dGpu_toeplitz.cuh"

#define Dhori_BLOCK_Y      256
#define Dhori_BLOCK_X_2D   16
#define Dhori_BLOCK_Y_2D   16

__global__ void Dhori2dGpuKernel1(
    cufftComplex *s, cufftComplex *p,
    const unsigned int num_row, const unsigned int num_col);

__global__ void Dhori2dGpuKernel2(
    cufftComplex *s, cufftComplex *p,
    const unsigned int num_row, const unsigned int num_col);


    void
Dhori2dGpu( matrix_t &s, matrix_t &p )
{
    const unsigned int num_row = p.dims[0];
    const unsigned int num_col = p.dims[0];
//    const unsigned int num = num_row*num_col;

    int num_blocks_y = ceil((float) num_row / (float) Dhori_BLOCK_Y);
    int num_blocks_x_2d = ceil((float) num_col / (float) Dhori_BLOCK_X_2D);
    int num_blocks_y_2d = ceil((float) num_row / (float) Dhori_BLOCK_Y_2D);

    assert( num_blocks_y <= 65535 );
    assert( num_blocks_x_2d * num_blocks_y_2d <= 65535 );

    Dhori2dGpuKernel1 <<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                         dim3(Dhori_BLOCK_X_2D, Dhori_BLOCK_Y_2D)>>>
        (s.device, p.device, num_row, num_col);

    Dhori2dGpuKernel2 <<<dim3(num_blocks_y), dim3(Dhori_BLOCK_Y)>>>
        (s.device, p.device, num_row, num_col);
}
__global__ void Dhori2dGpuKernel1(
    cufftComplex *s, cufftComplex *p,
    const unsigned int num_row, const unsigned int num_col)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;// x axis
    int y = blockIdx.y * blockDim.y + threadIdx.y;// y axis

    if( x < (num_col-1) )
    {
        // LHS(y,x) = RHS(y,x) - RHS(y,x+1)
        const unsigned int y_x = y*num_col+x;
        const unsigned int y_x1 = y_x + 1;
        s[y_x]REAL = p[y_x]REAL - p[y_x1]REAL;
        s[y_x]IMAG = p[y_x]IMAG - p[y_x1]IMAG;
   }
}

__global__ void Dhori2dGpuKernel2(
    cufftComplex *s, cufftComplex *p,
    const unsigned int num_row, const unsigned int num_col)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;//y axis

    if(y<num_row)
    {
        // LHS(y,num_col-1) = RHS(y,num_col-1) - RHS(y,0)
        const unsigned int y_num_col_1 = y*num_col+(num_col-1);
        const unsigned int y_zero = y*num_col;
        s[y_num_col_1]REAL = p[y_num_col_1]REAL - p[y_zero]REAL;
        s[y_num_col_1]IMAG = p[y_num_col_1]IMAG - p[y_zero]IMAG;
   }
}
