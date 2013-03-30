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

    File Name   [Dzaxis3dGpu.cu]

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
#include "Dzaxis3dGpu_toeplitz.cuh"

#define Dzaxis_BLOCK_X      128
#define Dzaxis_BLOCK_X_2D   16
#define Dzaxis_BLOCK_Z_2D   16

__global__ void Dzaxis3dGpuKernel1(
    cufftComplex *s, cufftComplex *p,
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep);

__global__ void Dzaxis3dGpuKernel2(
    cufftComplex *s, cufftComplex *p,
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep);

    void
Dzaxis3dGpu( matrix_t &s, matrix_t &p )
{
    const unsigned int num_row = p.dims[0];// y dimension
    const unsigned int num_col = p.dims[1];// x dimension
    const unsigned int num_dep = p.dims[2];// z dimension
    const unsigned int num = num_row*num_col*num_dep;

    int num_blocks_x = ceil((float) num_col / (float) Dzaxis_BLOCK_X);
    int num_blocks_x_2d = ceil((float) num_col / (float) Dzaxis_BLOCK_X_2D);
    int num_blocks_z_2d = ceil((float) num_dep / (float) Dzaxis_BLOCK_Z_2D);

    assert( num_blocks_x <= 65535 );
    assert( num_blocks_x_2d * num_blocks_z_2d <= 65535 );

    Dzaxis3dGpuKernel1 <<<dim3(num_blocks_x_2d, num_blocks_z_2d),
                          dim3(Dzaxis_BLOCK_X_2D, Dzaxis_BLOCK_Z_2D)>>>
        (s.device, p.device, num_row, num_col, num_dep);

    Dzaxis3dGpuKernel2 <<<dim3(num_blocks_x), dim3(Dzaxis_BLOCK_X)>>>
        (s.device, p.device, num_row, num_col, num_dep);
}

__global__ void Dzaxis3dGpuKernel1(
    cufftComplex *s, cufftComplex *p,
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;// x-axis
    int z = blockIdx.y * blockDim.y + threadIdx.y;// z-axis

    if( (z < (num_dep-1)) && (x < num_col) )
    {
        for(int y=0;y<num_row;y++) {
           // LHS(y,x,z) = RHS(y,x,z) - RHS(y,x,z+1)
           const unsigned int y_x_z = y*num_col*num_dep + x*num_dep + z;
           const unsigned int y_x_z1 = y*num_col*num_dep + x*num_dep + (z+1);
           s[y_x_z]REAL = p[y_x_z]REAL - p[y_x_z1]REAL;
           s[y_x_z]IMAG = p[y_x_z]IMAG - p[y_x_z1]IMAG;
        }
   }
}

__global__ void Dzaxis3dGpuKernel2(
    cufftComplex *s, cufftComplex *p,
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;// x axis

    if(x<num_col) 
    {
        for(int y=0;y<num_row;y++) {
           // LHS(y,x,num_dep-1) = RHS(y,x,num_dep-1) - RHS(y,x,0)
           const unsigned int y_x_num_dep_1 = y*num_col*num_dep + x*num_dep + (num_dep-1);
           const unsigned int y_x_zero = y*num_col*num_dep + x*num_dep;
           s[y_x_num_dep_1]REAL = p[y_x_num_dep_1]REAL - p[y_x_zero]REAL;
           s[y_x_num_dep_1]IMAG = p[y_x_num_dep_1]IMAG - p[y_x_zero]IMAG;
        }
    }
}
