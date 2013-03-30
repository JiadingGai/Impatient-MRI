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

    File Name   [DHWD2dGpu_toeplitz.cu]

    Synopsis    [GPU version of DHWD of 2D image for toeplitz recon.]

    Description []

    Revision    [0.1; Initial build; Jiading Gai, Beckman Institute,
                 Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Code cleaning and optimization; Xiao-Long Wu, ECE UIUC,
                 Jiading Gai, Beckman Institute]
    Date        [10/25/2010]

******************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <cutil.h>
#include <cufft.h>
#include "DHWD2dGpu_toeplitz.cuh"

//FIXME:The four macros constrain each dimension
//size to be divisible by both 128 and 16.
#define DHWD_BLOCK_SIZE_X       128  // 1D only
#define DHWD_BLOCK_SIZE_Y       128  // 1D only

#define DHWD_BLOCK_SIZE_X_2D    16
#define DHWD_BLOCK_SIZE_Y_2D    16


__global__ void DHWD2dGpuKernel1( 
    cufftComplex *t1, // 2D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col);

__global__ void DHWD2dGpuKernel2( 
    cufftComplex *t1, // 1D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col);

__global__ void DHWD2dGpuKernel3( 
    cufftComplex *s, // 1D kernel
    cufftComplex *t1,
    const unsigned int num_row,
    const unsigned int num_col);

__global__ void DHWD2dGpuKernel4(
    cufftComplex *s, // 2D kernel
    cufftComplex *t1,
    const unsigned int num_row,
    const unsigned int num_col);

__global__ void DHWD2dGpuKernel5(
    cufftComplex *t1, // 2D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col);

__global__ void DHWD2dGpuKernel6( 
    cufftComplex *t1, // 1D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col );

__global__ void DHWD2dGpuKernel7( 
    cufftComplex *s, // 1D kernel
    cufftComplex *t1,
    const unsigned int num_row, const unsigned int num_col );

__global__ void DHWD2dGpuKernel8(
    cufftComplex *s, // 2D kernel
    cufftComplex *t1,
    const unsigned int num_row, const unsigned int num_col);

__global__ void DHWD2dGpuKernel9( 
    cufftComplex *s, // 2D kernel
    const unsigned int num_row,
    const unsigned int num_col,
    const float fd_penalizer);


// s = (D^H w D)p
// fd_penalizer: regularization weight
    void
DHWD2dGpu( matrix_t &s, matrix_t &p, float *w, const float fd_penalizer )
{
    const unsigned int num_row = p.dims[0];
    const unsigned int num_col = p.dims[1];
    const unsigned int num = num_row*num_col;

#if DEBUG
    printf("num=%d,%d,%d\n", num_row, num_col, num);
#endif

    cufftComplex *t1_device;
    CUDA_SAFE_CALL(cudaMalloc((void**)&t1_device, num * sizeof(cufftComplex)));

    int num_blocks_x = ceil((float) num_col / (float) DHWD_BLOCK_SIZE_X);
    int num_blocks_y = ceil((float) num_row / (float) DHWD_BLOCK_SIZE_Y);
    int num_blocks_x_2d = ceil((float) num_col /
                               (float) DHWD_BLOCK_SIZE_X_2D);
    int num_blocks_y_2d = ceil((float) num_row /
                               (float) DHWD_BLOCK_SIZE_Y_2D);

    assert( num_blocks_x <= 65535 );
    assert( num_blocks_y <= 65535 );
    assert( (num_blocks_x_2d * num_blocks_y_2d) <= 65535 );

//Part 1.1. Dy - Forward scheme
    //common case
    // column wise finite difference DF1
    DHWD2dGpuKernel1<<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Y_2D)>>>
        (t1_device, p.device, w, num_row, num_col);    

    //boundary case
    // calculate the transpose of column wise finite difference operator DHWDF1
    DHWD2dGpuKernel2<<<dim3(num_blocks_x), dim3(DHWD_BLOCK_SIZE_X)>>>
        (t1_device, p.device, w, num_row, num_col);

//Part 1.2. Dy - Backward scheme
    //boundary case
    DHWD2dGpuKernel3<<<dim3(num_blocks_y), dim3(DHWD_BLOCK_SIZE_Y)>>>
        (s.device, t1_device, num_row, num_col);

    // common case
    // row wise finite difference DF2
    DHWD2dGpuKernel4<<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Y_2D)>>>
        (s.device, t1_device, num_row, num_col);

//Part 2.1. Dx - Forward scheme
    // common case
    DHWD2dGpuKernel5<<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Y_2D)>>>
        (t1_device, p.device, w, num_row, num_col);

    // boundary case
    // times the weighted coefficients WDF2
    DHWD2dGpuKernel6<<<dim3(num_blocks_y), dim3(DHWD_BLOCK_SIZE_Y)>>>
        (t1_device, p.device, w, num_row, num_col);
    
//Part 2.2. Dx - Backward scheme
    // boundary case
    // calculate the transpose of the column wise finite difference operator DHWDF2
    DHWD2dGpuKernel7<<<dim3(num_blocks_y), dim3(DHWD_BLOCK_SIZE_Y)>>>
        (s.device, t1_device, num_row, num_col);

    // common case
    DHWD2dGpuKernel8<<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Y_2D)>>>
        (s.device, t1_device, num_row, num_col);

//Part 3. summing up the result of DHWDF1 and DHWDF2
    DHWD2dGpuKernel9<<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Y_2D)>>>
        (s.device, num_row, num_col, fd_penalizer);

    // free space 
    CUDA_SAFE_CALL(cudaFree(t1_device));
}

//Dy - forward scheme - common case
__global__ void DHWD2dGpuKernel1( 
    cufftComplex *t1, // 2D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;// x axis
    int y = blockIdx.y * blockDim.y + threadIdx.y;// y axis

    // FIXME: Need to use registers for t1, and w.
    if ((y < (num_row-1)) && (x<num_col)) {
        // LHS(y,x) = RHS(y,x) - RHS(y+1,x)
        const unsigned int y_zero = y * num_col;
        const unsigned int y_x = y_zero + x;
        const unsigned int y1_x = y_x + num_col;
        // for elements not concerned with the periodic condition
        t1[y_x]REAL  = p[y_x]REAL - p[y1_x]REAL;
        t1[y_x]IMAG  = p[y_x]IMAG - p[y1_x]IMAG;
        // times the weighted coefficients WDF1
        t1[y_x]REAL *= w[y_x];
        t1[y_x]IMAG *= w[y_x];
    }
}

//Dy - forward scheme - boundary case
__global__ void DHWD2dGpuKernel2( 
    cufftComplex *t1, // 1D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;// x axis

    if(x < num_col) {
       // LHS(num_row-1,x) = RHS(num_row-1,x) - RHS(0,x)
       // FIXME: Need to use registers for t1_real, t1_imag, and w.
       const unsigned int num_row_1_x = (num_row-1)*num_col + x;
       // for elements not concerned with the periodic condition
       t1[num_row_1_x]REAL = p[num_row_1_x]REAL - p[x]REAL;
       t1[num_row_1_x]IMAG = p[num_row_1_x]IMAG - p[x]IMAG;
       // times the weighted coefficients WDF1
       t1[num_row_1_x]REAL *= w[num_row_1_x];
       t1[num_row_1_x]IMAG *= w[num_row_1_x];
    }
}

//Dy - backward scheme - boundary case
__global__ void DHWD2dGpuKernel3(
    cufftComplex *s, // 1D kernel
    cufftComplex *t1,
    const unsigned int num_row,
    const unsigned int num_col)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;// x axis

    if(x < num_col) {
       // LHS(0,x) = RHS(0,x) - RHS(num_row-1,x)
       const unsigned int num_row_1_x = (num_row-1)*num_col + x;

       s[x]REAL = t1[x]REAL - t1[num_row_1_x]REAL;
       s[x]IMAG = t1[x]IMAG - t1[num_row_1_x]IMAG;
    }
}

//Dy - backward scheme - common case
__global__ void DHWD2dGpuKernel4(
    cufftComplex *s, // 2D kernel
    cufftComplex *t1,
    const unsigned int num_row,
    const unsigned int num_col)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;// x axis
    int y = blockIdx.y * blockDim.y + threadIdx.y;// y axis

    if ( (y > 0) && (y < num_row) && (x < num_col) ) {
        // LHS(y,x) = RHS(y,x) - RHS(y-1,x)
        const unsigned int y_x = y * num_col + x;
        const unsigned int y_1_x = y_x - num_col;

        s[y_x]REAL = t1[y_x]REAL - t1[y_1_x]REAL;
        s[y_x]IMAG = t1[y_x]IMAG - t1[y_1_x]IMAG;
    }
}

//Dx - forward scheme - common case
__global__ void DHWD2dGpuKernel5(
    cufftComplex *t1, // 2D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;// x axis
    int y = blockIdx.y * blockDim.y + threadIdx.y;// y axis

    if ( (x < (num_col-1)) && (y<num_row) ) {
        // LHS(y,x) = RHS(y,x) - RHS(y,x+1)
        const unsigned int y_zero = y*num_col;
        const unsigned int y_x = y_zero + x;
        const unsigned int y_x1 = y_x + 1;

        t1[y_x]REAL = ( p[y_x]REAL - p[y_x1]REAL ) * w[y_x];
        t1[y_x]IMAG = ( p[y_x]IMAG - p[y_x1]IMAG ) * w[y_x];
    }
}

//Dx - forward scheme - boundary case
__global__ void DHWD2dGpuKernel6( 
    cufftComplex *t1, // 1D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col )
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;//y axis

    if(y<num_row) {
       //LHS(y,num_col-1) = RHS(y,num_col-1) - RHS(y,0)
       const unsigned int y_zero = y*num_col;
       const unsigned int y_num_col_1 = y_zero + (num_col-1);
       t1[y_num_col_1]REAL = ( p[y_num_col_1]REAL - p[y_zero]REAL ) * w[y_num_col_1];
       t1[y_num_col_1]IMAG = ( p[y_num_col_1]IMAG - p[y_zero]IMAG ) * w[y_num_col_1];
    }
}

//Dx - backward scheme - boundary case
__global__ void DHWD2dGpuKernel7( 
    cufftComplex *s, // 1D kernel
    cufftComplex *t1,
    const unsigned int num_row, const unsigned int num_col )
{
    // FIXME: Need to use registers for s_real and s_imag.
    int y = blockIdx.x * blockDim.x + threadIdx.x;//y axis

    if(y<num_row) {
       //LHS(y,0) = RHS(y,0) - RHS(y,num_col-1)
       const unsigned int y_zero = y * num_col;
       const unsigned int y_num_col_1 = y_zero + (num_col-1);
       // first num_row special rows
       s[y_zero]REAL += t1[y_zero]REAL - t1[y_num_col_1]REAL;
       s[y_zero]IMAG += t1[y_zero]IMAG - t1[y_num_col_1]IMAG;
    }
}

//Dx - backward scheme - common case
__global__ void DHWD2dGpuKernel8(
    cufftComplex *s, // 2D kernel
    cufftComplex *t1,
    const unsigned int num_row, const unsigned int num_col)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;//x axis
    int y = blockIdx.y * blockDim.y + threadIdx.y;//y axis

    // FIXME: Need to use registers for s_real and s_imag.
    if ( (x > 0) && (x<num_col) && (y<num_row) ) {
        //LHS(y,x) = RHS(y,x) - RHS(y,x-1)
        const unsigned int y_x = y*num_col + x;
        const unsigned int y_x_1 = y_x - 1;
        s[y_x]REAL += t1[y_x]REAL - t1[y_x_1]REAL;
        s[y_x]IMAG += t1[y_x]IMAG - t1[y_x_1]IMAG;
    }
}

__global__ void DHWD2dGpuKernel9( // = (DHWD2dCpu.cpp: Line 256 - 259)
    cufftComplex *s, // 2D kernel
    const unsigned int num_row,
    const unsigned int num_col,
    const float fd_penalizer)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;// x axis
    int y = blockIdx.y * blockDim.y + threadIdx.y;// y axis

    // FIXME: Need to use registers for s_real and s_imag.
    if ( (x<num_col) && (y<num_row) ) {
       const unsigned int y_x = y*num_col + x;
       s[y_x]REAL *= fd_penalizer;
       s[y_x]IMAG *= fd_penalizer;
    }
}

