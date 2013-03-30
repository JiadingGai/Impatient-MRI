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

    File Name   [DHWD3dGpu_toeplitz.cu]

    Synopsis    [GPU version of DHWD of 3D image for toeplitz recon.]

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
#include "DHWD3dGpu_toeplitz.cuh"

//FIXME:The four macros constrain each dimension
//size to be divisible by both 128 and 16.
#define DHWD_BLOCK_SIZE_X       128  // 1D only
#define DHWD_BLOCK_SIZE_Y       128  // 1D only
#define DHWD_BLOCK_SIZE_Z       128  // 1D only

#define DHWD_BLOCK_SIZE_X_2D    16
#define DHWD_BLOCK_SIZE_Y_2D    16
#define DHWD_BLOCK_SIZE_Z_2D    16


__global__ void DHWD3dGpuKernel1( 
    cufftComplex *t1, // 2D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col, 
    const unsigned int num_dep);

__global__ void DHWD3dGpuKernel2( 
    cufftComplex *t1, // 1D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep);

__global__ void DHWD3dGpuKernel3( 
    cufftComplex *s, // 1D kernel
    cufftComplex *t1,
    const unsigned int num_row,
    const unsigned int num_col,
    const unsigned int num_dep);

__global__ void DHWD3dGpuKernel4(
    cufftComplex *s, // 2D kernel
    cufftComplex *t1,
    const unsigned int num_row,
    const unsigned int num_col,
    const unsigned int num_dep);

__global__ void DHWD3dGpuKernel5(
    cufftComplex *t1, // 2D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep);

__global__ void DHWD3dGpuKernel6( 
    cufftComplex *t1, // 1D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep);

__global__ void DHWD3dGpuKernel7( 
    cufftComplex *s, // 1D kernel
    cufftComplex *t1,
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep);

__global__ void DHWD3dGpuKernel8(
    cufftComplex *s, // 2D kernel
    cufftComplex *t1,
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep);

__global__ void DHWD3dGpuKernel9(
    cufftComplex *t1, // 2D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep);

__global__ void DHWD3dGpuKernel10( 
    cufftComplex *t1, // 1D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep);

__global__ void DHWD3dGpuKernel11( 
    cufftComplex *s, // 1D kernel
    cufftComplex *t1,
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep);

__global__ void DHWD3dGpuKernel12(
    cufftComplex *s, // 2D kernel
    cufftComplex *t1,
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep);


__global__ void DHWD3dGpuKernel13( 
    cufftComplex *s, // 2D kernel
    const unsigned int num_row,
    const unsigned int num_col,
    const unsigned int num_dep,
    const float fd_penalizer);


// s = (D^H w D)p
// fd_penalizer: regularization weight
    void
DHWD3dGpu( matrix_t &s, matrix_t &p, float *w, const float fd_penalizer )
{
    const unsigned int num_row = p.dims[0];// y dimension
    const unsigned int num_col = p.dims[1];// x dimension
    const unsigned int num_dep = p.dims[2];// z dimension (depth)
    const unsigned int num = num_row * num_col * num_dep;

#if DEBUG
    printf("num=%d,%d,%d,%d\n", num_row, num_col, num_dep, num);
#endif

    cufftComplex *t1_device;
    CUDA_SAFE_CALL(cudaMalloc((void**)&t1_device, num * sizeof(cufftComplex)));

    int num_blocks_x = ceil((float) num_col / (float) DHWD_BLOCK_SIZE_X);
    int num_blocks_y = ceil((float) num_row / (float) DHWD_BLOCK_SIZE_Y);
    int num_blocks_z = ceil((float) num_dep / (float) DHWD_BLOCK_SIZE_Z);

    int num_blocks_x_2d = ceil((float) num_col /
                               (float) DHWD_BLOCK_SIZE_X_2D);
    int num_blocks_y_2d = ceil((float) num_row /
                               (float) DHWD_BLOCK_SIZE_Y_2D);
    int num_blocks_z_2d = ceil((float) num_dep /
                               (float) DHWD_BLOCK_SIZE_Z_2D);

    //Not all of the asserted items will be used, just to be safe.
    assert( num_blocks_x <= 65535 );
    assert( num_blocks_y <= 65535 );
    assert( num_blocks_z <= 65535 );
    assert( (num_blocks_x_2d * num_blocks_y_2d) <= 65535 );
    assert( (num_blocks_x_2d * num_blocks_z_2d) <= 65535 );
    assert( (num_blocks_y_2d * num_blocks_z_2d) <= 65535 );


//Part 1.1. Dy - Forward scheme 
    //common case
    DHWD3dGpuKernel1<<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Y_2D)>>>
        (t1_device, p.device, w, num_row, num_col, num_dep);    
    //boundary case
    DHWD3dGpuKernel2<<<dim3(num_blocks_x), dim3(DHWD_BLOCK_SIZE_X)>>>
        (t1_device, p.device, w, num_row, num_col, num_dep);

//Part 1.2. Dy - Backward scheme
    //boundary case
    DHWD3dGpuKernel3<<<dim3(num_blocks_x), dim3(DHWD_BLOCK_SIZE_X)>>>
        (s.device, t1_device, num_row, num_col, num_dep);
    //common case
    DHWD3dGpuKernel4<<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Y_2D)>>>
        (s.device, t1_device, num_row, num_col, num_dep);


//Part 2.1. Dx - Forward scheme
    //common case
    DHWD3dGpuKernel5<<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Y_2D)>>>
        (t1_device, p.device, w, num_row, num_col, num_dep);
    //boundary case
    DHWD3dGpuKernel6<<<dim3(num_blocks_y), dim3(DHWD_BLOCK_SIZE_Y)>>>
        (t1_device, p.device, w, num_row, num_col, num_dep);

//Part 2.2. Dx - Backward scheme    
    //boundary case
    DHWD3dGpuKernel7<<<dim3(num_blocks_y), dim3(DHWD_BLOCK_SIZE_Y)>>>
        (s.device, t1_device, num_row, num_col, num_dep);
    //common case
    DHWD3dGpuKernel8<<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Y_2D)>>>
        (s.device, t1_device, num_row, num_col, num_dep);

//Part 3.1. Dz - Forward scheme
    //common case
    DHWD3dGpuKernel9<<<dim3(num_blocks_x_2d, num_blocks_z_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Z_2D)>>>
        (t1_device, p.device, w, num_row, num_col, num_dep);
    //boundary case
    DHWD3dGpuKernel10<<<dim3(num_blocks_x), dim3(DHWD_BLOCK_SIZE_X)>>>
        (t1_device, p.device, w, num_row, num_col, num_dep);


//Part 3.2. Dz - Backward scheme    
    //boundary case
    DHWD3dGpuKernel11<<<dim3(num_blocks_x), dim3(DHWD_BLOCK_SIZE_X)>>>
        (s.device, t1_device, num_row, num_col, num_dep);
    //common case
    DHWD3dGpuKernel12<<<dim3(num_blocks_x_2d, num_blocks_z_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Z_2D)>>>
        (s.device, t1_device, num_row, num_col, num_dep);


//Part 4. summing up the result of DHWDF1 and DHWDF2
    DHWD3dGpuKernel13<<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Y_2D)>>>
        (s.device, num_row, num_col, num_dep, fd_penalizer);

    // free space 
    CUDA_SAFE_CALL(cudaFree(t1_device));
}


//Part 1.1. Dy - Forward scheme - common case
__global__ void DHWD3dGpuKernel1( 
    cufftComplex *t1, // (2D kernel): each thread loops thru the z-axis
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;// x axis
    int y = blockIdx.y * blockDim.y + threadIdx.y;// y axis

    // FIXME: Need to use registers for t1, and w.
    if ( (y < (num_row-1)) && (x < num_col) ) {
        for(int z=0;z<num_dep;z++) {
           // LHS(y,x,z) = RHS(y,x,z) - RHS(y+1,x,z)
           const unsigned int y_x_z = y*num_col*num_dep + x*num_dep + z;
           const unsigned int y1_x_z = y_x_z + num_col*num_dep;
           // for elements not concerned with the periodic condition
           t1[y_x_z]REAL  = p[y_x_z]REAL - p[y1_x_z]REAL;
           t1[y_x_z]IMAG  = p[y_x_z]IMAG - p[y1_x_z]IMAG;
           // times the weighted coefficients WDF1
           t1[y_x_z]REAL *= w[y_x_z];
           t1[y_x_z]IMAG *= w[y_x_z];
        }
    }
}

//Part 1.1. Dy - Forward scheme - boundary case
__global__ void DHWD3dGpuKernel2( 
    cufftComplex *t1, // (1D kernel): each thread loops thru the z-axis
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;// x axis

    if(x < num_col) {
       for(int z=0;z<num_dep;z++) {
          // FIXME: Need to use registers for t1_real, t1_imag, and w.
          // LHS(num_row-1,x,z) = RHS(num_row-1,x,z) - RHS(0,x,z)
          const unsigned int num_row_1_x_z = (num_row-1)*num_col*num_dep + x*num_dep + z;
          const unsigned int zero_x_z = x*num_dep + z;
          // for elements not concerned with the periodic condition
          t1[num_row_1_x_z]REAL = p[num_row_1_x_z]REAL - p[zero_x_z]REAL;
          t1[num_row_1_x_z]IMAG = p[num_row_1_x_z]IMAG - p[zero_x_z]IMAG;
          // times the weighted coefficients WDF1
          t1[num_row_1_x_z]REAL *= w[num_row_1_x_z];
          t1[num_row_1_x_z]IMAG *= w[num_row_1_x_z];
       }
    }
}

//Part 1.2. Dy - Backward scheme - boundary case
__global__ void DHWD3dGpuKernel3(
    cufftComplex *s, // (1D kernel): each thread loops thru the z-axis
    cufftComplex *t1,
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;// x axis

    if(x < num_col) {
       for(int z=0;z<num_dep;z++) {    
          //LHS(0,x,z) = RHS(0,x,z) - RHS(num_row-1,x,z)
          const unsigned int zero_x_z = x*num_dep + z;
          const unsigned int num_row_1_x_z = (num_row-1)*num_col*num_dep + x*num_dep + z;
      
          s[zero_x_z]REAL = t1[zero_x_z]REAL - t1[num_row_1_x_z]REAL;
          s[zero_x_z]IMAG = t1[zero_x_z]IMAG - t1[num_row_1_x_z]IMAG;
       }
    }
}


//Part 1.2. Dy - Backward scheme - common case
__global__ void DHWD3dGpuKernel4(
    cufftComplex *s, // (2D kernel): each thread loops thru the z-axis
    cufftComplex *t1,
    const unsigned int num_row,
    const unsigned int num_col,
    const unsigned int num_dep)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;//x axis
    int y = blockIdx.y * blockDim.y + threadIdx.y;//y axis

    if ((y>0) && (y<num_row) && (x<num_col)) {
       for(int z=0;z<num_dep;z++) {    
          // LHS(y,x,z) = RHS(y,x,z) - RHS(y-1,x,z)
          const unsigned int y_x_z = y*num_col*num_dep + x*num_dep + z;
          const unsigned int y_1_x_z = (y-1)*num_col*num_dep + x*num_dep + z;

          s[y_x_z]REAL = t1[y_x_z]REAL - t1[y_1_x_z]REAL;
          s[y_x_z]IMAG = t1[y_x_z]IMAG - t1[y_1_x_z]IMAG;
       }
    }
}



//Part 2.1. Dx - Forward scheme - common case
__global__ void DHWD3dGpuKernel5( 
    cufftComplex *t1, // 2D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;//x axis
    int y = blockIdx.y * blockDim.y + threadIdx.y;//y axis

    if ((x < (num_col-1)) && (y<num_row)) {
        for(int z=0;z<num_dep;z++) {
          // LHS(y,x,z) = RHS(y,x,z) - RHS(y,x+1,z)
          const unsigned int y_x_z = y*num_col*num_dep + x*num_dep + z;
          const unsigned int y_x1_z = y*num_col*num_dep + (x+1)*num_dep + z;

          t1[y_x_z]REAL = ( p[y_x_z]REAL - p[y_x1_z]REAL ) * w[y_x_z];
          t1[y_x_z]IMAG = ( p[y_x_z]IMAG - p[y_x1_z]IMAG ) * w[y_x_z];
        }
    }
}



//Part 2.1. Dx - Forward scheme - boundary case
__global__ void DHWD3dGpuKernel6( 
    cufftComplex *t1, // 1D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep )
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;//y axis
    if(y<num_row) {
       for(int z=0;z<num_dep;z++) {
          //LHS(y,num_col-1,z) = RHS(y,num_col-1,z) - RHS(y,0,z)
          const unsigned int y_num_col_1_z = y * num_col * num_dep + (num_col-1)*num_dep + z;
          const unsigned int y_zero_z = y*num_col*num_dep + z;
          t1[y_num_col_1_z]REAL = ( p[y_num_col_1_z]REAL - p[y_zero_z]REAL ) * w[y_num_col_1_z];
          t1[y_num_col_1_z]IMAG = ( p[y_num_col_1_z]IMAG - p[y_zero_z]IMAG ) * w[y_num_col_1_z];
       }
    }
}



//Part 2.2. Dx - Backward scheme - boundary case
__global__ void DHWD3dGpuKernel7( 
    cufftComplex *s, // 1D kernel
    cufftComplex *t1,
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep)
{
    // FIXME: Need to use registers for s_real and s_imag.
    int y = blockIdx.x * blockDim.x + threadIdx.x;//y axis
    
    if(y<num_row) {
       for(int z=0;z<num_dep;z++) {
          //LHS(y,0,z) = RHS(y,0,z) - RHS(y,num_col-1,z)
          const unsigned int y_zero_z = y*num_col*num_dep + z;
          const unsigned int y_num_col_1_z = y*num_col*num_dep + (num_col-1)*num_dep + z;

          // first num_row special rows
          s[y_zero_z]REAL += t1[y_zero_z]REAL - t1[y_num_col_1_z]REAL;
          s[y_zero_z]IMAG += t1[y_zero_z]IMAG - t1[y_num_col_1_z]IMAG;
       }
    }
}


//Part 2.2. Dx - Backward scheme - common case
__global__ void DHWD3dGpuKernel8(
    cufftComplex *s, // 2D kernel
    cufftComplex *t1,
    const unsigned int num_row, 
    const unsigned int num_col,
    const unsigned int num_dep)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;//x axis
    int y = blockIdx.y * blockDim.y + threadIdx.y;//y axis

    // FIXME: Need to use registers for s_real and s_imag.
    if ( (x > 0) && (x<num_col) && (y<num_row) ) {
        for(int z=0;z<num_dep;z++) {
          //LHS(y,x,z) = RHS(y,x,z) - RHS(y,x-1,z)
          const unsigned int y_x_z = y*num_col*num_dep + x*num_dep + z;
          const unsigned int y_x_1_z = y*num_col*num_dep + (x-1)*num_dep + z;

          s[y_x_z]REAL += t1[y_x_z]REAL - t1[y_x_1_z]REAL;
          s[y_x_z]IMAG += t1[y_x_z]IMAG - t1[y_x_1_z]IMAG;
        }
    }
}

//Part 3.1. Dz - Forward scheme - common case
__global__ void DHWD3dGpuKernel9( 
    cufftComplex *t1, // 2D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;// x-axis
    int z = blockIdx.y * blockDim.y + threadIdx.y;// z-axis

   
    if ((z < (num_dep-1)) && (x<num_col)) {
        for(int y=0;y<num_row;y++) {
          // LHS(y,x,z) = RHS(y,x,z) - RHS(y,x,z+1)
          const unsigned int y_x_z  = y*num_col*num_dep + x*num_dep + z;
          const unsigned int y_x_z1 = y*num_col*num_dep + x*num_dep + z + 1;

          t1[y_x_z]REAL = ( p[y_x_z]REAL - p[y_x_z1]REAL ) * w[y_x_z];
          t1[y_x_z]IMAG = ( p[y_x_z]IMAG - p[y_x_z1]IMAG ) * w[y_x_z];
        }
    }
}


//Part 3.1. Dz - Forward scheme - boundary case
__global__ void DHWD3dGpuKernel10( 
    cufftComplex *t1, // 1D kernel
    cufftComplex *p,
    float *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;//x-axis

    if(x<num_col) {
       for(int y=0;y<num_row;y++) {
          //LHS(y,x,num_dep-1) = RHS(y,x,num_dep-1) - RHS(y,x,0)
          const unsigned int y_x_num_dep_1 = y*num_col*num_dep + x*num_dep + (num_dep-1);
          const unsigned int y_x_zero = y*num_col*num_dep + x*num_dep;
          t1[y_x_num_dep_1]REAL = ( p[y_x_num_dep_1]REAL - p[y_x_zero]REAL ) * w[y_x_num_dep_1];
          t1[y_x_num_dep_1]IMAG = ( p[y_x_num_dep_1]IMAG - p[y_x_zero]IMAG ) * w[y_x_num_dep_1];
       }
    }
}


//Part 3.2. Dz - Backward scheme - boundary case
__global__ void DHWD3dGpuKernel11(
    cufftComplex *s, // 1D kernel
    cufftComplex *t1,
    const unsigned int num_row, const unsigned int num_col,
    const unsigned int num_dep)
{
    // FIXME: Need to use registers for s_real and s_imag.
    int x = blockIdx.x * blockDim.x + threadIdx.x;//x-axis

    if(x<num_col) {
       for(int y=0;y<num_row;y++) {
          //LHS(y,x,0) = RHS(y,x,0) - RHS(y,x,num_dep-1)
          const unsigned int y_x_zero = y*num_col*num_dep + x*num_dep;
          const unsigned int y_x_num_dep_1 = y*num_col*num_dep + x*num_dep + num_dep-1;

          // first num_row special rows
          s[y_x_zero]REAL += t1[y_x_zero]REAL - t1[y_x_num_dep_1]REAL;
          s[y_x_zero]IMAG += t1[y_x_zero]IMAG - t1[y_x_num_dep_1]IMAG;
       }
    }
}


//Part 3.2. Dz - Backward scheme - common case
__global__ void DHWD3dGpuKernel12(
    cufftComplex *s, // 2D kernel
    cufftComplex *t1,
    const unsigned int num_row, 
    const unsigned int num_col,
    const unsigned int num_dep)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;//x axis
    int z = blockIdx.y * blockDim.y + threadIdx.y;//z axis

    // FIXME: Need to use registers for s_real and s_imag.
    if ( (z > 0) && (x<num_col) && (z<num_dep) ) {
        for(int y=0;y<num_row;y++) {
          //LHS(y,x,z) = RHS(y,x,z) - RHS(y,x,z-1)
          const unsigned int y_x_z = y*num_col*num_dep + x*num_dep + z;
          const unsigned int y_x_z_1 = y*num_col*num_dep + x*num_dep + (z-1);

          s[y_x_z]REAL += t1[y_x_z]REAL - t1[y_x_z_1]REAL;
          s[y_x_z]IMAG += t1[y_x_z]IMAG - t1[y_x_z_1]IMAG;
        }
    }
}


//Part 4. summing up the result of DHWDF1 and DHWDF2
__global__ void DHWD3dGpuKernel13(
    cufftComplex *s, // 2D kernel on x-y: each thread loops thru the z-axis
    const unsigned int num_row,
    const unsigned int num_col,
    const unsigned int num_dep,
    const float fd_penalizer)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;//x-axis
    int y = blockIdx.y * blockDim.y + threadIdx.y;//y-axis

    if( (x<num_col) && (y<num_row) ) {
       for(int z=0;z<num_dep;z++) {
          // FIXME: Need to use registers for s_real and s_imag.
          const unsigned int y_x_z = y*num_col*num_dep + x*num_dep + z;
          s[y_x_z]REAL *= fd_penalizer;
          s[y_x_z]IMAG *= fd_penalizer;
       }
    }
}

