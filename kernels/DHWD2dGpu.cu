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

    File Name   [DHWD2dGpu.cu]

    Synopsis    [GPU version of DHWD of 2D image.]

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

// Project header files
#include <xcpplib_process.h>
#include <xcpplib_typesGpu.cuh>
#include <tools.h>
#include <structures.h>
#include <cutil_inline.h>
#include <DHWD2dGpu.cuh>

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
    #define DEBUG_DHWD2dGpu  true
#else
    #define DEBUG_DHWD2dGpu  false
#endif

#define USE_OPTIMIZED_CODE  true


#define DHWD_BLOCK_SIZE_X       64  // 1D only
#define DHWD_BLOCK_SIZE_Y       64 // 1D only

#define DHWD_BLOCK_SIZE_X_2D    16
#define DHWD_BLOCK_SIZE_Y_2D    16

__global__ void DHWD2dGpuKernel1( // = (DHWD2dCpu.cpp: Line 117-129)
    FLOAT_T *t1_real, // 2D kernel
    FLOAT_T *t1_imag, 
    const FLOAT_T *p_real, const FLOAT_T *p_imag,
    const FLOAT_T *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col);

__global__ void DHWD2dGpuKernel2( // = (DHWD2dCpu.cpp: Line 131-140)
    FLOAT_T *t1_real, // 1D kernel
    FLOAT_T *t1_imag, 
    const FLOAT_T *p_real, const FLOAT_T *p_imag,
    const FLOAT_T *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col);

__global__ void DHWD2dGpuKernel3( // = (DHWD2dCpu.cpp: Line 172 - 177)
    FLOAT_T *s_real, // 1D kernel
    FLOAT_T *s_imag, 
    const FLOAT_T *t1_real, const FLOAT_T *t1_imag,
    const unsigned int num_row, const unsigned int num_col);

__global__ void DHWD2dGpuKernel4( // = (DHWD2dCpu.cpp: Line 179 - 180)
    FLOAT_T *s_real, // 2D kernel
    FLOAT_T *s_imag, 
    const FLOAT_T *t1_real, const FLOAT_T *t1_imag,
    const unsigned int num_row);

__global__ void DHWD2dGpuKernel5( // = (DHWD2dCpu.cpp: Line 190 - 200)
    FLOAT_T *t1_real, // 2D kernel
    FLOAT_T *t1_imag, 
    const FLOAT_T *p_real, const FLOAT_T *p_imag,
    const FLOAT_T *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col);

__global__ void DHWD2dGpuKernel6( // = (DHWD2dCpu.cpp: Line 204 -208)
    FLOAT_T *t1_real, // 1D kernel
    FLOAT_T *t1_imag, 
    const FLOAT_T *p_real, const FLOAT_T *p_imag,
    const FLOAT_T *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col );

__global__ void DHWD2dGpuKernel7( // = (DHWD2dCpu.cpp: Line 241 -245)
    FLOAT_T *s_real, // 1D kernel
    FLOAT_T *s_imag, 
    const FLOAT_T *t1_real, const FLOAT_T *t1_imag,
    const unsigned int num_row, const unsigned int num_col );

__global__ void DHWD2dGpuKernel8( // = (DHWD2dCpu.cpp: Line 246-253)
    FLOAT_T *s_real, // 2D kernel
    FLOAT_T *s_imag, 
    const FLOAT_T *t1_real, const FLOAT_T *t1_imag,
    const unsigned int num_row, const unsigned int num_col);

__global__ void DHWD2dGpuKernel9( // = (DHWD2dCpu.cpp: Line 256 - 259)
    FLOAT_T *s_real, FLOAT_T *s_imag, // 2D kernel
    const unsigned int num_row,
    const FLOAT_T fd_penalizer);

    void
DHWD2dGpu(
    FLOAT_T *s_real, // DHWDF1 + DHWDF2 real
    FLOAT_T *s_imag, // DHWDF1 + DHWDF2 image
    const FLOAT_T *p_real, const FLOAT_T *p_imag,
    const FLOAT_T *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col,
    const FLOAT_T fd_penalizer)
{
    #if DEBUG_DHWD2dGpu
    msg(2, "DHWD2dGpu(): begin\n");
    #endif
    startMriTimer(getMriTimer()->timer_DHWD2dGpu);

    const unsigned int num = num_row*num_col;

    FLOAT_T *t1_real = mriNewGpu<FLOAT_T>(num); // WDF1 real
    FLOAT_T *t1_imag = mriNewGpu<FLOAT_T>(num); // WDF1 image

    int num_blocks_x = ceil((FLOAT_T) num_col / (FLOAT_T) DHWD_BLOCK_SIZE_X);
    int num_blocks_y = ceil((FLOAT_T) num_row / (FLOAT_T) DHWD_BLOCK_SIZE_Y);
    int num_blocks_x_2d = ceil((FLOAT_T) num_col /
                               (FLOAT_T) DHWD_BLOCK_SIZE_X_2D);
    int num_blocks_y_2d = ceil((FLOAT_T) num_row /
                               (FLOAT_T) DHWD_BLOCK_SIZE_Y_2D);

    makeSure(num_blocks_x <= 65535, "Maximum supported num_col is 64K-1.");
    makeSure(num_blocks_y <= 65535, "Maximum supported num_row is 64K-1.");
    makeSure(num_blocks_x_2d * num_blocks_y_2d <= 65535,
        "Maximum supported (num_col*num_row) is 64K-1.");

    // column wise finite difference DF1
    DHWD2dGpuKernel1<<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Y_2D)>>>
        (t1_real, t1_imag, p_real, p_imag, w, num_row, num_col);    

    // calculate the transpose of column wise finite difference operator DHWDF1
    DHWD2dGpuKernel2<<<dim3(num_blocks_x), dim3(DHWD_BLOCK_SIZE_X)>>>
        (t1_real, t1_imag, p_real, p_imag, w, num_row, num_col);

    DHWD2dGpuKernel3<<<dim3(num_blocks_x), dim3(DHWD_BLOCK_SIZE_X)>>>
        (s_real, s_imag, t1_real, t1_imag, num_row, num_col);

    // row wise finite difference DF2
    DHWD2dGpuKernel4<<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Y_2D)>>>
        (s_real, s_imag, t1_real, t1_imag, num_row);

    // this time the periodic condition is controlled by the i loop
    // different from the above column wise finite difference code
    DHWD2dGpuKernel5<<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Y_2D)>>>
        (t1_real, t1_imag, p_real, p_imag, w, num_row, num_col);

    // times the weighted coefficients WDF2
    DHWD2dGpuKernel6<<<dim3(num_blocks_y), dim3(DHWD_BLOCK_SIZE_Y)>>>
        (t1_real, t1_imag, p_real, p_imag, w, num_row, num_col);

#if 1 //JiadingGAI
    // calculate the transpose of the column wise finite difference operator DHWDF2
    DHWD2dGpuKernel7<<<dim3(num_blocks_y), dim3(DHWD_BLOCK_SIZE_Y)>>>
        (s_real, s_imag, t1_real, t1_imag, num_row, num_col);

#endif
    DHWD2dGpuKernel8<<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Y_2D)>>>
        (s_real, s_imag, t1_real, t1_imag, num_row, num_col);

    // summing up the result of DHWDF1 and DHWDF2
    DHWD2dGpuKernel9<<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                       dim3(DHWD_BLOCK_SIZE_X_2D, DHWD_BLOCK_SIZE_Y_2D)>>>
        (s_real, s_imag, num_row, fd_penalizer);

    // free space
    mriDeleteGpu(t1_real);
    mriDeleteGpu(t1_imag);

    stopMriTimer(getMriTimer()->timer_DHWD2dGpu);
    #if DEBUG_DHWD2dGpu
    msg(2, "DHWD2dGpu(): end\n");
    #endif
}

__global__ void DHWD2dGpuKernel1(
    FLOAT_T *t1_real, // 2D kernel
    FLOAT_T *t1_imag, 
    const FLOAT_T *p_real, const FLOAT_T *p_imag,
    const FLOAT_T *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // FIXME: Need to use registers for t1_real, t1_imag, and w.
    if (j < (num_row-1)) {
        const unsigned int i_num_row = i*num_row;
        const unsigned int i_num_row_j   = i_num_row + j;
        const unsigned int i_num_row_j_1 = i_num_row + j + 1;
		// for elements not concerned with the periodic condition
        t1_real[i_num_row_j]  = p_real[i_num_row_j] - p_real[i_num_row_j_1];
        t1_imag[i_num_row_j]  = p_imag[i_num_row_j] - p_imag[i_num_row_j_1];
		// times the weighted coefficients WDF1
		t1_real[i_num_row_j] *= w[i_num_row_j];
		t1_imag[i_num_row_j] *= w[i_num_row_j];
    }
}

__global__ void DHWD2dGpuKernel2(
    FLOAT_T *t1_real, // 1D kernel
    FLOAT_T *t1_imag, 
    const FLOAT_T *p_real, const FLOAT_T *p_imag,
    const FLOAT_T *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<num_col) {
      // FIXME: Need to use registers for t1_real, t1_imag, and w.
      const unsigned int i_num_row = i*num_row;
      const unsigned int idx = i_num_row + num_row - 1;
	  // for elements not concerned with the periodic condition
      t1_real[idx] = p_real[idx] - p_real[i_num_row];
      t1_imag[idx] = p_imag[idx] - p_imag[i_num_row];
	  // times the weighted coefficients WDF1
	  t1_real[idx] *= w[idx];
	  t1_imag[idx] *= w[idx];
    }
}

__global__ void DHWD2dGpuKernel3(
    FLOAT_T *s_real, // 1D kernel
    FLOAT_T *s_imag, 
    const FLOAT_T *t1_real, const FLOAT_T *t1_imag,
    const unsigned int num_row, const unsigned int num_col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
  
    if(i<num_col) {
      const unsigned int i_num_row = i*num_row;
      const unsigned int idx = i*num_row + num_row -1;

      s_real[i_num_row] = t1_real[i_num_row] - t1_real[idx];
      s_imag[i_num_row] = t1_imag[i_num_row] - t1_imag[idx];
    }
}


__global__ void DHWD2dGpuKernel4(
    FLOAT_T *s_real, // 2D kernel
    FLOAT_T *s_imag, 
    const FLOAT_T *t1_real, const FLOAT_T *t1_imag,
    const unsigned int num_row)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j > 0) {
        const unsigned int i_num_row_j = i * num_row + j;
        const unsigned int i_num_row_j_1 = i_num_row_j - 1;

        s_real[i_num_row_j] = t1_real[i_num_row_j] - t1_real[i_num_row_j_1];
        s_imag[i_num_row_j] = t1_imag[i_num_row_j] - t1_imag[i_num_row_j_1];
    }
}

__global__ void DHWD2dGpuKernel5( // = (DHWD2dCpu.cpp: Line 190 - 200)
    FLOAT_T *t1_real, // 2D kernel
    FLOAT_T *t1_imag, 
    const FLOAT_T *p_real, const FLOAT_T *p_imag,
    const FLOAT_T *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < (num_col-1)) {
		const unsigned int i_num_row = i*num_row;
        const unsigned int i_num_row_j = i_num_row + j;
        const unsigned int i_num_row_j_num_row = i_num_row + j + num_row;

        t1_real[i_num_row_j] = (p_real[i_num_row_j] -
                                p_real[i_num_row_j_num_row]) *
                                w[i_num_row_j];
        t1_imag[i_num_row_j] = (p_imag[i_num_row_j] -
                                p_imag[i_num_row_j_num_row]) *
                                w[i_num_row_j];
    }
}

__global__ void DHWD2dGpuKernel6( // = (DHWD2dCpu.cpp: Line 204 -208)
    FLOAT_T *t1_real, // 1D kernel
    FLOAT_T *t1_imag, 
    const FLOAT_T *p_real, const FLOAT_T *p_imag,
    const FLOAT_T *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col )
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(j<num_row) {
      const unsigned int idx = (num_col-1)*num_row+j;
      t1_real[idx] = (p_real[idx] - p_real[j]) * w[idx];
      t1_imag[idx] = (p_imag[idx] - p_imag[j]) * w[idx];
    }
}

__global__ void DHWD2dGpuKernel7( // = (DHWD2dCpu.cpp: Line 241 -245)
    FLOAT_T *s_real, // 1D kernel
    FLOAT_T *s_imag, 
    const FLOAT_T *t1_real, const FLOAT_T *t1_imag,
    const unsigned int num_row, const unsigned int num_col )
{
    // FIXME: Need to use registers for s_real and s_imag.
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(j<num_row) {
      const unsigned int idx = j + num_row*(num_col-1);
	  // first num_row special rows
      s_real[j] += t1_real[j] - t1_real[idx];
	  s_imag[j] += t1_imag[j] - t1_imag[idx];
    }
}

__global__ void DHWD2dGpuKernel8( // = (DHWD2dCpu.cpp: Line 246-253)
    FLOAT_T *s_real, // 2D kernel
    FLOAT_T *s_imag, 
    const FLOAT_T *t1_real, const FLOAT_T *t1_imag,
    const unsigned int num_row, const unsigned int num_col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // FIXME: Need to use registers for s_real and s_imag.
    if (i > 0) {
        const unsigned int i_num_row_j = i*num_row + j;
        const unsigned int i_1_num_row_j = (i - 1)*num_row + j;
        s_real[i_num_row_j] += t1_real[i_num_row_j] - t1_real[i_1_num_row_j];
        s_imag[i_num_row_j] += t1_imag[i_num_row_j] - t1_imag[i_1_num_row_j];
    }
}

__global__ void DHWD2dGpuKernel9( // = (DHWD2dCpu.cpp: Line 256 - 259)
    FLOAT_T *s_real, FLOAT_T *s_imag, // 2D kernel
    const unsigned int num_row,
    const FLOAT_T fd_penalizer)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // FIXME: Need to use registers for s_real and s_imag.
    const unsigned int i_num_row_j = i*num_row + j;
    s_real[i_num_row_j] *= fd_penalizer;
    s_imag[i_num_row_j] *= fd_penalizer;
}


/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

