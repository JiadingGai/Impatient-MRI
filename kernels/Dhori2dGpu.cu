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

// Project header files
#include <tools.h>
#include <structures.h>

#include <Dhori2dGpu.cuh>

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

#define Dhori_BLOCK_Y      256
#define Dhori_BLOCK_X_2D   16
#define Dhori_BLOCK_Y_2D   16

__global__ void Dhori2dGpuKernel1(FLOAT_T *s_real, FLOAT_T *s_imag,
                     const FLOAT_T *p_real, const FLOAT_T *p_imag,
                     const unsigned int num_row, const unsigned int num_col);

__global__ void Dhori2dGpuKernel2(FLOAT_T *s_real, FLOAT_T *s_imag,
                     const FLOAT_T *p_real, const FLOAT_T *p_imag,
                     const unsigned int num_row, const unsigned int num_col);


    void
Dhori2dGpu(
    FLOAT_T *s_real, FLOAT_T *s_imag,
    const FLOAT_T *p_real, const FLOAT_T *p_imag,
    const unsigned int num_row, const unsigned int num_col)
{
    startMriTimer(getMriTimer()->timer_Dhori2dGpu);

    int num_blocks_y = ceil((FLOAT_T) num_row / (FLOAT_T) Dhori_BLOCK_Y);
    int num_blocks_x_2d = ceil((FLOAT_T) num_col / (FLOAT_T) Dhori_BLOCK_X_2D);
    int num_blocks_y_2d = ceil((FLOAT_T) num_row / (FLOAT_T) Dhori_BLOCK_Y_2D);

    makeSure(num_blocks_y <= 65535, "Maximum supported num_row is 64K-1.");
    makeSure(num_blocks_x_2d * num_blocks_y_2d <= 65535,
        "Maximum supported (num_col*num_row) is 64K-1.");

    Dhori2dGpuKernel1 <<<dim3(num_blocks_x_2d, num_blocks_y_2d),
                         dim3(Dhori_BLOCK_X_2D, Dhori_BLOCK_Y_2D)>>>
        (s_real, s_imag, p_real, p_imag, num_row, num_col);

    Dhori2dGpuKernel2 <<<dim3(num_blocks_y), dim3(Dhori_BLOCK_Y)>>>
        (s_real, s_imag, p_real, p_imag, num_row, num_col);

    stopMriTimer(getMriTimer()->timer_Dhori2dGpu);
}

__global__ void Dhori2dGpuKernel1(FLOAT_T *s_real, FLOAT_T *s_imag,
                     const FLOAT_T *p_real, const FLOAT_T *p_imag,
                     const unsigned int num_row, const unsigned int num_col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if( i < (num_col-1) )
    {
        #if 1
        const unsigned int i_num_row_j = i * num_row + j;
        const unsigned int i_num_row_j_num_row = i_num_row_j + num_row;
        s_real[i_num_row_j] = p_real[i_num_row_j] -
                                  p_real[i_num_row_j_num_row];
        s_imag[i_num_row_j] = p_imag[i_num_row_j] -
                                  p_imag[i_num_row_j_num_row];
        #else
        s_real[i * num_row + j] = p_real[i * num_row + j] -
                                  p_real[i * num_row + j + num_row];
        s_imag[i * num_row + j] = p_imag[i * num_row + j] -
                                  p_imag[i * num_row + j + num_row];
        #endif
    }
}
__global__ void Dhori2dGpuKernel2(FLOAT_T *s_real, FLOAT_T *s_imag,
                     const FLOAT_T *p_real, const FLOAT_T *p_imag,
                     const unsigned int num_row, const unsigned int num_col)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    #if 1
    const unsigned int num_col_1_num_row_j = (num_col-1)*num_row + j;
    s_real[num_col_1_num_row_j] = p_real[num_col_1_num_row_j] - p_real[j];
    s_imag[num_col_1_num_row_j] = p_imag[num_col_1_num_row_j] - p_imag[j];
    #else
    s_real[(num_col-1)*num_row + j] = p_real[(num_col-1)*num_row+j] - 
                                      p_real[j];
    s_imag[(num_col-1)*num_row + j] = p_imag[(num_col-1)*num_row+j] - 
                                      p_imag[j];
    #endif
}




/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

