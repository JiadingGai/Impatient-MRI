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

    File Name   [DHWD2dCpu.cpp]

    Synopsis    [CPU version of DHWD of 2D image.]

    Description []

    Revision    [0.1; Initial build; Fan Lam, Mao-Jing Fu, ECE UIUC]
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
#include <tools.h>
#include <structures.h>

#include <DHWD2dCpu.h>

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
    #define DEBUG_DHWD2DCPU  true
#else
    #define DEBUG_DHWD2DCPU  false
#endif

#define USE_OPTIMIZED_CODE  true

#if 1
    void
DHWD2dCpu(
    FLOAT_T *s_real, // DHWDF1 + DHWDF2 real
    FLOAT_T *s_imag, // DHWDF1 + DHWDF2 image
    const FLOAT_T *p_real, const FLOAT_T *p_imag,
    const FLOAT_T *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col,
    const FLOAT_T fd_penalizer)
{
    #if DEBUG_DHWD2DCPU
    msg(3, "DHWD2dCpu(): begin\n");
    #endif
    startMriTimer(getMriTimer()->timer_DHWD2dCpu);

    const unsigned int num = num_row*num_col;

    FLOAT_T *t1_real = mriNewCpu<FLOAT_T>(num); // WDF1/WDF2 real
    FLOAT_T *t1_imag = mriNewCpu<FLOAT_T>(num); // WDF1/WDF2 image

    unsigned int i = 0, j = 0;

    #if 1 // optimized
    // DHWD2dGpuKernel1
    // column wise finite difference DF1
    for (i = 0; i < num_col; i++) {
        const unsigned int i_num_row = i*num_row;
        for (j = 0; j < num_row - 1; j++) {
            const unsigned int i_num_row_j = i_num_row + j;
            const unsigned int i_num_row_j_1 = i_num_row + j + 1;

            // for elements not concerned with the periodic condition
            t1_real[i_num_row_j] = p_real[i_num_row_j] - p_real[i_num_row_j_1];
            t1_imag[i_num_row_j] = p_imag[i_num_row_j] - p_imag[i_num_row_j_1];
            // times the weighted coefficients WDF1
            t1_real[i_num_row_j] *= w[i_num_row_j];
            t1_imag[i_num_row_j] *= w[i_num_row_j];

            #if COMPUTE_FLOPS
            getMriFlop()->flop_DHWD2dCpu += 4;
            #endif
        }
    }

    // DHWD2dGpuKernel2
    for (i = 0; i < num_col; i++) {
        const unsigned int i_num_row = i*num_row;
        const unsigned int idx = i_num_row + num_row - 1;

        // for elements not concerned with the periodic condition
        t1_real[idx] = p_real[idx] - p_real[i_num_row];
        t1_imag[idx] = p_imag[idx] - p_imag[i_num_row];
        // times the weighted coefficients WDF1
        t1_real[idx] *= w[idx];
        t1_imag[idx] *= w[idx];

        #if COMPUTE_FLOPS
        getMriFlop()->flop_DHWD2dCpu += 4;
        #endif
    }

    #else // original
    // column wise finite difference DF1
    for (i = 0; i < num_col; i++) {
        for (j = 0; j < num_row - 1; j++) {
            // for elements not concerned with the periodic condition
            t1_real[i*num_row + j] = p_real[i*num_row + j] -
                                     p_real[i*num_row + j + 1];
            t1_imag[i*num_row + j] = p_imag[i*num_row + j] -
                                     p_imag[i*num_row + j + 1];

            // FIXME: Can be move out of this loop.
            // for elements concerned with the periodic condition
            t1_real[i*num_row+num_row-1] = p_real[i*num_row + num_row - 1] -
                                           p_real[i*num_row];
            t1_imag[i*num_row+num_row-1] = p_imag[i*num_row + num_row - 1] -
                                           p_imag[i*num_row];
        }
    }

    // times the weighted coefficients WDF1
    for (i = 0; i < num; i++) {
        t1_real[i] *= w[i];
        t1_imag[i] *= w[i];
        // w[i] is renewed with each cg operation
    }
    #endif

    // DHWD2dGpuKernel3
    // calculate the transpose of column wise finite difference operator DHWDF1
    for (i = 0; i < num_col; i++) {
        const unsigned int i_num_row = i*num_row;
        const unsigned int idx = i*num_row + num_row - 1;
        s_real[i_num_row] = t1_real[i_num_row] - t1_real[idx];
        s_imag[i_num_row] = t1_imag[i_num_row] - t1_imag[idx];

        #if COMPUTE_FLOPS
        getMriFlop()->flop_DHWD2dCpu += 2;
        #endif
    }

    // DHWD2dGpuKernel4
    for (i = 0; i < num_col; i++) {
        for (j = 1; j < num_row; j++) {
            const unsigned int i_num_row_j = i*num_row + j;
            const unsigned int idx = i*num_row + j - 1;
            s_real[i_num_row_j] = t1_real[i_num_row_j] - t1_real[idx];
            s_imag[i_num_row_j] = t1_imag[i_num_row_j] - t1_imag[idx];

            #if COMPUTE_FLOPS
            getMriFlop()->flop_DHWD2dCpu += 2;
            #endif
        }
    }

    #if 1 // optimized
    // DHWD2dGpuKernel5
    // row wise finite difference DF2
    for (i = 0; i < num_col - 1; i++) {
        const unsigned int i_num_row = i*num_row;
        for (j = 0; j < num_row; j++) {
            const unsigned int i_num_row_j = i_num_row + j;
            const unsigned int i_num_row_j_num_row = i_num_row + j + num_row;
            t1_real[i_num_row_j] = (p_real[i_num_row_j] -
                                    p_real[i_num_row_j_num_row]) *
                                   w[i_num_row_j];
            t1_imag[i_num_row_j] = (p_imag[i_num_row_j] -
                                    p_imag[i_num_row_j_num_row]) *
                                   w[i_num_row_j];

            #if COMPUTE_FLOPS
            getMriFlop()->flop_DHWD2dCpu += 2;
            #endif
        }
    }

    // DHWD2dGpuKernel6
    // this time the periodic condition is controlled by the i loop
    // different from the above column wise finite difference code
    for (j = 0; j < num_row; j++) {
        const unsigned int idx = (num_col-1)*num_row+j;
        t1_real[idx] = (p_real[idx] - p_real[j]) * w[idx];
        t1_imag[idx] = (p_imag[idx] - p_imag[j]) * w[idx];

        #if COMPUTE_FLOPS
        getMriFlop()->flop_DHWD2dCpu += 2;
        #endif
    }

    #else // original
    // row wise finite difference DF2
    for (i = 0; i < num_col - 1; i++) {
        for (j = 0; j < num_row; j++) {
            t1_real[i*num_row + j] = p_real[i*num_row + j] -
                                     p_real[i*num_row + j + num_row];
            t1_imag[i*num_row + j] = p_imag[i*num_row + j] -
                                     p_imag[i*num_row + j + num_row];
        }
    }

    // this time the periodic condition is controlled by the i loop
    // different from the above column wise finite difference code
    for (j = 0; j < num_row; j++) {
        t1_real[(num_col-1)*num_row + j] = p_real[(num_col-1)*num_row+j] -
                                           p_real[j];
        t1_imag[(num_col-1)*num_row + j] = p_imag[(num_col-1)*num_row+j] -
                                           p_imag[j];
    }

    // times the weighted coefficients WDF2
    for (i = 0; i < num; i++) {
        t1_real[i] *= w[i];
        t1_imag[i] *= w[i];
        // w[i] is renewed with each cg operation
    }
    #endif

#if 1 //Jiading GAI
	// DHWD2dGpuKernel7
    // calculate the transpose of the column wise finite difference operator DHWDF2
    for (j = 0; j < num_row; j++) {
        // first num_row special rows
        const unsigned int idx = j + num_row*(num_col - 1);
        s_real[j] += t1_real[j] - t1_real[idx];
        s_imag[j] += t1_imag[j] - t1_imag[idx];

        #if COMPUTE_FLOPS
        getMriFlop()->flop_DHWD2dCpu += 4;
        #endif
    }
#endif

	// DHWD2dGpuKernel8
    for (i = 1; i < num_col; i++) {
        for (j = 0; j < num_row; j++) {
            const unsigned int i_num_row_j = i*num_row+j;
            const unsigned int idx = (i - 1)*num_row + j;
            s_real[i_num_row_j] += t1_real[i_num_row_j] - t1_real[idx];
            s_imag[i_num_row_j] += t1_imag[i_num_row_j] - t1_imag[idx];

            #if COMPUTE_FLOPS
            getMriFlop()->flop_DHWD2dCpu += 4;
            #endif
        }
    }



    // DHWD2dGpuKernel9
    // summing up the result of DHWDF1 and DHWDF2
    for (i = 0; i < num; i++) {
        s_real[i] *= fd_penalizer;
        s_imag[i] *= fd_penalizer;

        #if COMPUTE_FLOPS
        getMriFlop()->flop_DHWD2dCpu += 2;
        #endif
    }

    // free space
    mriDeleteCpu(t1_real);
    mriDeleteCpu(t1_imag);

    stopMriTimer(getMriTimer()->timer_DHWD2dCpu);
    #if DEBUG_DHWD2DCPU
    msg(3, "DHWD2dCpu(): end\n");
    #endif
}

#else // original ============================================================
    void
DHWD2dCpu(
    FLOAT_T *s_real, // DHWDF1 + DHWDF2 real
    FLOAT_T *s_imag, // DHWDF1 + DHWDF2 image
    const FLOAT_T *p_real, const FLOAT_T *p_imag,
    const FLOAT_T *w, // a vector storing the diagonal elements of W
    const unsigned int num_row, const unsigned int num_col,
    const FLOAT_T fd_penalizer)
{
    #if DEBUG_DHWD2DCPU
    msg(3, "DHWD2dCpu(): begin\n");
    #endif
    startMriTimer(getMriTimer()->timer_DHWD2dCpu);
    
    const unsigned int num = num_row*num_col;

    FLOAT_T *t1_real = mriNewCpu<FLOAT_T>(num); // WDF1 real
    FLOAT_T *t1_imag = mriNewCpu<FLOAT_T>(num); // WDF1 image
    FLOAT_T *c2_real = mriNewCpu<FLOAT_T>(num); // DHWDF1 real
    FLOAT_T *c2_imag = mriNewCpu<FLOAT_T>(num); // DHWDF1 image
    FLOAT_T *r2_real = mriNewCpu<FLOAT_T>(num); // DHWDF2 real
    FLOAT_T *r2_imag = mriNewCpu<FLOAT_T>(num); // DHWDF2 image

    unsigned int i = 0, j = 0;

    // GPU: Kernel 1
    // column wise finite difference DF1
    for (i = 0; i < num_col; i++) {
        #if USE_OPTIMIZED_CODE
        const unsigned int i_num_row = i*num_row;
        #endif
        for (j = 0; j < num_row - 1; j++) {
            #if USE_OPTIMIZED_CODE // Reorder statements to increase cache efficiency
            // for elements not concerned with the periodic condition
            const unsigned int t1_idx = i_num_row + j;
            const unsigned int t1_idx2 = i_num_row + num_row - 1;
            t1_real[t1_idx]  = p_real[t1_idx]  - p_real[t1_idx+1];
            t1_real[t1_idx2] = p_real[t1_idx2] - p_real[i_num_row];
            t1_imag[t1_idx]  = p_imag[t1_idx]  - p_imag[t1_idx+1];
            t1_imag[t1_idx2] = p_imag[t1_idx2] - p_imag[i_num_row];
            t1_real[t1_idx]  *= w[t1_idx];
            t1_real[t1_idx2] *= w[t1_idx2];
            t1_imag[t1_idx]  *= w[t1_idx];
            t1_imag[t1_idx2] *= w[t1_idx2];
            #else // Original
            // for elements not concerned with the periodic condition
            t1_real[i*num_row + j] = p_real[i*num_row + j] -
                                     p_real[i*num_row + j + 1];
            t1_imag[i*num_row + j] = p_imag[i*num_row + j] -
                                     p_imag[i*num_row + j + 1];

            // FIXME: Can be move out of this loop.
            // for elements concerned with the periodic condition
            t1_real[i*num_row+num_row-1] = p_real[i*num_row + num_row - 1] -
                                           p_real[i*num_row];
            t1_imag[i*num_row+num_row-1] = p_imag[i*num_row + num_row - 1] -
                                           p_imag[i*num_row];
            #endif
        }
    }

    // times the weighted coefficients WDF1
    #if !USE_OPTIMIZED_CODE
    for (i = 0; i < num; i++) {
        t1_real[i] *= w[i];
        t1_imag[i] *= w[i];
        // w[i] is renewed with each cg operation
    }
    #endif

    // GPU: Kernel 2
    // calculate the transpose of column wise finite difference operator DHWDF1
    for (i = 0; i < num_col; i++) {
        c2_real[i*num_row] = t1_real[i*num_row] -
                             t1_real[i*num_row + num_row - 1];
        c2_imag[i*num_row] = t1_imag[i*num_row] -
                             t1_imag[i*num_row + num_row - 1];
    }

    // GPU: Kernel 3
    for (i = 0; i < num_col; i++) {
        for (j = 1; j < num_row; j++) {
            c2_real[i*num_row + j] = t1_real[i*num_row + j] -
                                     t1_real[i*num_row + j - 1];
            c2_imag[i*num_row + j] = t1_imag[i*num_row + j] -
                                     t1_imag[i*num_row + j - 1];
        }
    }

    // row wise finite difference DF2
    for (i = 0; i < num_col - 1; i++) {
        for (j = 0; j < num_row; j++) {
            t1_real[i*num_row + j] = p_real[i*num_row + j] -
                                     p_real[i*num_row + j + num_row];
            t1_imag[i*num_row + j] = p_imag[i*num_row + j] -
                                     p_imag[i*num_row + j + num_row];
        }
    }

    // this time the periodic condition is controlled by the i loop
    // different from the above column wise finite difference code
    for (j = 0; j < num_row; j++) {
        t1_real[(num_col-1)*num_row + j] = p_real[(num_col-1)*num_row+j] -
                                           p_real[j];
        t1_imag[(num_col-1)*num_row + j] = p_imag[(num_col-1)*num_row+j] -
                                           p_imag[j];
    }

    // times the weighted coefficients WDF2
    for (i = 0; i < num; i++) {
        t1_real[i] = t1_real[i]*w[i];
        t1_imag[i] = t1_imag[i]*w[i];
        // w[i] is renewed with each cg operation
    }

    // calculate the transpose of the column wise finite difference operator DHWDF2
    for (j = 0; j < num_row; j++) {
        // first num_row special rows
        r2_real[j] = t1_real[j] - t1_real[j + num_row*(num_col - 1)];
        r2_imag[j] = t1_imag[j] - t1_imag[j + num_row*(num_col - 1)];
    }
    for (i = 1; i < num_col; i++) {
        for (j = 0; j < num_row; j++) {
            r2_real[i*num_row + j] = t1_real[i*num_row + j] -
                                     t1_real[(i - 1)*num_row + j];
            r2_imag[i*num_row + j] = t1_imag[i*num_row + j] -
                                     t1_imag[(i - 1)*num_row + j];
        }
    }

    // summing up the result of DHWDF1 and DHWDF2
    for (i = 0; i < num; i++) {
        s_real[i] = (c2_real[i] + r2_real[i]) * fd_penalizer;
        s_imag[i] = (c2_imag[i] + r2_imag[i]) * fd_penalizer;
    }

    // free space
    mriDeleteCpu(t1_real);
    mriDeleteCpu(t1_imag);
    mriDeleteCpu(c2_real);
    mriDeleteCpu(c2_imag);
    mriDeleteCpu(r2_real);
    mriDeleteCpu(r2_imag);

    stopMriTimer(getMriTimer()->timer_DHWD2dCpu);
    #if DEBUG_DHWD2DCPU
    msg(3, "DHWD2dCpu(): end\n");
    #endif
}
#endif

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

