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

    File Name   [ftCpu.cpp]

    Synopsis    [The CPU version of Fourier transform and inverse Fourier
        transform.]

    Description []

    Revision    [0.1; Initial build; Yue Zhuo, BIOE UIUC]
    Revision    [0.1.1; Add OpenMP, Code cleaning; Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Further optimization, Code cleaning, Adding comments;
                 Xiao-Long Wu, ECE UIUC, Jiading Gai, Beckman Institute]
    Date        [10/27/2010]

*****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>
#include <math.h>

#include <tools.h>
#include <structures.h>
#include <ftCpu.h>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [CPU version of the sin function.]                           */
/*                                                                           */
/*  Description [This function is used to avoid additional computations when */
/*      the data values are too small.]                                      */
/*                                                                           */
/*===========================================================================*/

#if COMPUTE_FLOPS
FLOAT_T sinc_cpu(FLOAT_T x, double *flop)
#else
FLOAT_T sinc_cpu(FLOAT_T x)
#endif
{
    if (fabs(x) < 0.0001) {
        return 1.0;
    }

    #if COMPUTE_FLOPS
    *flop += 13 + 1 + 1 + 1;
    #endif
    return sin(MRI_PI * x) / (MRI_PI * x);
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [CPU kernel of the Fourier Transformation (FT).]             */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    void
ftCpu(FLOAT_T *kdata_r, FLOAT_T *kdata_i,
      const FLOAT_T *idata_r, const FLOAT_T *idata_i,
      const DataTraj *ktraj, const DataTraj *itraj,
      const FLOAT_T *fm, const FLOAT_T *t,
      const int num_k, const int num_i
      )
{
    startMriTimer(getMriTimer()->timer_ftCpu);

    FLOAT_T sumr = 0, sumi = 0, tpi = 0, kzdeltaz = 0, kziztpi = 0,
            expr = 0, cosexpr = 0, sinexpr = 0, t_tpi = 0,
            kx_N = 0, ky_N = 0, kxtpi = 0, kytpi = 0;
    int i = 0, j = 0;

    //--------------------------------------------------------------------
    //                         Initialization
    //--------------------------------------------------------------------
    tpi = 2 * MRI_PI;
    #if DATATRAJ_NO_Z_DIM
    kzdeltaz = 0 * MRI_DELTAZ;
    kziztpi  = 0 * 0 * tpi;
    #else // DataTraj is using z dimension
    kzdeltaz = ktraj[0].z * MRI_DELTAZ;
    kziztpi  = ktraj[0].z * itraj[0].z * tpi;
    #endif

    #if COMPUTE_FLOPS
    getMriFlop()->flop_ftCpu += 1 + 1 + 2;
    #endif

    //--------------------------------------------------------------------
    //                    Fourier Transform:       Gx=G * x
    //--------------------------------------------------------------------
    // NON-conjugate transpose of G
    #if 0 //USE_OPENMP // FIXME: We can choose either this or the inner loop.
    #pragma omp parallel for
    #endif
    for (i = 0; i < num_k; i++) { // i is the time point in k-space
        sumr = 0.0;
        sumi = 0.0;

        t_tpi = t[i] / tpi;
        kx_N = ktraj[i].x / MRI_NN; // MRI_NN for normalized kx
        ky_N = ktraj[i].y / MRI_NN;
        kxtpi = ktraj[i].x * tpi;
        kytpi = ktraj[i].y * tpi;

        #if COMPUTE_FLOPS
        getMriFlop()->flop_ftCpu += 1 + 1 + 1 + 1 + 1;
        #endif

        #if USE_OPENMP
        #pragma omp parallel for default(none) reduction(+:sumr, sumi) \
         private(expr, cosexpr, sinexpr) \
         shared(i, kx_N, ky_N, kzdeltaz, t_tpi, fm, kxtpi, kytpi, \
                kziztpi, itraj, t, idata_r, idata_i)
        #endif
        for (j = 0; j < num_i; j++) { // j is the pixel point in image-space
            expr = (kxtpi * itraj[j].x + kytpi * itraj[j].y + kziztpi) +
                       (fm[j] * t[i]);
            #if ENABLE_DOUBLE_PRECISION
                cosexpr = cos(expr); sinexpr = sin(expr);
            #else
                cosexpr = cosf(expr); sinexpr = sinf(expr);
            #endif
            sumr += (cosexpr * idata_r[j]) + (sinexpr * idata_i[j]);
            sumi += (-sinexpr * idata_r[j]) + (cosexpr * idata_i[j]);

            #if COMPUTE_FLOPS
            // sin: 13 flop; cos: 12 flop; +,-,*,/: 1
            getMriFlop()->flop_ftCpu += 2 + 2 + 2 + 2 + 6 + 12 + 13 + 5 + 6;
            #endif
        }
        kdata_r[i] = sumr; // Real part
        kdata_i[i] = sumi; // Imaginary part
    }

    stopMriTimer(getMriTimer()->timer_ftCpu);
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [CPU kernel of the Inverse Fourier Transformation (IFT).]    */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    void
iftCpu(FLOAT_T *idata_r, FLOAT_T *idata_i,
       const FLOAT_T *kdata_r, const FLOAT_T *kdata_i,
       const DataTraj *ktraj, const DataTraj *itraj,
       const FLOAT_T *fm, const FLOAT_T *t,
       const int num_k, const int num_i
       )
{
    startMriTimer(getMriTimer()->timer_iftCpu);

    FLOAT_T sumr = 0, sumi = 0, tpi = 0, kzdeltaz = 0, kziztpi = 0,
            expr = 0, cosexpr = 0, sinexpr = 0,
            itraj_x_tpi = 0, itraj_y_tpi = 0;
    int i = 0, j = 0;

    //--------------------------------------------------------------------
    //                         Initialization
    //--------------------------------------------------------------------
    tpi = MRI_PI * 2.0;
    #if DATATRAJ_NO_Z_DIM
    kzdeltaz = 0 * MRI_DELTAZ;
    kziztpi = 0 * 0 * tpi;
    #else // DataTraj is using z dimension
    kzdeltaz = ktraj[0].z * MRI_DELTAZ;
    kziztpi = ktraj[0].z * itraj[0].z * tpi;
    #endif

    #if COMPUTE_FLOPS
    getMriFlop()->flop_iftCpu += 1 + 1 + 2;
    #endif

    //--------------------------------------------------------------------
    //               Inverse Fourier Transform:     x=(G^H) * Gx
    //--------------------------------------------------------------------
    // the conjugate transpose of G
    #if 0 //USE_OPENMP // FIXME: We can choose either this or the inner loop.
    #pragma omp parallel for
    #endif
    for (j = 0; j < num_i; j++) { // j is the pixel points in image-space
        sumr = 0.0;
        sumi = 0.0;

        itraj_x_tpi = itraj[j].x * tpi;
        itraj_y_tpi = itraj[j].y * tpi;

        #if COMPUTE_FLOPS
        getMriFlop()->flop_iftCpu += 1 + 1 + 1 + 1 + 1;
        #endif

        #if USE_OPENMP
        #pragma omp parallel for default(none) reduction(+:sumr, sumi) \
         private(expr, cosexpr, sinexpr) \
         shared(j, ktraj, t, kzdeltaz, itraj_x_tpi, itraj_y_tpi, kziztpi, \
                fm, kdata_r, kdata_i)
        #endif
        for (i = 0; i < num_k; i++) { // i is the time points in k-space
            expr = (ktraj[i].x * itraj_x_tpi +
                    ktraj[i].y * itraj_y_tpi + kziztpi) +
                   (fm[j] * t[i]);
            #if ENABLE_DOUBLE_PRECISION
                cosexpr = cos(expr); sinexpr = sin(expr);
            #else
                cosexpr = cosf(expr); sinexpr = sinf(expr);
            #endif
            sumr += (cosexpr * kdata_r[i]) - (sinexpr * kdata_i[i]);
            sumi += (sinexpr * kdata_r[i]) + (cosexpr * kdata_i[i]);

            #if COMPUTE_FLOPS
            // sin: 13 flop; cos: 12 flop; +,-,*,/: 1
            getMriFlop()->flop_iftCpu += 3 + 3 + 2 + 2 + 6 + 12 + 13 + 5 + 5;
            #endif
        }

        idata_r[j] = sumr;  // Real part
        idata_i[j] = sumi;  // Imaginary part
    }

    stopMriTimer(getMriTimer()->timer_iftCpu);
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

