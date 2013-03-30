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

    File Name   [multiplyCpu.cpp]

    Synopsis    [CPU version of the complex number vector multiplication.]

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

// Project header files
#include <tools.h>
#include <structures.h>

#include <multiplyCpu.h>

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

    void
multiplyCpu(
    FLOAT_T *ouptut_r,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i,
    const int num_elements
    )
{
    startMriTimer(getMriTimer()->timer_multiplyCpu);

    // Preferably not using multi-thread because cache may get dirty due to
    // multi-threading.
    #if 0 //USE_OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < num_elements; i++) {
        ouptut_r[i] = MRI_POINT_FIVE /
                      sqrt(a_r[i]*a_r[i] + a_i[i]*a_i[i] +
                           b_r[i]*b_r[i] + b_i[i]*b_i[i] + MRI_SMOOTH_FACTOR);

        #if COMPUTE_FLOPS // sqrt: 13 for software sqrt, 1 for SPF
        getMriFlop()->flop_multiplyCpu += 9 + 1;
        #endif
    }

    stopMriTimer(getMriTimer()->timer_multiplyCpu);
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

