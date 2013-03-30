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

    File Name   [dotProductCpu.cpp]

    Synopsis    [CPU version of the complex number dot product.]

    Description []

    Revision    [0.1; Initial build; Yue Zhuo, BIOE UIUC]
    Revision    [0.1.1; Add OpenMP, Code cleaning; Xiao-Long Wu, ECE UIUC]
    Date        [03/23/2010]

*****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <stdio.h>
#include <string.h>

// MRI project related files
#include <tools.h>
#include <structures.h>

#include "dotProductCpu.h"

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [CPU version of the complex number dot product.]             */
/*                                                                           */
/*  Description [Real part only of inner product.]                           */
/*                                                                           */
/*===========================================================================*/

    void
dotProductCpu(
    FLOAT_T *output,
    FLOAT_T *A_r, FLOAT_T *A_i,
    FLOAT_T *B_r, FLOAT_T *B_i,
    const int num_elements
    )
{
    startMriTimer(getMriTimer()->timer_dotProductCpu);

    *output = 0.0;
    FLOAT_T sum = 0.0;

    // Preferably not using multi-thread because memory to computation ratio
    // is too big.
    #if 0 //USE_OPENMP
    #pragma omp parallel for default(none) reduction(+:sum) \
     shared(A_r, A_i, B_r, B_i, num_elements)
    #endif
    for (int i = 0; i < num_elements; i++) {
        sum += A_r[i] * B_r[i] + A_i[i] * B_i[i]; // only need real part

        #if COMPUTE_FLOPS
        getMriFlop()->flop_dotProductCpu += 4;
        #endif
    }
    (*output) = sum;

    stopMriTimer(getMriTimer()->timer_dotProductCpu);
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

