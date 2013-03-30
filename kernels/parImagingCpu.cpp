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

    File Name   [parImagingCpu.cpp]

    Synopsis    [CPU version of the parallel imaging kernel wrappers.]

    Description []

    Revision    [1.0a; Initial build; Jiading Gai, Beckman Institute,
                 Xiao-Long Wu, ECE UIUC]
    Date        [10/24/2010]

 *****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <stdio.h>
#include <string.h>

// Project header files
#include <structures.h>
#include <ftCpu.h>
#include <pointMultCpu.h>
#include <addCpu.h>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [CPU entry of the parallel imaging kernel wrappers.]         */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    void
parallelFtCpu(
    FLOAT_T *pGx_r,         FLOAT_T *pGx_i,
    const FLOAT_T  *idata_r, const FLOAT_T  *idata_i,
    const DataTraj *ktraj,   const DataTraj *itraj,
    const FLOAT_T *fm,    const FLOAT_T  *time,
    const int num_k, const int num_i,
    const FLOAT_T  *sensi_r, const FLOAT_T *sensi_i,
    const int num_coil
    )
{
    startMriTimer(getMriTimer()->timer_parallelFtCpu);

    FLOAT_T *tmp_r = mriNewCpu<FLOAT_T>(num_i);
    FLOAT_T *tmp_i = mriNewCpu<FLOAT_T>(num_i);

    for (int l = 0; l < num_coil; l++) {
        pointMultCpu(tmp_r, tmp_i, sensi_r + l * num_i, sensi_i + l * num_i,
                     idata_r, idata_i, num_i);
        ftCpu(pGx_r + l * num_k, pGx_i + l * num_k, tmp_r, tmp_i, ktraj, itraj,
              fm, time, num_k, num_i);
    }

    mriDeleteCpu(tmp_r);
    mriDeleteCpu(tmp_i);

    stopMriTimer(getMriTimer()->timer_parallelFtCpu);
}

    void
parallelIftCpu(
    FLOAT_T  *output_r,      FLOAT_T  *output_i,
    const FLOAT_T  *input_r, const FLOAT_T  *input_i,
    const DataTraj *ktraj,   const DataTraj *itraj,
    const FLOAT_T *fm, const FLOAT_T *time,
    const int num_k, const int num_i,
    const FLOAT_T  *sensi_r, const FLOAT_T  *sensi_i, const int num_coil
    )
{
    startMriTimer(getMriTimer()->timer_parallelIftCpu);

    // Set output_r and output_i to zero before accumulating
    memset(output_r, 0, num_i * sizeof(FLOAT_T));
    memset(output_i, 0, num_i * sizeof(FLOAT_T));

    FLOAT_T *tmp_r = mriNewCpu<FLOAT_T>(num_i);
    FLOAT_T *tmp_i = mriNewCpu<FLOAT_T>(num_i);

    for (int l = 0; l < num_coil; l++) {
        iftCpu(tmp_r, tmp_i, input_r + l * num_k, input_i + l * num_k,
               ktraj, itraj, fm, time, num_k, num_i
               );
        pointMult_conjCpu(tmp_r, tmp_i,
                          sensi_r + l * num_i, sensi_i + l * num_i,
                          tmp_r, tmp_i, num_i);
        addCpu(output_r, output_i, output_r,
               output_i, tmp_r, tmp_i, MRI_ONE, num_i);
    }

    mriDeleteCpu(tmp_r);
    mriDeleteCpu(tmp_i);

    stopMriTimer(getMriTimer()->timer_parallelIftCpu);
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

