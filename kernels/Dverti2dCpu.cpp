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

    File Name   [Dverti2dCpu.cpp]

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

#include <Dverti2dCpu.h>

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
Dverti2dCpu(
    FLOAT_T *s_real, FLOAT_T *s_imag,
    const FLOAT_T *p_real, const FLOAT_T *p_imag,
    const unsigned int num_row, const unsigned int num_col)
{
    startMriTimer(getMriTimer()->timer_Dverti2dCpu);

    unsigned int i = 0, j = 0;

    // column wise finite difference DF1
    for (i = 0; i < num_col; i++) {
        for (j = 0; j < num_row - 1; j++) {
            // for elements not concerned with the periodic condition
            s_real[i * num_row + j] = p_real[i * num_row + j] -
                                      p_real[i * num_row + j + 1];
            s_imag[i * num_row + j] = p_imag[i * num_row + j] -
                                      p_imag[i * num_row + j + 1];

            #if COMPUTE_FLOPS
            getMriFlop()->flop_Dverti2dCpu += 2;
            #endif
        }
    }

    for (i = 0; i < num_col; i++) {
        // for elements concerned with the periodic condition
        s_real[i*num_row+num_row-1] = p_real[i*num_row+num_row-1] -
                                      p_real[i*num_row];
        s_imag[i*num_row+num_row-1] = p_imag[i*num_row+num_row-1] -
                                      p_imag[i*num_row];

        #if COMPUTE_FLOPS
        getMriFlop()->flop_Dverti2dCpu += 2;
        #endif
    }

    stopMriTimer(getMriTimer()->timer_Dverti2dCpu);
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

