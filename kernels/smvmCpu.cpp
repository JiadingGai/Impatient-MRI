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

    File Name   [smvmCpu.cpp]

    Synopsis    []

    Description []

    Revision    [0.1; Initial build; Yue Zhuo, BIOE UIUC]
    Revision    [0.1.1; Code cleaning; Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Further optimization, Code cleaning, Adding comments;
                 Xiao-Long Wu, ECE UIUC]
    Date        [10/27/2010]

*****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

#include <stdio.h>
#include <string.h>

#include <smvmCpu.h>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [CPU version of the sparse matrix-vector multiplication.]    */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    void
smvmCpu(
    FLOAT_T *y_r, FLOAT_T *y_i,                 // Output vector
    const FLOAT_T *x_r, const FLOAT_T *x_i,     // Input vector
    const CooMatrix *c)                         // Matrix in COO format
{
    startMriTimer(getMriTimer()->timer_smvmCpu);
    
    for (int i = 0; i < (c->num_rows); i++) 
	{
		y_r[i] = 0.0f;
		y_i[i] = 0.0f;
	}

    for(int k = 0; k < (c->num_nonzeros); k++)
	{
	    y_r[c->I[k]] += c->V[k] * x_r[c->J[k]];
	    y_i[c->I[k]] += c->V[k] * x_i[c->J[k]];
	}

    stopMriTimer(getMriTimer()->timer_smvmCpu);
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [CPU version of the sparse matrix-vector multiplication with */
/*      the transposed matrix.]                                              */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    void
smvmTransCpu(
    FLOAT_T *y_r, FLOAT_T *y_i,                 // Output vector
    const FLOAT_T *x_r, const FLOAT_T *x_i,     // Input vector
    const CooMatrix *c)                         // Matrix in COO format
{
    startMriTimer(getMriTimer()->timer_smvmCpu);

    for (int i = 0; i < (c->num_cols); i++) 
	{
		y_r[i] = 0.0f;
		y_i[i] = 0.0f;
	}

    for(int k = 0; k < (c->num_nonzeros); k++)
	{
	    y_r[c->J[k]] += c->V[k] * x_r[c->I[k]];
	    y_i[c->J[k]] += c->V[k] * x_i[c->I[k]];
	}

    stopMriTimer(getMriTimer()->timer_smvmCpu);
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

