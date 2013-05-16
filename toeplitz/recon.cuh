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

    File Name   [recon.cuh]

    Synopsis    [Toeplitz's CG solver header.]

    Description []

    Revision    [1.0; Sam S. Stone, ECE UIUC]
    Revision    [2.0; Jiading Gai, Beckman Institute UIUC and 
                      Xiao-Long Wu, ECE UIUC]
    Date        [03/23/2011]

*****************************************************************************/

#ifndef TOEPLITZ_RECON_CUH
#define TOEPLITZ_RECON_CUH
int toeplitz_recon(int toe_argc, char *toe_argv[], 
                   const bool enable_total_variation,
                   const bool enable_regularization,
                   const bool enable_finite_difference, 
				   const bool enable_reuseQ,
                   const CooMatrix *c,
				   float *Fhd_R, float *Fhd_I,
                   float **Q_R, float **Q_I);
#endif
