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

    File Name   [ftCpu.h]

    Synopsis    [The CPU version of Fourier transform and inverse Fourier
        transform.]

    Description []

    Revision    [0.1; Initial build; Yue Zhuo, BIOE UIUC]
    Revision    [0.1.1; Add OpenMP, Code cleaning; Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Further optimization, Code cleaning, Adding more comments;
                 Xiao-Long Wu, ECE UIUC, Jiading Gai, Beckman Institute]
    Date        [10/27/2010]

 *****************************************************************************/

#ifndef FT_CPU_H
#define FT_CPU_H

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Function prototypes                                                      */
/*---------------------------------------------------------------------------*/

    void
ftCpu(FLOAT_T *kdata_r, FLOAT_T *kdata_i,
      const FLOAT_T *idata_r, const FLOAT_T *idata_i,
      const DataTraj *ktraj, const DataTraj *itraj,
      const FLOAT_T *fm, const FLOAT_T *t,
      const int num_k, const int num_i
      );

    void
iftCpu(FLOAT_T *idata_r, FLOAT_T *idata_i,
       const FLOAT_T *kdata_r, const FLOAT_T *kdata_i,
       const DataTraj *ktraj, const DataTraj *itraj,
       const FLOAT_T *fm, const FLOAT_T *t,
       const int num_k, const int num_i
       );

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

#endif // FT_CPU_H

