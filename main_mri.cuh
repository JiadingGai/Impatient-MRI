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

    File Name   [main_mri.cuh]

    Synopsis    [Main/Toppest function to launch the MRI program.]

    Description []

    Revision    [0.1; Initial build; Yue Zhuo, BIOE UIUC]
    Revision    [0.1.1; Code cleaning; Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Further optimization, Code cleaning, Adding more comments;
                 Xiao-Long Wu, ECE UIUC, Jiading Gai, Beckman Institute]
    Date        [10/27/2010]

*****************************************************************************/

#ifndef MAIN_MRI_CUH
#define MAIN_MRI_CUH

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Namespace used                                                           */
/*---------------------------------------------------------------------------*/

using namespace std;

/*---------------------------------------------------------------------------*/
/*  Macro definitions                                                        */
/*---------------------------------------------------------------------------*/

static const char * const mriSolver_name         = "mriSolver";
static const char * const mriSolver_version      = "3.1 alpha";
static const char * const mriSolver_release_date = "03/21/2012";

/*---------------------------------------------------------------------------*/
/*  Function prototypes                                                      */
/*---------------------------------------------------------------------------*/

    void
mriSolverProgramHeader(FILE *fp);

    void
mriSolverVersion(FILE *fp);

    void
mriSolverLicense(FILE *fp);

    bool
toeplitz(
    const string &input_dir, const string &output_dir,
    const int cg_num, const bool enable_gpu, const bool enable_multi_gpu,
    const bool enable_cpu, const bool enable_regularization,
    const bool enable_finite_difference, const float fd_penalizer,
    const bool enable_total_variation, const int tv_num,
    const bool enable_tv_update, const bool enable_toeplitz_direct, 
	const bool enable_toeplitz_gridding, float gridOS_Q, float gridOS_FHD, 
	const float ntime_segments,	const int gpu_id,
	const bool enable_reuseQ, const string reuse_Qlocation,
	const bool enable_writeQ);

    bool
bruteForce(
    const string &input_dir, const string &output_dir,
    const int cg_num, const bool enable_gpu, const bool enable_multi_gpu,
    const bool enable_cpu, const bool enable_regularization,
    const bool enable_finite_difference, const float fd_penalizer,
    const bool enable_total_variation, const int tv_num,
    const bool enable_tv_update,
	const int gpu_id);

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

#endif // MAIN_MRI_CUH

