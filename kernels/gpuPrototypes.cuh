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

    File Name   [gpuPrototypes.cuh]

    Synopsis    [Function interface between C++ and CUDA programs. Without
        this file, compilers will complain.]

    Description [Remember to update all GPU function prototypes.]
    
    FIXME       [Try to remove this file.]

    Revision    [0.1; Initial build; Xiao-Long Wu, ECE UIUC]
    Date        [03/23/2010]

 *****************************************************************************/

#ifndef GPUPROTOTYPES_H
#define GPUPROTOTYPES_H

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA libraries
#include <cutil.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <multithreading.h>

// Project header files
#include <structures.h>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Function prototypes                                                      */
/*---------------------------------------------------------------------------*/

//---------------------------------------------------------------------------
// cgGpu.cuh
//---------------------------------------------------------------------------

    extern
    void
bruteForceGpu(
    FLOAT_T **idata_r_h, FLOAT_T **idata_i_h,
    const FLOAT_T *kdata_r_cpu, const FLOAT_T *kdata_i_cpu,
    const DataTraj *ktraj_cpu, const DataTraj *itraj_h,
    const FLOAT_T *fm_h, const FLOAT_T *time_cpu,
    const CooMatrix *c,
    const int cg_num, const int num_k_cpu, const int num_i,
    const bool enable_regularization,
    const bool enable_finite_difference, const FLOAT_T fd_penalizer,
    const bool enable_total_variation, const int tv_num,
    const FLOAT_T *sensi_r_h, const FLOAT_T *sensi_i_h, const int num_coil,
    const bool enable_tv_update,
    const string &output_file_gpu_r, const string &output_file_gpu_i,
    const int num_slices);

//---------------------------------------------------------------------------
// bruteForceMgpu.cuh
//---------------------------------------------------------------------------

    extern CUT_THREADPROC 
bruteForceMgpu(CgSolverData *sliceData);

    extern CUT_THREADPROC 
mgpuDummy(int GPU_Idx);

//---------------------------------------------------------------------------
// addGpu.cuh
//---------------------------------------------------------------------------
                                        
    extern
    void
addGpu(
    FLOAT_T *output_r, FLOAT_T *output_i,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i,
    const FLOAT_T alpha, const int num_elements
    );

// Used in main_mri.cu, bruteForceMgpu.cu, and addGpu.cu
    __global__ void
addGpuKernel(
    FLOAT_T *output_r, FLOAT_T *output_i,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i,
    const FLOAT_T alpha, const int num_elements);

//---------------------------------------------------------------------------
// dot_product.h
//---------------------------------------------------------------------------

    extern
    void
dotProductGpu(
    FLOAT_T *output,
    const FLOAT_T *a_r, const FLOAT_T *a_i,
    const FLOAT_T *b_r, const FLOAT_T *b_i,
    const int num_elements);

//---------------------------------------------------------------------------
// ftGpu.cuh
//---------------------------------------------------------------------------

    extern
    void
ftGpu(FLOAT_T *kdata_r_d, FLOAT_T *kdata_i_d,
      const FLOAT_T *idata_r_d, const FLOAT_T *idata_i_d,
      const DataTraj *ktraj_d, const DataTraj *itraj_d,
      const FLOAT_T *fm_d, const FLOAT_T *t_d,
      const int num_k, const int in_num_k, const int num_i
      );

    extern
    void
iftGpu(FLOAT_T *idata_r_d, FLOAT_T *idata_i_d,
       const FLOAT_T *kdata_r_d, const FLOAT_T *kdata_i_d,
       const DataTraj *ktraj_d, const DataTraj *itraj_d,
       const FLOAT_T *fm_d, const FLOAT_T *time_d,
       const int num_k, const int num_k_cpu, const int num_i
       );

//---------------------------------------------------------------------------
// smvmGpu.cuh
//---------------------------------------------------------------------------

    extern
void smvmGpu(
    FLOAT_T *Cf_r_d, FLOAT_T *Cfi_d,            // Output vector
    const FLOAT_T *xr_d, const FLOAT_T *xi_d,   // Input vector
    const int *Ap_d, const int *Aj_d,           // Matrix in CSR format
    const FLOAT_T *Ax_d, const int num_rows);

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

#endif // GPUPROTOTYPES_H
