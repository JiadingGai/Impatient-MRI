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

    File Name   [smvmGpu.cuh]

    Synopsis    [Sparse matrix-vector multiplications.]

    Description [This part is mainly from NVIDIA Corporation. Please read
        their license agreement before you use it.]

    Revision    [0.1; Initial build; Xiao-Long Wu, ECE UIUC]
    Revision    [0.1.1; Code cleaning; Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Further optimization, Code cleaning, Adding more comments;
                 Xiao-Long Wu, ECE UIUC]
    Date        [10/27/2010]

 *****************************************************************************/

#ifndef SPARSEMULT_GPU_CUH
#define SPARSEMULT_GPU_CUH

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
smvmGpu(
    FLOAT_T *Cf_r_d, FLOAT_T *Cfi_d,             // Output vector
    const FLOAT_T *xr_d, const FLOAT_T *xi_d,   // Input vector
    const int *Ap_d, const int *Aj_d,           // Matrix in CSR format
    const FLOAT_T *Ax_d, const int num_rows);

// CSR SpMV kernels based on a vector model (one warp per row)
    __global__ void
smvmGpu_kernel(
    FLOAT_T *y,
    const FLOAT_T *x, const int *Ap, const int *Aj, const FLOAT_T *Ax,
    const int num_rows);

// CSR SpMV kernels based on a scalar model (one thread per row)
    __global__ void
smvmGpuScalarKernel(
    FLOAT_T *y,
    const FLOAT_T *x, const int *Ap, const int *Aj, const FLOAT_T *Ax,
    const int num_rows);
/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

#endif // SPARSEMULT_GPU_CUH

