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

    File Name   [bruteForceGpu.cu]

    Synopsis    [The main body of the GPU-version CG solver.]

    Description []

    Revision    [0.1; Initial build; Yue Zhuo, BIOE UIUC]
    Revision    [0.1.1; Code cleaning; Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Further optimization, Code cleaning, Adding comments;
                 Xiao-Long Wu, ECE UIUC, Jiading Gai, Beckman Institute]
    Date        [10/27/2010]

 *****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <stdio.h>

// CUDA libraries
#include <cutil.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>

// Project header files
#include <xcpplib_process.h>
#include <xcpplib_typesGpu.cuh>
#include <tools.h>
#include <structures.h>
#include <parImagingGpu.cuh>
#include <DHWD2dGpu.cuh>
#include <Dverti2dGpu.cuh>
#include <Dhori2dGpu.cuh>
#include <multiplyGpu.cuh>
#include <pointMultGpu.cuh>

#include <ftCpu.h>
#include <Dverti2dCpu.h>
#include <Dhori2dCpu.h>
#include <addCpu.h>

#include <gpuPrototypes.cuh>
#include <bruteForceGpu.cuh>

#include <smvmCpu.h>
#include <smvmGpu.cuh>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Conjugate Gradient solver: GPU kernel interface.]           */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*  Note        [Regarding variable naming, all variables on the host side   */
/*      should be with the postfix of "_h" or "_cpu". Variables copied to    */
/*      to the device side should be with "_h". Variables need padding or    */
/*      other modifications should be with "_cpu".                           */
/*      For the device side variables, they should be the same with the      */
/*      names on the bruteForceCpu.cpp file. We do this in order to keep the easy    */
/*      mapping from the CPU-version code to the GPU-version.]               */
/*                                                                           */
/*===========================================================================*/

// Kernel debugging macros
#if DEBUG_KERNEL_MSG
    #define DEBUG_CGGPU  true
#else
    #define DEBUG_CGGPU  false
#endif

// Note: When developing new versions, these options should be enabled for
//       regression test.
#if DEBUG_CGGPU
    // ADJOINT_TEST_GPU will disable all other tests.
    #define ADJOINT_TEST_GPU              false
    #define COMPUTE_CG_COST_FUNC_GPU      true
    #define COMPUTE_REG_FD_COST_FUNC_GPU  true
#else
    #define ADJOINT_TEST_GPU              false
    #define COMPUTE_CG_COST_FUNC_GPU      false
    #define COMPUTE_REG_FD_COST_FUNC_GPU  false
#endif

#define NEW_GPU_REG  true

#define PARALLEL_IMAGING_GPU  true

// FIXME: Since some data are same for all slices, we don't need to
// do the allocation/deallocation for multiple times.

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
    const int num_slices)
{

    #if DEBUG_CGGPU
    msg(MSG_PLAIN, "\n");
    msg(1, "bruteForceGpu() begin");

    msg(2, "num_k_cpu: %d, num_i: %d", num_k_cpu, num_i);
    #endif

    startMriTimer(getMriTimer()->timer_bruteForceGpu);

    // =======================================================================
    //  Padding the kdata to the multiples of power of 2.
    //  For example, we must pad 3770 to 4096 for faster manipulation in GPU.
    //  Note: The idata is assumed to be the power of 2.
    // =======================================================================

    const int num_k = getLeastPowerOfTwo(num_k_cpu);
    #if DEBUG_CGGPU
    msg(2, "num_k (after getLeastPowerOfTwo()): %d", num_k);
    #endif

    DataTraj *ktraj_h  = mriNewCpu<DataTraj>(num_k);
    FLOAT_T *kdata_r_h = mriNewCpu<FLOAT_T>(num_k * num_coil);
    FLOAT_T *kdata_i_h = mriNewCpu<FLOAT_T>(num_k * num_coil);

    // FIXME: Need to fuse both loops
    for (int i = 0; i < num_k_cpu; i++) {
        ktraj_h[i].x = ktraj_cpu[i].x;
        ktraj_h[i].y = ktraj_cpu[i].y;
    }
    for (int i = num_k_cpu; i < num_k; i++) {
        ktraj_h[i].x = MRI_ZERO;
        ktraj_h[i].y = MRI_ZERO;
    }

    // FIXME: Need to speed up
    // kdata_r and kdata_i are related to num_coil.
    for (int l=0;l < num_coil; l++) {
        for (int i = 0; i < num_k; i++) {
            if(i < num_k_cpu) {
                kdata_r_h[l*num_k+i] = kdata_r_cpu[l*num_k_cpu+i];
                kdata_i_h[l*num_k+i] = kdata_i_cpu[l*num_k_cpu+i];
            } else {
                kdata_r_h[l*num_k+i] = MRI_ZERO;
                kdata_i_h[l*num_k+i] = MRI_ZERO;
            }
        }
    }

    FLOAT_T *time_h = padVectorPowerOfTwo(time_cpu, num_k_cpu);

    // FIXME: Need to check if the image size is power of 2.

    // =======================================================================
    //  Copy host data to device
    // =======================================================================

    // Generic data
    // ============

    DataTraj *ktraj  = mriNewGpu<DataTraj>(num_k);
    DataTraj *itraj  = mriNewGpu<DataTraj>(num_i);
    FLOAT_T *kdata_r = mriNewGpu<FLOAT_T>(num_k * num_coil);
    FLOAT_T *kdata_i = mriNewGpu<FLOAT_T>(num_k * num_coil);
    FLOAT_T *idata_r = mriNewGpu<FLOAT_T>(num_i);
    FLOAT_T *idata_i = mriNewGpu<FLOAT_T>(num_i);
    FLOAT_T *sensi_r = mriNewGpu<FLOAT_T>(num_i * num_coil);
    FLOAT_T *sensi_i = mriNewGpu<FLOAT_T>(num_i * num_coil);
    FLOAT_T *fm      = mriNewGpu<FLOAT_T>(num_i);
    FLOAT_T *time    = mriNewGpu<FLOAT_T>(num_k);

    mriCopyHostToDevice<DataTraj>(ktraj, ktraj_h, num_k);
    mriCopyHostToDevice<DataTraj>(itraj, itraj_h, num_i);
    mriCopyHostToDevice<FLOAT_T>(kdata_r, kdata_r_h, num_k * num_coil);
    mriCopyHostToDevice<FLOAT_T>(kdata_i, kdata_i_h, num_k * num_coil);
    mriCopyHostToDevice<FLOAT_T>(idata_r, *idata_r_h, num_i);
    mriCopyHostToDevice<FLOAT_T>(idata_i, *idata_i_h, num_i);
    mriCopyHostToDevice<FLOAT_T>(sensi_r, sensi_r_h, num_i * num_coil);
    mriCopyHostToDevice<FLOAT_T>(sensi_i, sensi_i_h, num_i * num_coil);
    mriCopyHostToDevice<FLOAT_T>(fm, fm_h, num_i);
    mriCopyHostToDevice<FLOAT_T>(time, time_h, num_k);

    #if ADJOINT_TEST_GPU // ==================================================

    FLOAT_T * a_r_h = mriNewCpu<FLOAT_T>(num_i);
    FLOAT_T * a_i_h = mriNewCpu<FLOAT_T>(num_i);
    FLOAT_T * b_r_h = mriNewCpu<FLOAT_T>(num_k);
    FLOAT_T * b_i_h = mriNewCpu<FLOAT_T>(num_k);
    FLOAT_T * c1_r_h = mriNewCpu<FLOAT_T>(num_k);
    FLOAT_T * c1_i_h = mriNewCpu<FLOAT_T>(num_k);
    FLOAT_T * c2_r_h = mriNewCpu<FLOAT_T>(num_i);
    FLOAT_T * c2_i_h = mriNewCpu<FLOAT_T>(num_i);

    for (int i = 0; i < num_i; i++) {
        #if 0
        a_r_h[i] = 1; a_i_h[i] = 1;
        #else // Produce the data with big variations
        if (i % 10) { a_r_h[i] = 1; a_i_h[i] = num_k_cpu; }
        else        { a_r_h[i] = num_i; a_i_h[i] = 1; }
        #endif
        c2_r_h[i] = 0;  c2_i_h[i] = 0;
    }
    for (int i = 0; i < num_k_cpu; i++) {
        #if 0
        b_r_h[i] = 1; b_i_h[i] = 1;
        #else // Produce the data with big variations
        if (i % 10) { b_r_h[i] = num_i+1; b_i_h[i] = 1; }
        else        { b_r_h[i] = 1; b_i_h[i] = num_k_cpu+2; }
        #endif
        c1_r_h[i] = 0;  c1_i_h[i] = 0;
    }
    // Padded elements
    for (int i = num_k_cpu; i < num_k; i++) {
        b_r_h[i] = 0; b_i_h[i] = 0;
        c1_r_h[i] = 0;  c1_i_h[i] = 0;
    }

    FLOAT_T * a_r = mriNewGpu<FLOAT_T>(num_i);
    FLOAT_T * a_i = mriNewGpu<FLOAT_T>(num_i);
    FLOAT_T * b_r = mriNewGpu<FLOAT_T>(num_k);
    FLOAT_T * b_i = mriNewGpu<FLOAT_T>(num_k);
    FLOAT_T * c1_r = mriNewGpu<FLOAT_T>(num_k);
    FLOAT_T * c1_i = mriNewGpu<FLOAT_T>(num_k);
    FLOAT_T * c2_r = mriNewGpu<FLOAT_T>(num_i);
    FLOAT_T * c2_i = mriNewGpu<FLOAT_T>(num_i);

    mriCopyHostToDevice<FLOAT_T>(a_r, a_r_h, num_i);
    mriCopyHostToDevice<FLOAT_T>(a_i, a_i_h, num_i);
    mriCopyHostToDevice<FLOAT_T>(b_r, b_r_h, num_k);
    mriCopyHostToDevice<FLOAT_T>(b_i, b_i_h, num_k);

    FLOAT_T o1 = 0, o2 = 0;

    ftGpu(c1_r, c1_i, a_r, a_i, ktraj, itraj, fm, time,
          num_k, num_k_cpu, num_i);
    dotProductGpu(&o1, c1_r, c1_i, b_r, b_i, num_k);

    iftGpu(c2_r, c2_i, b_r, b_i, ktraj, itraj, fm,
           time, num_k, num_k_cpu, num_i);
    dotProductGpu(&o2, a_r, a_i, c2_r, c2_i, num_i);

    msg(1, "gpu o1: %.8f\n", o1); msg(1, "gpu o2: %.8f\n", o2);
    msg(1, "gpu o1/o2: %.8f, o2/o1: %.8f\n", o1/o2, o2/o1);

    exit(1);
    #endif // End of ADJOINT_TEST_GPU ========================================

    // Sparse matrix multiplication
    // ============================

    CsrMatrix c_csr, c_trans_csr;
    if (enable_regularization) {
        #if DEBUG_CGGPU
        msg(2, "c->num_rows: %d\n", c->num_rows);
        msg(2, "c->num_cols: %d\n", c->num_cols);
        msg(2, "c->num_nonzeros: %d\n", c->num_nonzeros);
        #endif

        // Convert matrix C to CSR format.
        c_csr = mtx2Csr(c->I, c->J, c->V,
                        c->num_rows, c->num_cols, c->num_nonzeros);

        // Convert transposed matrix C_trans to CSR format.
        // In fact, we just need to switch the row and column indices.
        c_trans_csr = mtx2Csr(c->J, c->I, c->V,
                              c->num_cols, c->num_rows, c->num_nonzeros);
    }

    int *c_Ap_d = NULL, *c_Aj_d = NULL;
    FLOAT_T *c_Ax_d = NULL;
    int *c_trans_Ap_d = NULL, *c_trans_Aj_d = NULL;
    FLOAT_T *c_trans_Ax_d = NULL;

    if (enable_regularization) {
        int num_Ap = (c_csr.num_rows + 1);
        int num_Aj = (c_csr.num_nonzeros);
        int num_Ax = (c_csr.num_nonzeros);
        c_Ap_d  = mriNewGpu<int>(num_Ap);
        c_Aj_d  = mriNewGpu<int>(num_Aj);
        c_Ax_d  = mriNewGpu<FLOAT_T>(num_Ax);

        int num_trans_Ap = (c_trans_csr.num_rows + 1);
        int num_trans_Aj = (c_trans_csr.num_nonzeros);
        int num_trans_Ax = (c_trans_csr.num_nonzeros);
        c_trans_Ap_d  = mriNewGpu<int>(num_trans_Ap);
        c_trans_Aj_d  = mriNewGpu<int>(num_trans_Aj);
        c_trans_Ax_d  = mriNewGpu<FLOAT_T>(num_trans_Ax);

        #if 1 //DEBUG_CGGPU
        msg(2, "bruteForceGpu(): total memory size for SMVM: %d bytes\n",
             (num_Ap + num_Aj + num_trans_Ap + num_trans_Aj) * sizeof(int) + (num_Ax + num_trans_Ax) * sizeof(FLOAT_T));
        #endif

        mriCopyHostToDevice<int>    (c_Ap_d, c_csr.Ap, num_Ap);
        mriCopyHostToDevice<int>    (c_Aj_d, c_csr.Aj, num_Aj);
        mriCopyHostToDevice<FLOAT_T>(c_Ax_d, c_csr.Ax, num_Ax);
        mriCopyHostToDevice<int>(c_trans_Ap_d, c_trans_csr.Ap, num_trans_Ap);
        mriCopyHostToDevice<int>(c_trans_Aj_d, c_trans_csr.Aj, num_trans_Aj);
        mriCopyHostToDevice<FLOAT_T>(c_trans_Ax_d, c_trans_csr.Ax,
                                      num_trans_Ax);
    }

    // =======================================================================
    //  Define CG variables
    // =======================================================================

    FLOAT_T gamma        = MRI_ZERO;
    FLOAT_T newinprod    = MRI_ZERO;
    FLOAT_T oldinprod    = MRI_ZERO;
    FLOAT_T alpha        = MRI_ZERO;
    FLOAT_T alpha_d_grad = MRI_ZERO;
    FLOAT_T alpha_q      = MRI_ZERO;
    FLOAT_T alpha_CtCd   = MRI_ZERO;

    // G*x
    FLOAT_T *Af_r = mriNewGpu<FLOAT_T>(num_k * num_coil);
    FLOAT_T *Af_i = mriNewGpu<FLOAT_T>(num_k * num_coil);
    FLOAT_T *Cf_r = NULL; // C*x
    FLOAT_T *Cf_i = NULL;
    FLOAT_T *Cd_r = NULL; // C*x
    FLOAT_T *Cd_i = NULL;
    if (enable_regularization) {
        Cf_r = mriNewGpu<FLOAT_T>(2*num_i);
        Cf_i = mriNewGpu<FLOAT_T>(2*num_i);
        Cd_r = mriNewGpu<FLOAT_T>(2*num_i);
        Cd_i = mriNewGpu<FLOAT_T>(2*num_i);
    }
    // y-G*x
    FLOAT_T *y_Af_r = mriNewGpu<FLOAT_T>(num_k * num_coil);
    FLOAT_T *y_Af_i = mriNewGpu<FLOAT_T>(num_k * num_coil);
    // G component of grad
    FLOAT_T *GtAf_r = mriNewGpu<FLOAT_T>(num_i);
    FLOAT_T *GtAf_i = mriNewGpu<FLOAT_T>(num_i);
    // C component of grad
    FLOAT_T *CtCf_r = NULL;
    FLOAT_T *CtCf_i = NULL;
    if (enable_regularization || enable_finite_difference) {
        CtCf_r = mriNewGpu<FLOAT_T>(num_i);
        CtCf_i = mriNewGpu<FLOAT_T>(num_i);
    }
    // grad
    FLOAT_T *grad_r = mriNewGpu<FLOAT_T>(num_i);
    FLOAT_T *grad_i = mriNewGpu<FLOAT_T>(num_i);
    // dir
    FLOAT_T *d_r = mriNewGpu<FLOAT_T>(num_i);
    FLOAT_T *d_i = mriNewGpu<FLOAT_T>(num_i);
    // G component of "dir"
    FLOAT_T *q_r = mriNewGpu<FLOAT_T>(num_k * num_coil);
    FLOAT_T *q_i = mriNewGpu<FLOAT_T>(num_k * num_coil);
    FLOAT_T *CtCd_r = NULL; // C component of "dir"
    FLOAT_T *CtCd_i = NULL;
    if (enable_regularization || enable_finite_difference) {
        CtCd_r = mriNewGpu<FLOAT_T>(num_i);
        CtCd_i = mriNewGpu<FLOAT_T>(num_i);
    }

    FLOAT_T * dhori_r  = NULL;
    FLOAT_T * dhori_i  = NULL;
    FLOAT_T * dverti_r = NULL;
    FLOAT_T * dverti_i = NULL;
    FLOAT_T * w_r      = NULL;
    if (enable_finite_difference || enable_regularization) {
        dhori_r  = mriNewGpu<FLOAT_T>(num_i);
        dhori_i  = mriNewGpu<FLOAT_T>(num_i);
        dverti_r = mriNewGpu<FLOAT_T>(num_i);
        dverti_i = mriNewGpu<FLOAT_T>(num_i);
        w_r      = mriNewGpu<FLOAT_T>(num_i);
    }

    // =======================================================================
    //  TV (total variation) iterations
    // =======================================================================

    for (int tv_i = 0; tv_i < tv_num; tv_i++) {
        if (enable_finite_difference || enable_regularization) {
            // Calculating real part W
            if (tv_i == 0) {
                FLOAT_T *w_r_tmp = mriNewCpu<FLOAT_T>(num_i);
                for(unsigned int i = 0; i < num_i; i++) w_r_tmp[i] = MRI_ONE;
                mriCopyHostToDevice<FLOAT_T>(w_r, w_r_tmp, num_i);
                mriDeleteCpu(w_r_tmp);
            }
        }
        if (enable_total_variation) {
            msg(MSG_PLAIN, "  GPU: TV: %d", tv_i);
            if (enable_finite_difference || enable_regularization) {
                // Calculating real part W
                if (tv_i != 0) {
                    Dhori2dGpu(dhori_r, dhori_i, idata_r, idata_i,
                               sqrt(num_i), sqrt(num_i));
                    Dverti2dGpu(dverti_r, dverti_i, idata_r, idata_i,
                                sqrt(num_i), sqrt(num_i));
                    multiplyGpu(w_r, dhori_r, dhori_i, dverti_r, dverti_i,
                                num_i);
                }
            }
            // Output each TV iteration if necessary
            if (enable_tv_update) {
                msg(MSG_PLAIN, " (update data)");
                FLOAT_T *idata_r_tmp = mriNewCpu<FLOAT_T>(num_i);
                FLOAT_T *idata_i_tmp = mriNewCpu<FLOAT_T>(num_i);
                copyCudaDeviceToHost<FLOAT_T>(idata_r_tmp, idata_r, num_i);
                copyCudaDeviceToHost<FLOAT_T>(idata_i_tmp, idata_i, num_i);
                exportDataCpu(output_file_gpu_r, idata_r_tmp, num_slices*num_i);
                exportDataCpu(output_file_gpu_i, idata_i_tmp, num_slices*num_i);
                mriDeleteCpu(idata_r_tmp); mriDeleteCpu(idata_i_tmp);
            }
            msg(MSG_PLAIN, "\n");
        }

        // CG initialization =================================================

        msg(MSG_PLAIN, "  GPU: CG: 0.");

        // Calculation of the gradient
        // Af = G * x;
        // Cf = C * x;
        //--------------------------------------------------------------------

        #if PARALLEL_IMAGING_GPU
        parallelFtGpu(Af_r, Af_i, idata_r, idata_i, ktraj, itraj, fm, time,
                      num_k, num_k_cpu, num_i, sensi_r, sensi_i, num_coil);
        #else // Original
        ftGpu(Af_r, Af_i, idata_r, idata_i, ktraj, itraj, fm, time,
              num_k, num_k_cpu, num_i);
        #endif

        //if (enable_regularization) {
        //    smvmGpu(Cf_r, Cf_i, idata_r, idata_i, c_Ap_d, c_Aj_d,
        //            c_Ax_d, num_i);
        //}

        // CG iterations =====================================================

        for (int cg_i = 1; cg_i < cg_num; cg_i++) {
            msg(MSG_PLAIN, "%d.", cg_i);

            // Calculation of the gradient
            // grad = G' * (W * (yi-Af) - nder1) - C' * Cf;
            //----------------------------------------------------------------

            // Update
            // r = r + alpha*q;
            addGpu(y_Af_r, y_Af_i, kdata_r, kdata_i, Af_r, Af_i, MRI_NEG_ONE,
                   num_k * num_coil);

            #if PARALLEL_IMAGING_GPU
            parallelIftGpu(GtAf_r, GtAf_i, y_Af_r, y_Af_i, ktraj, itraj, fm,
                           time, num_k, num_k_cpu, num_i, sensi_r, sensi_i,
                           num_coil);
            #else // Original
            iftGpu(GtAf_r, GtAf_i, y_Af_r, y_Af_i, ktraj, itraj, fm,
                   time, num_k, num_k_cpu, num_i);
            #endif

            if (enable_regularization || enable_finite_difference) {
                if (enable_regularization) {
                    smvmGpu(Cf_r, Cf_i, idata_r, idata_i, c_Ap_d, c_Aj_d, 
                            c_Ax_d, 2*num_i);
                     
                    // w is real (w_r), but pointMultGpu takes complex
                    // numbers. Hence, we define a temporary zero vector
                    // zv_temp, which has a length of num_i and is set
                    // to zero. zv_temp is used below twice: first, as
                    // the imaginary part of w_r; second, as a dummy
                    // place holder so that we could use addGpu(.) to
                    // do vector-constant multiplication.
                    FLOAT_T *zv_temp = mriNewGpu<FLOAT_T>(num_i);
                    cutilSafeCall(
                       cudaMemset(zv_temp, 0, num_i*sizeof(FLOAT_T))
                    );
                    pointMultGpu(Cf_r, Cf_i, Cf_r, Cf_i, w_r, zv_temp, 
                                 num_i);
                    pointMultGpu(Cf_r+num_i, Cf_i+num_i, Cf_r+num_i, 
                                 Cf_i+num_i, w_r, zv_temp, num_i);

                    // FIXME: ERROR: We can't transpose the merged rectangle
                    // matrix and multiply it with the image. The width of the
                    // transposed regularization matrix becomes 4 * num_i.
                    // Use the same SMVM function but with transposed data
                    smvmGpu(CtCf_r, CtCf_i, Cf_r, Cf_i, c_trans_Ap_d,
                            c_trans_Aj_d, c_trans_Ax_d, num_i);
                    addGpu(CtCf_r, CtCf_i, zv_temp, zv_temp, CtCf_r, 
                           CtCf_i, fd_penalizer, num_i);
                    mriDeleteGpu(zv_temp);

                } else {
                    DHWD2dGpu(CtCf_r, CtCf_i, idata_r, idata_i, w_r,
                              sqrt(num_i), sqrt(num_i), fd_penalizer);
                }
                addGpu(grad_r, grad_i, GtAf_r, GtAf_i, CtCf_r, CtCf_i,
                       MRI_NEG_ONE, num_i);
            } else {
                mriCopyDeviceToDevice<FLOAT_T>(grad_r, GtAf_r, num_i);
                mriCopyDeviceToDevice<FLOAT_T>(grad_i, GtAf_i, num_i);
            }

            dotProductGpu(&newinprod, grad_r, grad_i, grad_r, grad_i, num_i);

            if (cg_i == 1) {
                mriCopyDeviceToDevice<FLOAT_T>(d_r, grad_r, num_i);
                mriCopyDeviceToDevice<FLOAT_T>(d_i, grad_i, num_i);
            } else {
                if (oldinprod == MRI_ZERO) {
                    gamma = MRI_ZERO;
                } else {
                    gamma = newinprod / oldinprod;
                }
                addGpu(d_r, d_i, grad_r, grad_i, d_r, d_i, gamma, num_i);
            }
            oldinprod = newinprod;

            #if PARALLEL_IMAGING_GPU
            parallelFtGpu(q_r, q_i, d_r, d_i, ktraj, itraj, fm, time,
                    num_k, num_k_cpu, num_i, sensi_r, sensi_i, num_coil);
            #else // Original
            ftGpu(q_r, q_i, d_r, d_i, ktraj, itraj, fm, time,
                  num_k, num_k_cpu, num_i);
            #endif


            if (enable_regularization) {

                    smvmGpu(Cd_r, Cd_i, d_r, d_i, c_Ap_d, c_Aj_d, 
                            c_Ax_d, 2*num_i);

                    // w is real (w_r), but pointMultGpu takes complex
                    // numbers. Hence, we define a temporary zero vector
                    // zv_temp, which has a length of num_i and is set
                    // to zero. zv_temp is used below twice: first, as
                    // the imaginary part of w_r; second, as a dummy
                    // place holder so that we could use addGpu(.) to
                    // do vector-constant multiplication.
                    FLOAT_T *zv_temp = mriNewGpu<FLOAT_T>(num_i);
                    cutilSafeCall(
                       cudaMemset(zv_temp, 0, num_i*sizeof(FLOAT_T))
                    );
                    pointMultGpu(Cd_r, Cd_i, Cd_r, Cd_i, w_r, zv_temp, 
                                 num_i);
                    pointMultGpu(Cd_r+num_i, Cd_i+num_i, Cd_r+num_i, 
                                 Cd_i+num_i, w_r, zv_temp, num_i);


                    // FIXME: ERROR: We can't transpose the merged rectangle
                    // matrix and multiply it with the image. The width of the
                    // transposed regularization matrix becomes 4 * num_i.
                    // Use the same SMVM function but with transposed data
                    smvmGpu(CtCd_r, CtCd_i, Cd_r, Cd_i, c_trans_Ap_d,
                            c_trans_Aj_d, c_trans_Ax_d, num_i);

                    addGpu(CtCd_r, CtCd_i, zv_temp, zv_temp, CtCd_r, 
                           CtCd_i, fd_penalizer, num_i);
                    mriDeleteGpu(zv_temp);

            } else if (enable_finite_difference) {
                    DHWD2dGpu(CtCd_r, CtCd_i, d_r, d_i, w_r,
                          sqrt(num_i), sqrt(num_i), fd_penalizer);
            }

            // Calculation of the alpha
            // if (enable_regularization)
            //     alpha = real(dir' * grad) / real(q'*(W*q) + CtCd'*CtCd);
            // else if (enable_finite_difference)
            //     alpha = real(dir' * grad) / real(q'*q + dt*CtCd);
            // endif
            // ---------------------------------------------------------------

            dotProductGpu(&alpha_d_grad, d_r, d_i, grad_r, grad_i, num_i);
            dotProductGpu(&alpha_q, q_r, q_i, q_r, q_i, num_k * num_coil);
            if (enable_regularization) {
                //Jiading GAI
                dotProductGpu(&alpha_CtCd, CtCd_r, CtCd_i, d_r, d_i, num_i);
                //dotProductGpu(&alpha_CtCd, CtCd_r, CtCd_i, CtCd_r, CtCd_i, 
                //              num_i);
            } else if (enable_finite_difference) {
                dotProductGpu(&alpha_CtCd, CtCd_r, CtCd_i, d_r, d_i, num_i);
            } else {
                alpha_CtCd = 0;
            }
            alpha = alpha_d_grad / (alpha_q + alpha_CtCd);

            // Update
            // Af = Af + q;
            // Cf = Cf + CtCd;
            // x = x + dir;
            //----------------------------------------------------------------

            addGpu(Af_r, Af_i, Af_r, Af_i, q_r, q_i,
                   alpha, num_k * num_coil);

            //Jiading GAI
            //if (enable_regularization) {
            //    addGpu(Cf_r, Cf_i, Cf_r, Cf_i, CtCd_r, CtCd_i, alpha, num_i);
            //}

            addGpu(idata_r, idata_i, idata_r, idata_i, d_r, d_i,
                   alpha, num_i);

            #if COMPUTE_CG_COST_FUNC_GPU
            #if PARALLEL_IMAGING_GPU == false
            #error PARALLEL_IMAGING_GPU must be enabled for this section.
            #endif
            // Compute cost function of data consistency term
            // Ff (idata) => b = (Ff - y) (kdata), consistency = sum(|b|^2)
            // ---------------------------------------------------------------

            FLOAT_T *b_r = mriNewGpu<FLOAT_T>(num_k * num_coil);
            FLOAT_T *b_i = mriNewGpu<FLOAT_T>(num_k * num_coil);
            parallelFtGpu(b_r, b_i, idata_r, idata_i, ktraj, itraj, fm, time,
                          num_k, num_k_cpu, num_i, sensi_r, sensi_i, num_coil);
            addGpu(b_r, b_i, b_r, b_i, kdata_r, kdata_i, MRI_NEG_ONE,
                   num_k * num_coil);
            FLOAT_T *b_r_h = mriNewCpu<FLOAT_T>(num_k * num_coil);
            FLOAT_T *b_i_h = mriNewCpu<FLOAT_T>(num_k * num_coil);
            copyCudaDeviceToHost<FLOAT_T>(b_r_h, b_r, num_k * num_coil);
            copyCudaDeviceToHost<FLOAT_T>(b_i_h, b_i, num_k * num_coil);
            mriDeleteGpu(b_r); mriDeleteGpu(b_i);

            FLOAT_T consistency = MRI_ZERO;
            for(int i = 0; i < num_k * num_coil; i++) {
                consistency += b_r_h[i] * b_r_h[i] + b_i_h[i] * b_i_h[i];
            }
            mriDeleteCpu(b_r_h); mriDeleteCpu(b_i_h);

            // real(r^H*r)
            FLOAT_T residual = MRI_ZERO;
            dotProductGpu(&residual, y_Af_r, y_Af_i, y_Af_r, y_Af_i,
                          num_k * num_coil);

            // Display
            static FLOAT_T consistency_pre = 0;
            FLOAT_T consistency_diff = 0;
            if (consistency_pre == 0) { consistency_diff = consistency;
            } else { consistency_diff = consistency - consistency_pre;
            }
            consistency_pre = consistency;
            msg(MSG_PLAIN, "\n(DC:%.3f, \t%.3f, \t%.3f)",
                consistency, consistency_diff, residual);
            #endif

            #if COMPUTE_REG_FD_COST_FUNC_GPU
            // Compute cost function of regularization term
            // Cf (idata) => bh = Dhori*f => cost_bh = sum(|bh|^2)
            //               bv = Dvert*f => cost_bv = sum(|bv|^2)
            //            => cost = cost_bh + cost_bv
            // ---------------------------------------------------------------

            if (enable_regularization || enable_finite_difference) {
                // bh = Dhori*f => cost_bh = sum(|bh|^2) =====================
                FLOAT_T *bh_r = mriNewGpu<FLOAT_T>(num_i);
                FLOAT_T *bh_i = mriNewGpu<FLOAT_T>(num_i);
                Dhori2dGpu(bh_r, bh_i, idata_r, idata_i,
                           sqrt(num_i), sqrt(num_i));
                FLOAT_T *bh_r_h = mriNewCpu<FLOAT_T>(num_i);
                FLOAT_T *bh_i_h = mriNewCpu<FLOAT_T>(num_i);
                copyCudaDeviceToHost<FLOAT_T>(bh_r_h, bh_r, num_i);
                copyCudaDeviceToHost<FLOAT_T>(bh_i_h, bh_i, num_i);
                mriDeleteGpu(bh_r); mriDeleteGpu(bh_i);
                FLOAT_T cost_bh = MRI_ZERO;
                for(int i = 0; i < num_i; i++) {
                    cost_bh += bh_r_h[i] * bh_r_h[i] + bh_i_h[i] * bh_i_h[i];
                }
                mriDeleteCpu(bh_r_h); mriDeleteCpu(bh_i_h);

                // bv = Dvert*f => cost_bv = sum(|bv|^2) =====================
                FLOAT_T *bv_r = mriNewGpu<FLOAT_T>(num_i);
                FLOAT_T *bv_i = mriNewGpu<FLOAT_T>(num_i);
                Dverti2dGpu(bv_r, bv_i, idata_r, idata_i,
                           sqrt(num_i), sqrt(num_i));
                FLOAT_T *bv_r_h = mriNewCpu<FLOAT_T>(num_i);
                FLOAT_T *bv_i_h = mriNewCpu<FLOAT_T>(num_i);
                copyCudaDeviceToHost<FLOAT_T>(bv_r_h, bv_r, num_i);
                copyCudaDeviceToHost<FLOAT_T>(bv_i_h, bv_i, num_i);
                mriDeleteGpu(bv_r); mriDeleteGpu(bv_i);

                FLOAT_T cost_bv = MRI_ZERO;
                for(int i = 0; i < num_i; i++) {
                    cost_bv += bv_r_h[i] * bv_r_h[i] + bv_i_h[i] * bv_i_h[i];
                }
                mriDeleteCpu(bv_r_h); mriDeleteCpu(bv_i_h);

                // cost = cost_bh + cost_bv ==================================
                FLOAT_T cost_reg = cost_bh + cost_bv;
                msg(MSG_PLAIN, "(CR:%.3f)", cost_reg);
            }
            #endif

            #if COMPUTE_CG_COST_FUNC_GPU
            if (cg_i >= 50 && consistency_diff > 0) break;
            #endif
        } // End of for (cg_i...)

        msg(MSG_PLAIN, "done.\n");

        if (!enable_total_variation) { // No TV.
            break;
        }

    } // end of for (tv_i...)

    // =======================================================================
    //  Copy device data back to host
    // =======================================================================

    mriCopyDeviceToHost<FLOAT_T>(*idata_r_h, idata_r, num_i);
    mriCopyDeviceToHost<FLOAT_T>(*idata_i_h, idata_i, num_i);

    // =======================================================================
    //  Free memory
    // =======================================================================

    mriDeleteCpu(ktraj_h  );
    mriDeleteCpu(kdata_r_h); mriDeleteCpu(kdata_i_h);
    mriDeleteCpu(time_h   );

    mriDeleteGpu(ktraj);   mriDeleteGpu(itraj);
    mriDeleteGpu(kdata_r); mriDeleteGpu(kdata_i);
    mriDeleteGpu(idata_r); mriDeleteGpu(idata_i);
    mriDeleteGpu(sensi_r); mriDeleteGpu(sensi_i);
    mriDeleteGpu(fm);
    mriDeleteGpu(time);

    if (enable_regularization) {
        mriDeleteGpu(c_Ap_d);
        mriDeleteGpu(c_Aj_d);
        mriDeleteGpu(c_Ax_d);
        mriDeleteGpu(c_trans_Ap_d);
        mriDeleteGpu(c_trans_Aj_d);
        mriDeleteGpu(c_trans_Ax_d);
    }

    mriDeleteGpu(Af_r  ); mriDeleteGpu(Af_i  );
    mriDeleteGpu(y_Af_r); mriDeleteGpu(y_Af_i);
    mriDeleteGpu(GtAf_r); mriDeleteGpu(GtAf_i);
    mriDeleteGpu(grad_r); mriDeleteGpu(grad_i);
    mriDeleteGpu(d_r   ); mriDeleteGpu(d_i   );
    mriDeleteGpu(q_r   ); mriDeleteGpu(q_i   );

    if (enable_regularization || enable_finite_difference) {
        mriDeleteGpu(CtCf_r); mriDeleteGpu(CtCf_i);
        mriDeleteGpu(CtCd_r); mriDeleteGpu(CtCd_i);

        mriDeleteGpu(dhori_r ); mriDeleteGpu(dhori_i );
        mriDeleteGpu(dverti_r); mriDeleteGpu(dverti_i);
        mriDeleteGpu(w_r     );
    }

    if (enable_regularization) {
        mriDeleteGpu(Cf_r); mriDeleteGpu(Cf_i);
        mriDeleteGpu(Cd_r); mriDeleteGpu(Cd_i);
    }
    stopMriTimer(getMriTimer()->timer_bruteForceGpu);
    #if DEBUG_CGGPU // Error check
    msg(1, "bruteForceGpu() end");
    #endif
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

