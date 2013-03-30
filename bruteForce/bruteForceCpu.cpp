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

    File Name   [bruteForceCpu.cpp]

    Synopsis    [The main body of the CPU-version CG solver.]

    Description []

    Revision    [0.1; Initial build; Yue Zhuo, BIOE UIUC]
    Revision    [0.1.1; Code cleaning; Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Further optimization, Code cleaning, Adding more comments;
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
#include <tools.h>
#include <structures.h>
#include <ftCpu.h>
#include <parImagingCpu.h>
#include <addCpu.h>
#include <dotProductCpu.h>
#include <smvmCpu.h>
#include <DHWD2dCpu.h>
#include <Dverti2dCpu.h>
#include <Dhori2dCpu.h>
#include <multiplyCpu.h>

#include <bruteForceCpu.h>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Conjugate Gradient solver: CPU kernel.]                     */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*  Note        [This is a single thread CPU kernel.]                        */
/*                                                                           */
/*===========================================================================*/

// Kernel debugging macros
#if DEBUG_KERNEL_MSG
    #define DEBUG_CGCPU  true
#else
    #define DEBUG_CGCPU  false
#endif

#if DEBUG_CGCPU
    // ADJOINT_TEST_CPU will disable all other tests.
    #define ADJOINT_TEST_CPU              false
    #define COMPUTE_CG_COST_FUNC_CPU      true
    #define COMPUTE_REG_FD_COST_FUNC_CPU  true
#else
    #define ADJOINT_TEST_CPU              false
    #define COMPUTE_CG_COST_FUNC_CPU      false
    #define COMPUTE_REG_FD_COST_FUNC_CPU  false
#endif

#define NEW_CPU_REG  true

    void
bruteForceCpu(
    FLOAT_T **idata_r, FLOAT_T **idata_i,
    const FLOAT_T *kdata_r, const FLOAT_T *kdata_i,
    const DataTraj *ktraj, const DataTraj *itraj,
    const FLOAT_T *fm, const FLOAT_T *time, const CooMatrix *c,
    const int cg_num, const int num_k, const int num_i,
    const bool enable_regularization,
    const bool enable_finite_difference, const FLOAT_T fd_penalizer,
    const bool enable_total_variation, const int tv_num,
    const FLOAT_T *sensi_r, const FLOAT_T *sensi_i, const int num_coil)
{
    #if DEBUG_CGCPU
    msg(MSG_PLAIN, "\n");
    msg(1, "bruteForceCpu() begin");
    #endif
    startMriTimer(getMriTimer()->timer_bruteForceCpu);

    #if ADJOINT_TEST_CPU // ==================================================

    FLOAT_T * a_r = mriNewCpu<FLOAT_T>(num_i);
    FLOAT_T * a_i = mriNewCpu<FLOAT_T>(num_i);
    FLOAT_T * b_r = mriNewCpu<FLOAT_T>(num_k);
    FLOAT_T * b_i = mriNewCpu<FLOAT_T>(num_k);
    FLOAT_T * c1_r = mriNewCpu<FLOAT_T>(num_k);
    FLOAT_T * c1_i = mriNewCpu<FLOAT_T>(num_k);
    FLOAT_T * c2_r = mriNewCpu<FLOAT_T>(num_i);
    FLOAT_T * c2_i = mriNewCpu<FLOAT_T>(num_i);

    for (int i = 0; i < num_i; i++) {
        #if 0
        a_r[i] = 1; a_i[i] = 1;
        #else // Produce the data with big variations
        if (i % 10) { a_r[i] = 1; a_i[i] = num_k; }
        else        { a_r[i] = num_i; a_i[i] = 1; }
        #endif
        c2_r[i] = 0;  c2_i[i] = 0;
    }
    for (int i = 0; i < num_k; i++) {
        #if 0
        b_r[i] = 1; b_i[i] = 1;
        #else // Produce the data with big variations
        if (i % 10) { b_r[i] = num_i+1; b_i[i] = 1; }
        else        { b_r[i] = 1; b_i[i] = num_k+2; }
        #endif
        c1_r[i] = 0;  c1_i[i] = 0;
    }

    FLOAT_T o1 = 0, o2 = 0;

    ftCpu(c1_r, c1_i, a_r, a_i, ktraj, itraj, fm, time, num_k, num_i);
    dotProductCpu(&o1, c1_r, c1_i, b_r, b_i, num_k);

    iftCpu(c2_r, c2_i, b_r, b_i, ktraj, itraj, fm, time, num_k, num_i);
    dotProductCpu(&o2, a_r, a_i, c2_r, c2_i, num_i);

    msg(1, "cpu o1: %.8f", o1); msg(1, "cpu o2: %.8f", o2);
    msg(1, "cpu o1/o2: %.8f, o2/o1: %.8f", o1/o2, o2/o1);

    exit(1);
    #endif // End of ADJOINT_TEST_CPU ========================================
    
    // =======================================================================
    //  CG variables
    // =======================================================================

    FLOAT_T gamma            = MRI_ZERO;
    FLOAT_T newinprod        = MRI_ZERO;
    FLOAT_T oldinprod        = MRI_ZERO;
    FLOAT_T alpha            = MRI_ZERO;
    FLOAT_T alpha_d_grad     = MRI_ZERO;
    FLOAT_T alpha_q          = MRI_ZERO;
    FLOAT_T alpha_CtCd       = MRI_ZERO;

    // G*x
    FLOAT_T *Af_r = mriNewCpu<FLOAT_T>(num_coil * num_k);
    FLOAT_T *Af_i = mriNewCpu<FLOAT_T>(num_coil * num_k);
    FLOAT_T *Cf_r = NULL; // C*f
    FLOAT_T *Cf_i = NULL;
	FLOAT_T *Cd_r = NULL; // C*d
	FLOAT_T *Cd_i = NULL;
    if (enable_regularization) {
        Cf_r = mriNewCpu<FLOAT_T>(2*num_i);
        Cf_i = mriNewCpu<FLOAT_T>(2*num_i);
        Cd_r = mriNewCpu<FLOAT_T>(2*num_i);
        Cd_i = mriNewCpu<FLOAT_T>(2*num_i);
    }

    // y-G*x
    FLOAT_T *y_Af_r = mriNewCpu<FLOAT_T>(num_coil * num_k); 
    FLOAT_T *y_Af_i = mriNewCpu<FLOAT_T>(num_coil * num_k);
    // G component of grad
    FLOAT_T *GtAf_r = mriNewCpu<FLOAT_T>(num_i); 
    FLOAT_T *GtAf_i = mriNewCpu<FLOAT_T>(num_i);
    FLOAT_T *CtCf_r = NULL;
    FLOAT_T *CtCf_i = NULL;
    if (enable_regularization || enable_finite_difference) {
        CtCf_r = mriNewCpu<FLOAT_T>(num_i);
        CtCf_i = mriNewCpu<FLOAT_T>(num_i);
    }
    FLOAT_T *grad_r = mriNewCpu<FLOAT_T>(num_i); // grad
    FLOAT_T *grad_i = mriNewCpu<FLOAT_T>(num_i);
    FLOAT_T *d_r    = mriNewCpu<FLOAT_T>(num_i); // dir
    FLOAT_T *d_i    = mriNewCpu<FLOAT_T>(num_i);
    FLOAT_T *q_r    = mriNewCpu<FLOAT_T>(num_coil*num_k); // G component of dir
    FLOAT_T *q_i    = mriNewCpu<FLOAT_T>(num_coil*num_k);
    FLOAT_T *CtCd_r = NULL;
    FLOAT_T *CtCd_i = NULL;
    if (enable_regularization || enable_finite_difference) {
        CtCd_r = mriNewCpu<FLOAT_T>(num_i);
        CtCd_i = mriNewCpu<FLOAT_T>(num_i);
    }

    FLOAT_T * dhori_r  = NULL;
    FLOAT_T * dhori_i  = NULL;
    FLOAT_T * dverti_r = NULL;
    FLOAT_T * dverti_i = NULL;
    FLOAT_T * w_r      = NULL;
    if (enable_finite_difference || enable_regularization) {
        dhori_r  = mriNewCpu<FLOAT_T>(num_i);
        dhori_i  = mriNewCpu<FLOAT_T>(num_i);
        dverti_r = mriNewCpu<FLOAT_T>(num_i);
        dverti_i = mriNewCpu<FLOAT_T>(num_i);
        w_r      = mriNewCpu<FLOAT_T>(num_i);
    }

    // =======================================================================
    //  TV (total variation) iterations
    // =======================================================================

    for (int tv_i = 0; tv_i < tv_num; tv_i++) {
        if (enable_finite_difference || enable_regularization) {
            // Calculating real part W
            if (tv_i == 0) {
                for(int i = 0; i < num_i; i++) w_r[i] = MRI_ONE;
            }
        }
        if (enable_total_variation) {
            msg(1, "CPU: TV: %d", tv_i);
            if (enable_finite_difference || enable_regularization) {
                // Calculating real part W
                if (tv_i != 0) {
                    Dhori2dCpu(dhori_r, dhori_i, *idata_r, *idata_i,
                               sqrt(num_i), sqrt(num_i));
                    Dverti2dCpu(dverti_r, dverti_i, *idata_r, *idata_i,
                                sqrt(num_i), sqrt(num_i));
                    multiplyCpu(w_r, dhori_r, dhori_i, dverti_r, dverti_i,
                                num_i);
                }
            }
        }

        // CG initialization =================================================

        msg(MSG_PLAIN, "  CPU: CG: 0.");

        // Calculation of the gradient
        // Af = G * x;
        // Cf = C * x;
        // -------------------------------------------------------------------

        parallelFtCpu(Af_r, Af_i, *idata_r, *idata_i, ktraj, itraj, fm, time,
                      num_k, num_i, sensi_r, sensi_i, num_coil);

        //if (enable_regularization) {
        //    smvmCpu(Cf_r, Cf_i, *idata_r, *idata_i, c);
        //}

        // CG iterations =====================================================

        for (int cg_i = 1; cg_i < cg_num; cg_i++) {
            msg(MSG_PLAIN, "%d.", cg_i);

            // Calculation of the gradient
            // grad = G' * (W * (yi-Af) - nder1) - C' * Cf;
            // ---------------------------------------------------------------

            // Update
            // r = r + alpha*q;
            addCpu(y_Af_r, y_Af_i, kdata_r, kdata_i, Af_r, Af_i, MRI_NEG_ONE,
                   num_k * num_coil);

            parallelIftCpu(GtAf_r, GtAf_i, y_Af_r, y_Af_i, ktraj, itraj, fm,
                           time, num_k, num_i, sensi_r, sensi_i, num_coil);

            if (enable_regularization || enable_finite_difference) {
                if (enable_regularization) {
                    smvmCpu(Cf_r, Cf_i, *idata_r, *idata_i, c);
					for(int i=0;i<num_i;i++)
					{
					   Cf_r[i] *= w_r[i];
					   Cf_i[i] *= w_r[i];

					   Cf_r[i+num_i] *= w_r[i];
					   Cf_i[i+num_i] *= w_r[i];
					}
                    smvmTransCpu(CtCf_r, CtCf_i, Cf_r, Cf_i, c);
					for(int i=0;i<num_i;i++)
					{
					   CtCf_r[i] *= fd_penalizer;
					   CtCf_i[i] *= fd_penalizer;
					}
                } else {
                    DHWD2dCpu(CtCf_r, CtCf_i, *idata_r, *idata_i, w_r,
                              sqrt(num_i), sqrt(num_i), fd_penalizer);
                }
                addCpu(grad_r, grad_i, GtAf_r, GtAf_i, CtCf_r, CtCf_i,
                       MRI_NEG_ONE, num_i);
            } else {
                memcpy(grad_r, GtAf_r, num_i * sizeof(FLOAT_T));
                memcpy(grad_i, GtAf_i, num_i * sizeof(FLOAT_T));
            }

            dotProductCpu(&newinprod, grad_r, grad_i, grad_r, grad_i, num_i);

            if (cg_i == 1) {
                memcpy(d_r, grad_r, num_i * sizeof(FLOAT_T));
                memcpy(d_i, grad_i, num_i * sizeof(FLOAT_T));
            } else {
                if (oldinprod == MRI_ZERO) {
                    gamma = MRI_ZERO;
                } else {
                    gamma = newinprod / oldinprod;
                }
                addCpu(d_r, d_i, grad_r, grad_i, d_r, d_i, gamma, num_i);
            }
            oldinprod = newinprod;

            parallelFtCpu(q_r, q_i, d_r, d_i, ktraj, itraj, fm, time,
                          num_k, num_i, sensi_r, sensi_i, num_coil);

            if (enable_regularization) {

				smvmCpu(Cd_r, Cd_i, d_r, d_i, c);

				for(int i=0;i<num_i;i++)
				{
				   Cd_r[i] *= w_r[i];
				   Cd_i[i] *= w_r[i];

				   Cd_r[i+num_i] *= w_r[i];
				   Cd_i[i+num_i] *= w_r[i];
				}

				smvmTransCpu(CtCd_r, CtCd_i, Cd_r, Cd_i, c);

			    for(int i=0;i<num_i;i++)
				{
				   CtCd_r[i] *= fd_penalizer;
				   CtCd_i[i] *= fd_penalizer;
				}
                    

            } else if (enable_finite_difference) {
                DHWD2dCpu(CtCd_r, CtCd_i, d_r, d_i, w_r,
                          sqrt(num_i), sqrt(num_i), fd_penalizer);
            }

            // Calculation of the alpha
            // if (enable_regularization)
            //     alpha = real(dir' * grad) / real(q'*(W*q) + CtCd'*CtCd);
            // else if (enable_finite_difference)
            //     alpha = real(dir' * grad) / real(q'*q + dt*CtCd);
            // endif
            // ---------------------------------------------------------------

            dotProductCpu(&alpha_d_grad, d_r, d_i, grad_r, grad_i, num_i);
            dotProductCpu(&alpha_q, q_r, q_i, q_r, q_i, num_k * num_coil);
            if (enable_regularization) {
				//Jiading GAI
                dotProductCpu(&alpha_CtCd, CtCd_r, CtCd_i, d_r, d_i, num_i);
                //dotProductCpu(&alpha_CtCd, CtCd_r, CtCd_i, CtCd_r, CtCd_i,
                //              num_i);
            } else if (enable_finite_difference) {
                dotProductCpu(&alpha_CtCd, CtCd_r, CtCd_i, d_r, d_i, num_i);
            } else {
                alpha_CtCd = 0;
            }
            alpha = alpha_d_grad / (alpha_q + alpha_CtCd);

            // Update
            // Af = Af + q;
            // Cf = Cf + CtCd;
            // x = x + d;
            //----------------------------------------------------------------

            addCpu(Af_r, Af_i, Af_r, Af_i, q_r, q_i, alpha, num_k * num_coil);

			//Jiading GAI
            //if (enable_regularization) {
            //    addCpu(Cf_r, Cf_i, Cf_r, Cf_i, CtCd_r, CtCd_i, alpha, num_i);
            //}

            addCpu(*idata_r, *idata_i, *idata_r, *idata_i, d_r, d_i,
                   alpha, num_i);

            #if COMPUTE_CG_COST_FUNC_CPU
            // Compute cost function of data consistency term
            // Ff (idata) => b = (Ff - y) (kdata), consistency = sum(|b|^2)
            // ---------------------------------------------------------------

            FLOAT_T *b_r = mriNewCpu<FLOAT_T>(num_k * num_coil);
            FLOAT_T *b_i = mriNewCpu<FLOAT_T>(num_k * num_coil);
            #if 1
            parallelFtCpu(b_r, b_i, *idata_r, *idata_i, ktraj, itraj, fm, time,
                  num_k, num_i, sensi_r, sensi_i, num_coil);
            #else
            ftCpu(b_r, b_i, *idata_r, *idata_i, ktraj, itraj, fm, time,
                  num_k, num_i);
            #endif
            addCpu(b_r, b_i, b_r, b_i, kdata_r, kdata_i, -1, num_k * num_coil);
            FLOAT_T consistency = MRI_ZERO;
            for(int i = 0; i < num_k * num_coil; i++) {
                consistency += b_r[i] * b_r[i] + b_i[i] * b_i[i];
            }
            mriDeleteCpu(b_r); mriDeleteCpu(b_i);

            // real(r^H*r)
            FLOAT_T residual = MRI_ZERO;
            dotProductCpu(&residual, y_Af_r, y_Af_i, y_Af_r, y_Af_i,
                          num_k * num_coil);

            // Display
            static FLOAT_T consistency_pre = 0;
            FLOAT_T diff = 0;
            if (consistency_pre == 0) { diff = consistency;
            } else { diff = consistency - consistency_pre; }
            consistency_pre = consistency;
            msg(MSG_PLAIN, "\n(DC:%.3f, \t%.3f, \t%.3f)", consistency, diff, residual);
            #endif

            #if COMPUTE_REG_FD_COST_FUNC_CPU
            // Compute cost function of regularization term
            // Cf (idata) => bh = Dhori*f => cost_bh = sum(|bh|^2)
            //               bv = Dvert*f => cost_bv = sum(|bv|^2)
            //            => cost = cost_bh + cost_bv
            // ---------------------------------------------------------------

            if (enable_regularization || enable_finite_difference) {
                FLOAT_T *bh_r = mriNewCpu<FLOAT_T>(num_i);
                FLOAT_T *bh_i = mriNewCpu<FLOAT_T>(num_i);
                Dhori2dCpu(bh_r, bh_i, *idata_r, *idata_i,
                           sqrt(num_i), sqrt(num_i));
                FLOAT_T cost_bh = MRI_ZERO;
                for(int i = 0; i < num_i; i++) {
                    cost_bh += bh_r[i] * bh_r[i] + bh_i[i] * bh_i[i];
                }
                mriDeleteCpu<FLOAT_T>(bh_r);
				mriDeleteCpu<FLOAT_T>(bh_i);

                FLOAT_T *bv_r = mriNewCpu<FLOAT_T>(num_i);
                FLOAT_T *bv_i = mriNewCpu<FLOAT_T>(num_i);
                Dverti2dCpu(bv_r, bv_i, *idata_r, *idata_i,
                           sqrt(num_i), sqrt(num_i));
                FLOAT_T cost_bv = MRI_ZERO;
                for(int i = 0; i < num_i; i++) {
                    cost_bv += bv_r[i] * bv_r[i] + bv_i[i] * bv_i[i];
                }
                mriDeleteCpu<FLOAT_T>(bv_r);
				mriDeleteCpu<FLOAT_T>(bv_i);
                
				FLOAT_T cost_reg = cost_bh + cost_bv;
                msg(MSG_PLAIN, "(CR:%.3f)", cost_reg);
            }
            #endif

        } // end of for (cg_i...)

        msg(MSG_PLAIN, "done.\n");

        // No TV.
        if (!enable_total_variation) break;

    } // end of for (tv_i...)

    // =======================================================================
    //  Free memory
    // =======================================================================

    mriDeleteCpu(Af_r  ); mriDeleteCpu(Af_i  );
    mriDeleteCpu(y_Af_r); mriDeleteCpu(y_Af_i);
    mriDeleteCpu(GtAf_r); mriDeleteCpu(GtAf_i);
    mriDeleteCpu(grad_r); mriDeleteCpu(grad_i);
    mriDeleteCpu(d_r   ); mriDeleteCpu(d_i   );
    mriDeleteCpu(q_r   ); mriDeleteCpu(q_i   );

    if (enable_regularization || enable_finite_difference) {
        mriDeleteCpu(CtCf_r); mriDeleteCpu(CtCf_i);
        mriDeleteCpu(CtCd_r); mriDeleteCpu(CtCd_i);

		mriDeleteCpu(dhori_r ); mriDeleteCpu(dhori_i );
        mriDeleteCpu(dverti_r); mriDeleteCpu(dverti_i);
        mriDeleteCpu(w_r     );
    }

    if (enable_regularization) {
        mriDeleteCpu(Cf_r);
        mriDeleteCpu(Cf_i);
		mriDeleteCpu(Cd_r);
		mriDeleteCpu(Cd_i);
    }

    stopMriTimer(getMriTimer()->timer_bruteForceCpu);
    #if DEBUG_CGCPU
    msg(MSG_PLAIN, "\n");
    msg(1, "bruteForceCpu() end");
    #endif
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}
