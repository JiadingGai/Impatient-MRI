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

    File Name   [bruteForceMgpu.cu]

    Synopsis    []

    Description []

    Revision    [0.1; Initial build; Yue Zhuo, BIOE UIUC]
    Revision    [0.1.1; Code cleaning; Xiao-Long Wu, ECE UIUC]
    Date        [03/23/2010]

*****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// CUDA libraries
#include <cutil.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <multithreading.h>

// Project header files
#include <xcpplib_process.h>
#include <xcpplib_typesGpu.cuh>
#include <tools.h>
#include <structures.h>
#include <gpuPrototypes.cuh>

#include <bruteForceMgpu.cuh>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

    CUT_THREADPROC 
bruteForceMgpu(CgSolverData *sliceData)
{
    // Set up device for current slices
    cutilSafeCall(cudaSetDevice(sliceData[0].gpu_id));

    int numLoops = sliceData[0].num_slices_GPU;

    for (int s = 0; s < numLoops; s++) {
        msg(1, "GPU %d: performing slice: %d\n",
            sliceData[0].gpu_id, sliceData[s].slice_id);
        FLOAT_T *idata_r = sliceData[s].idata_r;
        FLOAT_T *idata_i = sliceData[s].idata_i;
        FLOAT_T *kdata_r = sliceData[s].kdata_r;
        FLOAT_T *kdata_i = sliceData[s].kdata_i;
        FLOAT_T *sensi_r = sliceData[s].sensi_r;
        FLOAT_T *sensi_i = sliceData[s].sensi_i;
        int num_coil = sliceData[s].num_coil;
        DataTraj *ktraj = sliceData[s].ktraj;
        DataTraj *itraj = sliceData[s].itraj;
        FLOAT_T *fm = sliceData[s].fm;
        FLOAT_T *time = sliceData[s].time;
        CooMatrix *c = sliceData[s].c;
        int cg_num = sliceData[s].cg_num;
        int num_k = sliceData[s].num_k;
        int num_i = sliceData[s].num_i;
        const bool enable_regularization = sliceData[s].enable_regularization;
        const bool enable_finite_difference =
                   sliceData[s].enable_finite_difference;
        const FLOAT_T fd_penalizer = sliceData[s].fd_penalizer;
        const bool enable_total_variation = sliceData[s].enable_total_variation;
        int tv_num = sliceData[s].tv_num;
        const bool enable_tv_update = sliceData[s].enable_tv_update;
        const string output_file_gpu_r = sliceData[s].output_file_gpu_r;
        const string output_file_gpu_i = sliceData[s].output_file_gpu_i;
        const int num_slices = sliceData[s].num_slices_total;

        bruteForceGpu(&idata_r, &idata_i, kdata_r, kdata_i, ktraj, itraj,
              fm, time, c, cg_num, num_k, num_i,
              enable_regularization,
              enable_finite_difference, fd_penalizer,
              enable_total_variation, tv_num,
              sensi_r, sensi_i, num_coil,
              enable_tv_update,
              output_file_gpu_r, output_file_gpu_i, num_slices);
    }
    CUT_THREADEND;
}

// Dummy multiGPU call to cleanup memory (AC cluster recommandations)
    CUT_THREADPROC 
mgpuDummy(int GPU_Idx)
{
    cutilSafeCall(cudaSetDevice(GPU_Idx));
    FLOAT_T *dummy_A = mriNewGpu<FLOAT_T>(10);
    dim3 dummy_block(10);
    dim3 dummy_grid(1);
    addGpuKernel <<< dummy_grid, dummy_block >>> 
        (dummy_A, dummy_A, dummy_A, dummy_A, dummy_A, dummy_A, 1, 10);
    mriDeleteGpu(dummy_A);
    CUT_THREADEND;
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

