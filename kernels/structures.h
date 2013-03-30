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

    File Name   [structure.h]

    Synopsis    [This file defines the common data structures used in the
        whole application.]

    Description []

    Revision    [0.1; Initial build; Yue Zhuo, BIOE UIUC]
    Revision    [0.1.1; Change all data structures to C++ forms, Code cleaning;
                 Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Further optimization, Code cleaning, Adding more comments;
                 Xiao-Long Wu, ECE UIUC, Jiading Gai, Beckman Institute]
    Date        [10/27/2010]

 *****************************************************************************/

#ifndef STRUCTURES_H
#define STRUCTURES_H

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>

// CUDA libraries
#include <cutil.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>

// MRI project related files
#include <xcpplib_process.h>
#include <xcpplib_typesGpu.cuh>
#include <xcpplib_others.h>
#include <tools.h>
#include <mmio.h>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Namespace used                                                           */
/*---------------------------------------------------------------------------*/

using namespace xcpplib;

/*---------------------------------------------------------------------------*/
/*  Function prototypes                                                      */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*  Project related macros                                                   */
/*---------------------------------------------------------------------------*/

// Numeric constants according to the precision type.
#ifdef ENABLE_DOUBLE_PRECISION
    #define MRI_PI               3.1415926535897932384626433832795029
    #define MRI_NN               64
    #define MRI_DELTAZ           0.003
    #define MRI_ZERO             0.0
    #define MRI_ONE              1.0
    #define MRI_NEG_ONE         -1.0
    #define MRI_POINT_FIVE       0.5
    #define MRI_SMOOTH_FACTOR    0.0000001
#else
    #define MRI_PI               3.1415926535897932384626433832795029f
    #define MRI_NN               64
    #define MRI_DELTAZ           0.003f
    #define MRI_ZERO             0.0f
    #define MRI_ONE              1.0f
    #define MRI_NEG_ONE         -1.0f
    #define MRI_POINT_FIVE       0.5f
    #define MRI_SMOOTH_FACTOR    0.000001f
#endif

// For 2D image reconstruction, saving the storage space for z dimension
// when double-precision is enabled.
#if ENABLE_DOUBLE_PRECISION
    #define DATATRAJ_NO_Z_DIM           true
#else // Single precision
    #define DATATRAJ_NO_Z_DIM           false
#endif

/*---------------------------------------------------------------------------*/
/*  Performance evaluation data structures                                   */
/*---------------------------------------------------------------------------*/

class mriFlop
{
public:
    // Overall
    unsigned int flop_Cpu            ;
    unsigned int flop_Gpu            ;

    // CPU kernels
    unsigned int flop_addCpu         ;
    unsigned int flop_Dhori2dCpu     ;
    unsigned int flop_DHWD2dCpu      ;
    unsigned int flop_dotProductCpu  ;
    unsigned int flop_Dverti2dCpu    ;
    unsigned int flop_ftCpu          ;
    unsigned int flop_iftCpu         ;
    unsigned int flop_multiplyCpu    ;
    unsigned int flop_parallelFtCpu  ;
    unsigned int flop_parallelIftCpu ;
    unsigned int flop_pointMultCpu   ;
    unsigned int flop_smvmCpu        ;

    // GPU kernels
    unsigned int flop_addGpu         ;
    unsigned int flop_Dhori2dGpu     ;
    unsigned int flop_DHWD2dGpu      ;
    unsigned int flop_dotProductGpu  ;
    unsigned int flop_Dverti2dGpu    ;
    unsigned int flop_ftGpu          ;
    unsigned int flop_iftGpu         ;
    unsigned int flop_multiplyGpu    ;
    unsigned int flop_parallelFtGpu  ;
    unsigned int flop_parallelIftGpu ;
    unsigned int flop_pointMultGpu   ;
    unsigned int flop_smvmGpu        ;

public:
    // Constructors
    mriFlop(void) {
        // Overall
        flop_Cpu            = 0;
        flop_Gpu            = 0;

        // CPU kernels
        flop_addCpu         = 0;
        flop_Dhori2dCpu     = 0;
        flop_DHWD2dCpu      = 0;
        flop_dotProductCpu  = 0;
        flop_Dverti2dCpu    = 0;
        flop_ftCpu          = 0;
        flop_iftCpu         = 0;
        flop_multiplyCpu    = 0;
        flop_parallelFtCpu  = 0;
        flop_parallelIftCpu = 0;
        flop_pointMultCpu   = 0;
        flop_smvmCpu        = 0;

        // GPU kernels
        flop_addGpu         = 0;
        flop_Dhori2dGpu     = 0;
        flop_DHWD2dGpu      = 0;
        flop_dotProductGpu  = 0;
        flop_Dverti2dGpu    = 0;
        flop_ftGpu          = 0;
        flop_iftGpu         = 0;
        flop_multiplyGpu    = 0;
        flop_parallelFtGpu  = 0;
        flop_parallelIftGpu = 0;
        flop_pointMultGpu   = 0;
        flop_smvmGpu        = 0;
    }
}; // End of class mriFlop

    mriFlop *
getMriFlop(void);

class mriTimer
{
private:
    // This is enabled in initTimer() which must be activated before all other
    // timer member functions.
    bool if_create_timer;

public:
    // Overall
    unsigned int timer_Cpu            ;
    unsigned int timer_Gpu            ;

    // CPU kernels
    unsigned int timer_addCpu         ;
    unsigned int timer_bruteForceCpu          ;
    unsigned int timer_Dhori2dCpu     ;
    unsigned int timer_DHWD2dCpu      ;
    unsigned int timer_dotProductCpu  ;
    unsigned int timer_Dverti2dCpu    ;
    unsigned int timer_ftCpu          ;
    unsigned int timer_iftCpu         ;
    unsigned int timer_multiplyCpu    ;
    unsigned int timer_parallelFtCpu  ;
    unsigned int timer_parallelIftCpu ;
    unsigned int timer_pointMultCpu   ;
    unsigned int timer_smvmCpu        ;

    // GPU kernels
    unsigned int timer_addGpu         ;
    unsigned int timer_bruteForceGpu          ;
    unsigned int timer_Dhori2dGpu     ;
    unsigned int timer_DHWD2dGpu      ;
    unsigned int timer_dotProductGpu  ;
    unsigned int timer_Dverti2dGpu    ;
    unsigned int timer_ftGpu          ;
    unsigned int timer_iftGpu         ;
    unsigned int timer_multiplyGpu    ;
    unsigned int timer_parallelFtGpu  ;
    unsigned int timer_parallelIftGpu ;
    unsigned int timer_pointMultGpu   ;
    unsigned int timer_smvmGpu        ;

    // System
    unsigned int timer_memAllocateCpu ;
    unsigned int timer_memAllocateGpu ;
    unsigned int timer_host2Device    ;
    unsigned int timer_device2Host    ;
    unsigned int timer_readFile       ;
    unsigned int timer_writeFile      ;

public:
    // Constructors
    mriTimer(void) {
        if_create_timer = false;

        // Overall
        timer_Cpu            = 0;
        timer_Gpu            = 0;

        // CPU kernels
        timer_addCpu         = 0;
        timer_bruteForceCpu          = 0;
        timer_Dhori2dCpu     = 0;
        timer_DHWD2dCpu      = 0;
        timer_dotProductCpu  = 0;
        timer_Dverti2dCpu    = 0;
        timer_ftCpu          = 0;
        timer_iftCpu         = 0;
        timer_multiplyCpu    = 0;
        timer_parallelFtCpu  = 0;
        timer_parallelIftCpu = 0;
        timer_pointMultCpu   = 0;
        timer_smvmCpu        = 0;

        // GPU kernels
        timer_addGpu         = 0;
        timer_bruteForceGpu          = 0;
        timer_Dhori2dGpu     = 0;
        timer_DHWD2dGpu      = 0;
        timer_dotProductGpu  = 0;
        timer_Dverti2dGpu    = 0;
        timer_ftGpu          = 0;
        timer_iftGpu         = 0;
        timer_multiplyGpu    = 0;
        timer_parallelFtGpu  = 0;
        timer_parallelIftGpu = 0;
        timer_pointMultGpu   = 0;
        timer_smvmGpu        = 0;

        // System
        timer_memAllocateCpu = 0;
        timer_memAllocateGpu = 0;
        timer_host2Device    = 0;
        timer_device2Host    = 0;
        timer_readFile       = 0;
        timer_writeFile      = 0;
    }

public:
        void
    initTimer(void) {
        if_create_timer = true;

        // Overall
        cutilCheckError(cutCreateTimer(&timer_Cpu            ));
        cutilCheckError(cutCreateTimer(&timer_Gpu            ));

        // CPU kernels
        cutilCheckError(cutCreateTimer(&timer_addCpu         ));
        cutilCheckError(cutCreateTimer(&timer_bruteForceCpu          ));
        cutilCheckError(cutCreateTimer(&timer_Dhori2dCpu     ));
        cutilCheckError(cutCreateTimer(&timer_DHWD2dCpu      ));
        cutilCheckError(cutCreateTimer(&timer_dotProductCpu  ));
        cutilCheckError(cutCreateTimer(&timer_Dverti2dCpu    ));
        cutilCheckError(cutCreateTimer(&timer_ftCpu          ));
        cutilCheckError(cutCreateTimer(&timer_iftCpu         ));
        cutilCheckError(cutCreateTimer(&timer_multiplyCpu    ));
        cutilCheckError(cutCreateTimer(&timer_parallelFtCpu  ));
        cutilCheckError(cutCreateTimer(&timer_parallelIftCpu ));
        cutilCheckError(cutCreateTimer(&timer_pointMultCpu   ));
        cutilCheckError(cutCreateTimer(&timer_smvmCpu        ));

        // GPU kernels
        cutilCheckError(cutCreateTimer(&timer_addGpu         ));
        cutilCheckError(cutCreateTimer(&timer_bruteForceGpu          ));
        cutilCheckError(cutCreateTimer(&timer_Dhori2dGpu     ));
        cutilCheckError(cutCreateTimer(&timer_DHWD2dGpu      ));
        cutilCheckError(cutCreateTimer(&timer_dotProductGpu  ));
        cutilCheckError(cutCreateTimer(&timer_Dverti2dGpu    ));
        cutilCheckError(cutCreateTimer(&timer_ftGpu          ));
        cutilCheckError(cutCreateTimer(&timer_iftGpu         ));
        cutilCheckError(cutCreateTimer(&timer_multiplyGpu    ));
        cutilCheckError(cutCreateTimer(&timer_parallelFtGpu  ));
        cutilCheckError(cutCreateTimer(&timer_parallelIftGpu ));
        cutilCheckError(cutCreateTimer(&timer_pointMultGpu   ));
        cutilCheckError(cutCreateTimer(&timer_smvmGpu        ));

        // System
        cutilCheckError(cutCreateTimer(&timer_memAllocateCpu ));
        cutilCheckError(cutCreateTimer(&timer_memAllocateGpu ));
        cutilCheckError(cutCreateTimer(&timer_host2Device    ));
        cutilCheckError(cutCreateTimer(&timer_device2Host    ));
        cutilCheckError(cutCreateTimer(&timer_readFile       ));
        cutilCheckError(cutCreateTimer(&timer_writeFile      ));
    }

        void
    deleteTimer(void) {
		//Jiading GAI
        //if_create_timer = false;

        // Overall
        cutilCheckError(cutDeleteTimer(timer_Cpu            ));
        cutilCheckError(cutDeleteTimer(timer_Gpu            ));

        // CPU kernels
        cutilCheckError(cutDeleteTimer(timer_addCpu         ));
        cutilCheckError(cutDeleteTimer(timer_bruteForceCpu          ));
        cutilCheckError(cutDeleteTimer(timer_Dhori2dCpu     ));
        cutilCheckError(cutDeleteTimer(timer_DHWD2dCpu      ));
        cutilCheckError(cutDeleteTimer(timer_dotProductCpu  ));
        cutilCheckError(cutDeleteTimer(timer_Dverti2dCpu    ));
        cutilCheckError(cutDeleteTimer(timer_ftCpu          ));
        cutilCheckError(cutDeleteTimer(timer_iftCpu         ));
        cutilCheckError(cutDeleteTimer(timer_multiplyCpu    ));
        cutilCheckError(cutDeleteTimer(timer_parallelFtCpu  ));
        cutilCheckError(cutDeleteTimer(timer_parallelIftCpu ));
        cutilCheckError(cutDeleteTimer(timer_pointMultCpu   ));
        cutilCheckError(cutDeleteTimer(timer_smvmCpu        ));

        // GPU kernels
        cutilCheckError(cutDeleteTimer(timer_addGpu         ));
        cutilCheckError(cutDeleteTimer(timer_bruteForceGpu          ));
        cutilCheckError(cutDeleteTimer(timer_Dhori2dGpu     ));
        cutilCheckError(cutDeleteTimer(timer_DHWD2dGpu      ));
        cutilCheckError(cutDeleteTimer(timer_dotProductGpu  ));
        cutilCheckError(cutDeleteTimer(timer_Dverti2dGpu    ));
        cutilCheckError(cutDeleteTimer(timer_ftGpu          ));
        cutilCheckError(cutDeleteTimer(timer_iftGpu         ));
        cutilCheckError(cutDeleteTimer(timer_multiplyGpu    ));
        cutilCheckError(cutDeleteTimer(timer_parallelFtGpu  ));
        cutilCheckError(cutDeleteTimer(timer_parallelIftGpu ));
        cutilCheckError(cutDeleteTimer(timer_pointMultGpu   ));
        cutilCheckError(cutDeleteTimer(timer_smvmGpu        ));

        // System
        cutilCheckError(cutDeleteTimer(timer_memAllocateCpu ));
        cutilCheckError(cutDeleteTimer(timer_memAllocateGpu ));
        cutilCheckError(cutDeleteTimer(timer_host2Device    ));
        cutilCheckError(cutDeleteTimer(timer_device2Host    ));
        cutilCheckError(cutDeleteTimer(timer_readFile       ));
        cutilCheckError(cutDeleteTimer(timer_writeFile      ));
    }

    // FIXME: Need to check if one timer is started multiple times before
    //        it's stopped.
        void
    startTimer(unsigned int &timer) {
        makeSure(if_create_timer,
            "initTimer() must be activated before all timer member functions.");
        cutilCheckError(cutStartTimer(timer));
    }

        void
    stopTimer(unsigned int &timer) {
        makeSure(if_create_timer,
            "initTimer() must be activated before all timer member functions.");

        // All kernels will be finished after this statement.
        cudaThreadSynchronize();
        cutilCheckError(cutStopTimer(timer));
    }

        FLOAT_T
    getTime(unsigned int &timer) {
        makeSure(if_create_timer,
            "initTimer() must be activated before all timer member functions.");
        return cutGetTimerValue(timer);
    }

        void
    printTimer(void) {
        msg(MSG_PLAIN, "\n");
        msg(1, "Input/Output data:");
        msg(2, "Load input from files: %f (ms)", getTime(timer_readFile));
        msg(2, "Export results to files: %f (ms)", getTime(timer_writeFile));

        msg(MSG_PLAIN, "\n");
        msg(1, "Memory allocation:");
        if (getTime(timer_bruteForceCpu) > 0) {
        msg(2, "CPU:     %f (ms)", getTime(timer_memAllocateCpu)); }
        msg(2, "GPU:     %f (ms)", getTime(timer_memAllocateGpu));
        if (getTime(timer_bruteForceCpu) > 0) {
        msg(2, "Speedup: %f x", getTime(timer_memAllocateCpu) /
                                getTime(timer_memAllocateGpu)); }

        msg(MSG_PLAIN, "\n");
        msg(1, "Copy from host to device:");
        msg(2, "GPU:     %f (ms)", cutGetTimerValue(timer_host2Device));
        msg(1, "Copy from device to host:");
        msg(2, "GPU:     %f (ms)", getTime(timer_device2Host));

        msg(MSG_PLAIN, "\n");
        msg(1, "Forward operator (Fourier transform):");
        if (getTime(timer_ftCpu) > 0) {
        msg(2, "CPU:     %f (ms)", getTime(timer_ftCpu)); }
        msg(2, "GPU:     %f (ms)", getTime(timer_ftGpu));
        if (getTime(timer_ftCpu) > 0) {
        msg(2, "Speedup: %f x", getTime(timer_ftCpu) / getTime(timer_ftGpu));}
        #if COMPUTE_FLOPS // FIXME: Not finished yet.
        if (getTime(timer_ftCpu) > 0) {
        msg(2, "CPU GFLOP/S:\t%3.3f (total %.0f FLOPs)",
            (getMriFlop()->flop_ftCpu / (getTime(timer_ftCpu) / 1000)) /
            1000000000, getMriFlop()->flop_ftCpu); }
        msg(2, "GPU GFLOP/S:\t%3.3f (total %.0f FLOPs)",
            (getMriFlop()->flop_ftGpu / (getTime(timer_ftGpu) / 1000)) /
            1000000000, getMriFlop()->flop_ftGpu);
        #endif

        msg(MSG_PLAIN, "\n");
        msg(1, "Parallel imaging (Fourier transform):");
        if (getTime(timer_parallelFtCpu) > 0) {
        msg(2, "CPU:     %f (ms)", getTime(timer_parallelFtCpu)); }
        msg(2, "GPU:     %f (ms)", getTime(timer_parallelFtGpu));
        if (getTime(timer_parallelFtCpu) > 0) {
        msg(2, "Speedup: %f x", getTime(timer_parallelFtCpu) /
                                getTime(timer_parallelFtGpu)); }

        msg(MSG_PLAIN, "\n");
        msg(1, "Backward operator (inverse Fourier transform):");
        if (getTime(timer_iftCpu) > 0) {
        msg(2, "CPU:     %f (ms)", getTime(timer_iftCpu)); }
        msg(2, "GPU:     %f (ms)", getTime(timer_iftGpu));
        if (getTime(timer_iftCpu) > 0) {
        msg(2, "Speedup: %f x", getTime(timer_iftCpu)/getTime(timer_iftGpu));}
        #if COMPUTE_FLOPS
        if (getTime(timer_iftCpu) > 0) {
        msg(2, "CPU GFLOP/S:\t%3.3f (total %.0f FLOPs)",
            (getMriFlop()->flop_iftCpu / (getTime(timer_iftCpu) / 1000)) /
            1000000000, getMriFlop()->flop_iftCpu); }
        msg(2, "GPU GFLOP/S:\t%3.3f (total %.0f FLOPs)",
            (getMriFlop()->flop_iftGpu / (getTime(timer_iftGpu) / 1000)) /
            1000000000, getMriFlop()->flop_iftGpu);
        #endif

        msg(MSG_PLAIN, "\n");
        msg(1, "Parallel imaging (inverse Fourier transform):");
        if (getTime(timer_parallelIftCpu) > 0) {
        msg(2, "CPU:     %f (ms)", getTime(timer_parallelIftCpu)); }
        msg(2, "GPU:     %f (ms)", getTime(timer_parallelIftGpu));
        if (getTime(timer_parallelIftCpu) > 0) {
        msg(2, "Speedup: %f x", getTime(timer_parallelIftCpu) /
                                getTime(timer_parallelIftGpu)); }

        msg(MSG_PLAIN, "\n");
        msg(1, "Vector addition:");
        if (getTime(timer_addCpu) > 0) {
        msg(2, "CPU:     %f (ms)", getTime(timer_addCpu)); }
        msg(2, "GPU:     %f (ms)", getTime(timer_addGpu));
        if (getTime(timer_addCpu) > 0) {
        msg(2, "Speedup: %f x", getTime(timer_addCpu)/getTime(timer_addGpu));}
        #if COMPUTE_FLOPS
        if (getTime(timer_addCpu) > 0) {
        msg(2, "CPU GFLOP/S: %3.3f (total %.0f FLOPs)",
            (getMriFlop()->flop_addCpu / (getTime(timer_addCpu) / 1000)) /
            1000000000, getMriFlop()->flop_addCpu); }
        msg(2, "GPU GFLOP/S: %3.3f (total %.0f FLOPs)",
            (getMriFlop()->flop_addGpu / (getTime(timer_addGpu) / 1000)) /
            1000000000, getMriFlop()->flop_addGpu);
        #endif

        msg(MSG_PLAIN, "\n");
        msg(1, "Dot product:");
        if (getTime(timer_dotProductCpu) > 0) {
        msg(2, "CPU:     %f (ms)", getTime(timer_dotProductCpu)); }
        msg(2, "GPU:     %f (ms)", getTime(timer_dotProductGpu));
        if (getTime(timer_dotProductCpu) > 0) {
        msg(2, "Speedup: %f x", getTime(timer_dotProductCpu) /
                                getTime(timer_dotProductGpu)); }

        #if COMPUTE_FLOPS
        if (getTime(timer_dotProductCpu) > 0) {
        msg(2, "CPU GFLOP/S: %3.3f (total %.0f FLOPs)",
            (getMriFlop()->flop_dotProductCpu /
            (getTime(timer_dotProductCpu) / 1000)) / 1000000000,
            getMriFlop()->flop_dotProductCpu); }
        msg(2, "GPU GFLOP/S: %3.3f (total %.0f FLOPs)",
            (getMriFlop()->flop_dotProductGpu /
            (getTime(timer_dotProductGpu) / 1000)) / 1000000000,
            getMriFlop()->flop_dotProductGpu);
        #endif

        msg(MSG_PLAIN, "\n");
        msg(1, "Point Multiply:");
        if (getTime(timer_pointMultCpu) > 0) {
        msg(2, "CPU:     %f (ms)", getTime(timer_pointMultCpu)); }
        msg(2, "GPU:     %f (ms)", getTime(timer_pointMultGpu));
        if (getTime(timer_pointMultCpu) > 0) {
        msg(2, "Speedup: %f x", getTime(timer_pointMultCpu) /
                                getTime(timer_pointMultGpu)); }

        if (getTime(timer_smvmGpu) > 0) {
            msg(MSG_PLAIN, "\n");
            msg(1, "Sparse multiplication:");
            if (getTime(timer_smvmCpu) > 0) {
            msg(2, "CPU:     %f (ms)", getTime(timer_smvmCpu)); }
            msg(2, "GPU:     %f (ms)", getTime(timer_smvmGpu));
            if (getTime(timer_smvmCpu) > 0) {
            msg(2, "Speedup: %f x", getTime(timer_smvmCpu) /
                                    getTime(timer_smvmGpu)); }
        }

        if (getTime(timer_DHWD2dGpu) > 0) {
            msg(MSG_PLAIN, "\n");
            msg(1, "DHWD:");
            if (getTime(timer_DHWD2dCpu) > 0) {
            msg(2, "CPU:     %f (ms)", getTime(timer_DHWD2dCpu)); }
            msg(2, "GPU:     %f (ms)", getTime(timer_DHWD2dGpu));
            if (getTime(timer_DHWD2dCpu) > 0) {
            msg(2, "Speedup: %f x", getTime(timer_DHWD2dCpu) /
                                    getTime(timer_DHWD2dGpu)); }
        }

        if (getTime(timer_Dverti2dGpu) > 0) {
            msg(MSG_PLAIN, "\n");
            msg(1, "D vertical:");
            if (getTime(timer_Dverti2dCpu) > 0) {
            msg(2, "CPU:     %f (ms)", getTime(timer_Dverti2dCpu)); }
            msg(2, "GPU:     %f (ms)", getTime(timer_Dverti2dGpu));
            if (getTime(timer_Dverti2dCpu) > 0) {
            msg(2, "Speedup: %f x", getTime(timer_Dverti2dCpu) /
                                    getTime(timer_Dverti2dGpu)); }
        }

        if (getTime(timer_Dhori2dGpu) > 0) {
            msg(MSG_PLAIN, "\n");
            msg(1, "D horizontal:");
            if (getTime(timer_Dhori2dCpu) > 0) {
            msg(2, "CPU:     %f (ms)", getTime(timer_Dhori2dCpu)); }
            msg(2, "GPU:     %f (ms)", getTime(timer_Dhori2dGpu));
            if (getTime(timer_Dhori2dCpu) > 0) {
            msg(2, "Speedup: %f x", getTime(timer_Dhori2dCpu) /
                                    getTime(timer_Dhori2dGpu)); }
        }

        msg(MSG_PLAIN, "\n");
        msg(1, "Overall: (Including both memory and computation)");
        if (getTime(timer_Cpu) > 0) {
        msg(2, "CPU:     %f (ms)", getTime(timer_Cpu)); }
        msg(2, "GPU:     %f (ms)", getTime(timer_Gpu));
        if (getTime(timer_bruteForceCpu) > 0) {
        msg(2, "Speedup: %f x", getTime(timer_Cpu) / getTime(timer_Gpu)); }
        msg(MSG_PLAIN, "\n");
    }

}; // End of class mriTimer

// Global functions for accessing static mriTimer_g variable.
    mriTimer *
getMriTimer(void);
    void
initMriTimer(void);

    void
deleteMriTimer(void);

    void
startMriTimer(unsigned int &timer);
    void
stopMriTimer(unsigned int &timer);
    void
printMriTimer(void);

/*---------------------------------------------------------------------------*/
/*  Memory usage tracing routines                                            */
/*---------------------------------------------------------------------------*/

    void
insertMemoryUsageCpu(void *addr, const unsigned int usage);
    void
insertMemoryUsageGpu(void *addr, const unsigned int usage);

    void
eraseMemoryUsageCpu(void *addr);
    void
eraseMemoryUsageGpu(void *addr);

    unsigned int
getMemoryUsageCpu(void);
    unsigned int
getMemoryUsageGpu(void);

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Allocate/Deallocate a GPU/CPU array in a good-looking way.] */
/*                                                                           */
/*  Description [The returned array should be freed by the caller using the  */
/*      corresponding delete function.]                                      */
/*                                                                           */
/*===========================================================================*/

// GPU =======================================================================

    template <class T> T *
mriNewGpu(const int dim_x, const bool if_clean = true) // Size of X dimension
{
    startMriTimer(getMriTimer()->timer_memAllocateGpu);
    T *t = newCuda<T>(dim_x, if_clean);
    stopMriTimer(getMriTimer()->timer_memAllocateGpu);

    #if DEBUG_MEMORY
    insertMemoryUsageGpu(t, dim_x * sizeof(T));
    if (getMemoryUsageGpu() < 1000000) {
        printf("GPU memory usage: %u bytes\n", getMemoryUsageGpu());
    } else {
        printf("GPU memory usage: %0.2f Mega bytes\n",
            (FLOAT_T) getMemoryUsageGpu()/1000000.0f);
    }
    #endif

    return t;
}

    template <class T> void
mriDeleteGpu(T * var_1d)
{
    #if DEBUG_MEMORY
    eraseMemoryUsageGpu(var_1d);
    #endif

    startMriTimer(getMriTimer()->timer_memAllocateGpu);
    deleteCuda(var_1d);
    stopMriTimer(getMriTimer()->timer_memAllocateGpu);
}

    template <class T> void
mriCopyHostToDevice(T * dst, const T * src, const int dim_x)
{
    // Before accumulating the data copy time, let all kernels finish.
    cudaThreadSynchronize();

    startMriTimer(getMriTimer()->timer_host2Device);
    copyCudaHostToDevice<T>(dst, src, dim_x);
    stopMriTimer(getMriTimer()->timer_host2Device);
}

    template <class T> void
mriCopyDeviceToHost(T * dst, const T * src, const int dim_x)
{
    // Before accumulating the data copy time, let all kernels finish.
    cudaThreadSynchronize();

    startMriTimer(getMriTimer()->timer_device2Host);
    copyCudaDeviceToHost<T>(dst, src, dim_x);
    stopMriTimer(getMriTimer()->timer_device2Host);
}

    template <class T> void
mriCopyDeviceToDevice(T * dst, const T * src, const int dim_x)
{
    // Before doing data copy, let all kernels finish.
    cudaThreadSynchronize();

    copyCudaDeviceToDevice<T>(dst, src, dim_x);
}

// CPU =======================================================================

    template <class T> T *
mriNewCpu(const int dim_x, // Size of X dimension
    T * var_1d2 = NULL) // If given, a copy of var_1d2 is applied.
{
    startMriTimer(getMriTimer()->timer_memAllocateCpu);
    T * t = newArray1D(dim_x, true, var_1d2);
    stopMriTimer(getMriTimer()->timer_memAllocateCpu);

    #if DEBUG_MEMORY
    insertMemoryUsageCpu(t, dim_x * sizeof(T));
    if (getMemoryUsageCpu() < 1000000) {
        printf("CPU memory usage: %u bytes\n", getMemoryUsageCpu());
    } else {
        printf("CPU memory usage: %0.2f Mega bytes\n",
            (FLOAT_T) getMemoryUsageCpu()/1000000.0f);
    }
    #endif
    return t;
}

    template <class T> void
mriDeleteCpu(T * var_1d)
{
    #if DEBUG_MEMORY
    eraseMemoryUsageCpu(var_1d);
    #endif

    startMriTimer(getMriTimer()->timer_memAllocateCpu);
    deleteArray1D(var_1d);
    stopMriTimer(getMriTimer()->timer_memAllocateCpu);
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [A class of FieldMap data elements.]                         */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

class FieldMap
{
public:
    FieldMap(void) {
        fm   = 0.0;
    };

public:
    FLOAT_T fm;
};

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [A class of pointer array of FieldMap type.]                 */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

class FieldMapArray
{
public:
    FieldMapArray(void) { array = NULL; size = 0; };
    FieldMapArray(const int s) { allocate(s); };
    ~FieldMapArray() {
        if (size > 0) mriDeleteCpu(array);
    };

    void allocate(const int s) {
        size = s;
        array = mriNewCpu<FieldMap>(size);
    };

        void
    operator =(const FieldMapArray &a) {
        array = mriNewCpu<FieldMap>(a.size);
        memcpy(array, a.array, sizeof(a.array));
        size = a.size;
    };

public:
    FieldMap *array;
    int size;   // Number of elements in array
};

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [A class of DataTraj data elements.]                         */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

class DataTraj
{
public:
    #if 0   // NVCC doesn't support this C++ feature yet since this class
            // is used to declare the constant memory.
    DataTraj(void) {
        x = 0.0;
        y = 0.0;
        #if !DATATRAJ_NO_Z_DIM
        z = 0.0;
        #endif
    };
    #endif

    FLOAT_T x;
    FLOAT_T y;
    #if !DATATRAJ_NO_Z_DIM
    // Too big to fit in constant memory with 4096 DataTraj elements
    FLOAT_T z;
    #endif
};

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [A class of pointer array of DataTraj type.]                 */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

class DataTrajArray
{
public:
    DataTrajArray(void) { array = NULL; size = 0; };
    DataTrajArray(const int s) { allocate(s); };
    ~DataTrajArray() {
        if (size > 0) mriDeleteCpu(array);
    };

    void allocate(const int s) {
        size = s;
        array = mriNewCpu<DataTraj>(size);
    };

        void
    operator =(const DataTrajArray &a) {
        array = mriNewCpu<DataTraj>(a.size);
        memcpy(array, a.array, sizeof(a.array));
        size = a.size;
    };

public:
    DataTraj *array;
    int size;   // Number of elements in array
};

/*---------------------------------------------------------------------------*/
/*  Sparse-matrix manipulation data structures                               */
/*---------------------------------------------------------------------------*/

#include <mmio.h>

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [The most basic attribute of the matrix data format.]        */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

class MatrixShape
{
public:
    // Constructors/Destructors
    // ========================

    MatrixShape(void) {num_rows = 0; num_cols = 0; num_nonzeros = 0; };

    MatrixShape(int rows, int cols, int nonzeros) {
        num_rows = rows; num_cols = cols; num_nonzeros = nonzeros;
        #if DEBUG_MODE
        cout<< "MatrixShape:"<< " rows: "<< rows<< " cols: "<< cols
            << " nonzeros: "<< nonzeros<< endl;
        #endif
    };

    //~MatrixShape() {}; // use default destructor

    // Copy constructor and copy assignment operator
    // =============================================

    MatrixShape(const MatrixShape &rhs) {
        *this = rhs; // call the assignment operator
    };

        MatrixShape &
    operator = (const MatrixShape &rhs) {
        if (this != &rhs) {
            num_rows = rhs.num_rows;
            num_cols = rhs.num_cols;
            num_nonzeros = rhs.num_nonzeros;
        }
        return *this;
    };

public:
    int num_rows;       // size of rows
    int num_cols;       // size of columns
    int num_nonzeros;   // number of nonzero elements
};

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [COO matrix format data structure]                           */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

// COOrdinate matrix (aka IJV or Triplet format)
class CooMatrix: public MatrixShape
{
public:
    // Constructors/Destructors
    // ========================

    CooMatrix(void) {I = NULL; J = NULL; V = NULL;};
    CooMatrix(const int rows, const int cols, const int nonzeros,
        const int *i, const int *j, const FLOAT_T *v) {
        num_rows = rows;
        num_cols = cols;
        num_nonzeros = nonzeros;
        I = mriNewCpu<int>(num_nonzeros);
        J = mriNewCpu<int>(num_nonzeros);
        V = mriNewCpu<FLOAT_T>(num_nonzeros);
        memcpy(I, i, num_nonzeros*sizeof(int));
        memcpy(J, j, num_nonzeros*sizeof(int));
        memcpy(V, v, num_nonzeros*sizeof(FLOAT_T));
    };

    ~CooMatrix() {
        if (I) mriDeleteCpu(I);
        if (J) mriDeleteCpu(J);
        if (V) mriDeleteCpu(V);
    };

    // Copy constructor and copy assignment operator
    // =============================================

    CooMatrix(const CooMatrix &rhs) : MatrixShape(rhs) {
        *this = rhs; // call the assignment operator
    };

        CooMatrix &
    operator = (const CooMatrix &rhs) {
        if (this != &rhs) {
            MatrixShape::num_rows = rhs.num_rows;
            MatrixShape::num_cols = rhs.num_cols;
            MatrixShape::num_nonzeros = rhs.num_nonzeros;
            I = mriNewCpu<int>(rhs.num_nonzeros);
            J = mriNewCpu<int>(rhs.num_nonzeros);
            V = mriNewCpu<FLOAT_T>(rhs.num_nonzeros);
            memcpy(I, rhs.I, rhs.num_nonzeros*sizeof(int));
            memcpy(J, rhs.J, rhs.num_nonzeros*sizeof(int));
            memcpy(V, rhs.V, rhs.num_nonzeros*sizeof(FLOAT_T));
        }
        return *this;
    };

        void
    print(const string &s) const {
        cout<< s<< endl;
        printf("num_rows: %d, num_cols: %d, num_nonzeros: %d\n",
            num_rows, num_cols, num_nonzeros);
        for (int i = 0; i < num_nonzeros; i++) {
            if (I[i] >= num_rows) {
                printf("Errors on row %d at line %d of %s\n",
                    i, __LINE__, __FILE__);
            }
            if (J[i] >= num_cols) {
                printf("Errors on column %d at line %d of %s\n",
                    i, __LINE__, __FILE__);
            }
        }
    };

public:
    int *I;         // row indices
    int *J;         // column indices
    FLOAT_T *V;     // nonzero values
};

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [CSR matrix format data structure]                           */
/*                                                                           */
/*  Description [CSR format takes three arrays to represent a sparse matrix. */
/*      The first array Ax stores nonzero values in row-major order. The     */
/*      second array Aj stores the corresponding column indexes of in array  */
/*      Av. The numbers of non-zeros in all rows and the corresponding row   */
/*      indexes are encoded in the third array Ap as paired value list.]     */
/*                                                                           */
/*===========================================================================*/

class CsrMatrix: public MatrixShape
{
public:
    // Constructors/Destructors
    // ========================

    CsrMatrix(void) {
        num_Ap = 0; num_Aj = 0; num_Ax = 0;
        Ap = NULL; Aj = NULL; Ax = NULL;
    };

    CsrMatrix(const int rows, const int cols, const int nonzeros,
        const int *ap, const int *aj, const FLOAT_T *ax) :
        MatrixShape(rows, cols, nonzeros) {
        num_Ap = num_rows + 1;
        num_Aj = num_nonzeros;
        num_Ax = num_nonzeros;
        memcpy(Ap, ap, num_Ap * sizeof(int));
        memcpy(Aj, aj, num_Aj * sizeof(int));
        memcpy(Ax, ax, num_Ax * sizeof(FLOAT_T));

        #if DEBUG_MODE
        cout<< "CsrMatrix 1:"<< " num_Ap: "<< num_Ap<< " num_Aj: "<< num_Aj
            << " num_Ax: "<< num_Ax<< endl;
        #endif
    };

    CsrMatrix(const CooMatrix &coo) :
        MatrixShape(coo.num_rows, coo.num_cols, coo.num_nonzeros) {
        num_Ap = num_rows + 1; Ap = mriNewCpu<int>(num_Ap);
        num_Aj = num_nonzeros; Aj = mriNewCpu<int>(num_Aj);
        num_Ax = num_nonzeros; Ax = mriNewCpu<FLOAT_T>(num_Ax);

        #if DEBUG_MODE
        cout<< "CsrMatrix 2:"<< " num_Ap: "<< num_Ap<< " num_Aj: "<< num_Aj
            << " num_Ax: "<< num_Ax<< endl;
        #endif
    };

    ~CsrMatrix() {
        if (Ap) mriDeleteCpu(Ap);
        if (Aj) mriDeleteCpu(Aj);
        if (Ax) mriDeleteCpu(Ax);
    };

    // Copy constructor and copy assignment operator
    // =============================================

    CsrMatrix(const CsrMatrix &rhs) : MatrixShape(rhs) {
        *this = rhs; // call the assignment operator
    };

        CsrMatrix &
    operator = (const CsrMatrix &rhs) {
        if (this != &rhs) {
            MatrixShape::num_rows = rhs.num_rows;
            MatrixShape::num_cols = rhs.num_cols;
            MatrixShape::num_nonzeros = rhs.num_nonzeros;
            num_Ap = rhs.num_Ap; Ap = mriNewCpu<int>(num_Ap);
            num_Aj = rhs.num_Aj; Aj = mriNewCpu<int>(num_Aj);
            num_Ax = rhs.num_Ax; Ax = mriNewCpu<FLOAT_T>(num_Ax);
            memcpy(Ap, rhs.Ap, num_Ap * sizeof(int));
            memcpy(Aj, rhs.Aj, num_Aj * sizeof(int));
            memcpy(Ax, rhs.Ax, num_Ax * sizeof(FLOAT_T));
            #if DEBUG_MODE
            cout<< "CsrMatrix:"<< " num_Ap: "<< num_Ap<< " num_Aj: "<< num_Aj
                << " num_Ax: "<< num_Ax<< endl;
            #endif
        }
        return *this;
    };

public:
    unsigned int num_Ap;
    unsigned int num_Aj;
    unsigned int num_Ax;

    int *Ap;        // row pointer
    int *Aj;        // column indices
    FLOAT_T *Ax;    // nonzeros
};

/*---------------------------------------------------------------------------*/
/*  Sparse-matrix manipulation function prototypes                           */
/*---------------------------------------------------------------------------*/

enum memory_location { HOST_MEMORY, DEVICE_MEMORY };

    template <typename T> T *
new_array(const size_t N, const memory_location loc);

    template <typename T> void
delete_array(T * p, const memory_location loc);

    template<typename T> T *
new_host_array(const size_t N);

    template<typename T> T *
new_device_array(const size_t N);

    template<typename T> void
delete_host_array(T *p);

    template<typename T> void
delete_device_array(T *p);

    void
delete_CooMatrix(CooMatrix & coo, const memory_location loc);

    void
delete_host_matrix(CooMatrix &coo);

    void
sum_csr_duplicates(const int num_rows,
                   const int num_cols,
                         int * Ap,
                         int * Aj,
                         FLOAT_T * Ax);

    void
coo2Csr(const int * rows,
           const int * cols,
           const FLOAT_T * data,
           const int num_rows,
           const int num_cols,
           const int num_nonzeros,
                 int * Ap,
                 int * Aj,
                 FLOAT_T * Ax);

    CsrMatrix
coo2Csr(const CooMatrix & coo, const bool compact = false);

    CsrMatrix
mtx2Csr(
    const int *I, const int *J, const FLOAT_T *V,
    const int num_rows, const int num_cols, const int num_nonzeros);

    CooMatrix
read_CooMatrix(const char * mm_filename);

    CsrMatrix
read_CsrMatrix(const char * mm_filename, bool compact = false);

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [A class of data elements used in a CG solver.]              */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

class CgSolverData
{
public:
    CgSolverData(void) {
        num_gpus = 0;
        num_slices_total = 0;
        num_slices_GPU = 0;
        gpu_id = 0;
        sliceGPU_Idx = 0;
        slice_id = 0;
        idata_r = NULL;
        idata_i = NULL;
        kdata_r = NULL;
        kdata_i = NULL;
        sensi_r = NULL;
        sensi_i = NULL;
        num_coil  = 0;
        ktraj   = NULL;
        itraj   = NULL;
        fm      = NULL;
        time    = NULL;
        c       = NULL;
        cg_num = 0;
        num_k = 0;
        num_i = 0;
        timer_gpu = NULL;
        enable_regularization = false;
        enable_finite_difference = false;
        fd_penalizer = 0.0;
        enable_total_variation = false;
        tv_num = 0;
        enable_tv_update = false;

		//Jiading GAI
		output_file_gpu_r = string("dummy_r");
		output_file_gpu_i = string("dummy_i");
    };

    ~CgSolverData(void) {
        if (idata_r  ) mriDeleteCpu(idata_r  );
        if (idata_i  ) mriDeleteCpu(idata_i  );
        if (kdata_r  ) mriDeleteCpu(kdata_r  );
        if (kdata_i  ) mriDeleteCpu(kdata_i  );
        if (sensi_r  ) mriDeleteCpu(sensi_r  );
        if (sensi_i  ) mriDeleteCpu(sensi_i  );
        if (ktraj    ) mriDeleteCpu(ktraj    );
        if (itraj    ) mriDeleteCpu(itraj    );
        if (fm       ) mriDeleteCpu(fm     );
        if (time     ) mriDeleteCpu(time     );
        if (c        ) mriDeleteCpu(c        );
        if (timer_gpu) mriDeleteCpu(timer_gpu);
    };

public:
    int num_gpus;
    int num_slices_total;
    int num_slices_GPU;
    int gpu_id;
    int sliceGPU_Idx;
    int slice_id;
    FLOAT_T *idata_r;
    FLOAT_T *idata_i;
    FLOAT_T *kdata_r;
    FLOAT_T *kdata_i;
    FLOAT_T *sensi_r;
    FLOAT_T *sensi_i;
    int num_coil;
    DataTraj *ktraj;
    DataTraj *itraj;
    FLOAT_T *fm;
    FLOAT_T *time;
    CooMatrix *c;
    int cg_num;
    int num_k;
    int num_i;
    class mriTimer *timer_gpu;
    bool enable_regularization;
    bool enable_finite_difference;
    FLOAT_T fd_penalizer;
    bool enable_total_variation;
    int tv_num;
    bool enable_tv_update;
    string output_file_gpu_r;
    string output_file_gpu_i;
};

/*---------------------------------------------------------------------------*/
/*  Other Function prototypes                                                */
/*---------------------------------------------------------------------------*/

    void
loadInputData(
    const std::string &input_folder_path,FLOAT_T &version,
    int &num_slices, int &num_k, int &num_i, int &num_coil,
    DataTrajArray &ktraj, DataTrajArray &itraj,
    TArray<FLOAT_T> &kdata_r, TArray<FLOAT_T> &kdata_i,
    TArray<FLOAT_T> &idata_r, TArray<FLOAT_T> &idata_i,
    TArray<FLOAT_T> &sensi_r, TArray<FLOAT_T> &sensi_i,
    TArray<FLOAT_T> &fm, TArray<FLOAT_T> &t,
    CooMatrix &c, const bool enable_regularization,
	int &Nx, int &Ny, int &Nz);

    TArray<FLOAT_T>
readDataFile(const string &fn, FLOAT_T &version, int *data_dims, int &ncoils,
    int &nslices, int &sensed_data_size);

    TArray<FLOAT_T>
readDataFile_10(const string &fn, FLOAT_T &version, int *data_dims, int &ncoils,
    int &nslices, int &sensed_data_size, int &Nx, int &Ny, int &Nz);

// Export FLOAT_T array to file
    void 
exportDataCpu(const string &fn, const FLOAT_T *array, int num_elements);
    void 
exportDataCpu(const std::string &fn, const TArray<FLOAT_T> &fa, int num_elements);
    void
exportDataGpu(const std::string &fn, FLOAT_T *array, int num_elements);

    FLOAT_T *
padVectorPowerOfTwo(const FLOAT_T *array, const int element_num);

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

#endif // STRUCTURES_H

