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

    File Name   [tools.h]

    Synopsis    [This file defines the necessary macros and helper functions
        used in other files.]

    Description [This file is the toppest file included by all other files.
        So you must include this file at the very first place than other
        project header files.]

    Revision    [0.1; Initial build; Yue Zhuo, BIOE UIUC;
                 Xiao-Long Wu, ECE UIUC]
    Revision    [0.1.1; Calculating FLOPS, Code cleaning;
                 Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Further optimization, Code cleaning, Adding more comments;
                 Xiao-Long Wu, ECE UIUC, Jiading Gai, Beckman Institute]
    Date        [10/27/2010]

*****************************************************************************/

#ifndef TOOLS_H
#define TOOLS_H

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>

// XCPPLIB libraries
#include <xcpplib_process.h>
#include <xcpplib_types.h>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Namespace used                                                           */
/*---------------------------------------------------------------------------*/

using namespace xcpplib;

/*---------------------------------------------------------------------------*/
/*  Macro definitions from Makefile                                          */
/*---------------------------------------------------------------------------*/

// FIXME: Need to add a prefix to differentiate the debug modes.
// Note: Emulation mode is no longer supported in the newer NVCC compilers.
#ifdef DEBUG                            // Flag from Makefile
    #define DEBUG_MODE          true    // enable to display debugging messages
    #define DEBUG_KERNEL        true    // enable to have additional checks
    #define DEBUG_KERNEL_MSG    true    // enabled to show more information
    #define DEBUG_MEMORY        true    // enabled to check memory usage
#else
    #define DEBUG_MODE          false
    #define DEBUG_KERNEL        false
    #define DEBUG_KERNEL_MSG    false
    #define DEBUG_MEMORY        false
#endif

#ifdef ENABLE_OPENMP                    // Flag from Makefile
    #define USE_OPENMP          true    // enabled OpenMP on CPU code
#else
    #define USE_OPENMP          false   // disabled OpenMP on CPU code
#endif

// Enable double-precision support or not. It is supported in GT200 based
// cards, GTX260/275/280/285/295, Telsa C1060, Telsa S1070, Quadro FX5800
// and the new Fermi based cards.
// If double precision is supported in current device, this should be enabled
// in the Makefile. If not, using single-precision computation is faster.
// Note: Using double-precision support on platforms not supporting it
//       can cause execution failure.

#ifdef ENABLE_DOUBLE_PRECISION          // Flag from Makefile
    #define FLOAT_T             double  // float
#else
    #define FLOAT_T             float
#endif

#ifdef ENABLE_COMPUTE_FLOPS             // Flag from Makefile
    #define COMPUTE_FLOPS       true    // enabled to compute FLOPs
                                        // This can slow down the performance
                                        // a little.
#else
    #define COMPUTE_FLOPS       false   // disabled to compute FLOPS
#endif

// Error checking scheme for defined macros ==================================

#if COMPUTE_FLOPS && USE_OPENMP
    #error Makefile flag ENABLE_COMPUTE_FLOPS conflicts with ENABLE_OPENMP flag.
    #error You can set either ENABLE_COMPUTE_FLOPS or ENABLE_OPENMP.
#endif

/*---------------------------------------------------------------------------*/
/*  Target device-related parameters                                         */
/*---------------------------------------------------------------------------*/

// Include OpenMP library
#if USE_OPENMP
    #include <omp.h>
#endif

/*---------------------------------------------------------------------------*/
/*  Project macro definitions                                                */
/*---------------------------------------------------------------------------*/

// Macros
#ifndef MIN
    #define MIN(a, b)    (a < b ? a : b)
#endif

// Error handling
#define CUDA_ERRCK {                                                \
        cudaError_t err;                                            \
        if ((err = cudaGetLastError()) != cudaSuccess) {            \
                 fprintf(stderr, "CUDA error on line %d: %s\n",     \
                         __LINE__, cudaGetErrorString(err));        \
                 exit(-1);                                          \
             }                                                      \
        }

/*---------------------------------------------------------------------------*/
/*  Helper data structures                                                   */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*  Function prototypes                                                      */
/*---------------------------------------------------------------------------*/

    bool
isPowerOfTwo(int n);

    int
getLeastPowerOfTwo(const int value);

// Duplicate the content of an array on the CPU
    void
duplicate1dArray(FLOAT_T **output, FLOAT_T *input, int num_elements);

// Evaluate the Normalized Root Mean Squared Deviation
    FLOAT_T
getNRMSD(
    const FLOAT_T *idata_r, const FLOAT_T *idata_i,
    const FLOAT_T *idata_cpu_r, const FLOAT_T *idata_cpu_i,
    const int num_elements);

// Evaluate the Normalized Root Mean Squared Error
    double
getNRMSE(
    const FLOAT_T *idata_r, const FLOAT_T *idata_i,
    const FLOAT_T *idata_cpu_r, const FLOAT_T *idata_cpu_i,
    const int num_elements);

// Evaluate the Mean Squared Error
    FLOAT_T
get_MSE(FLOAT_T *idata_r, FLOAT_T *idata_i, FLOAT_T *idata_cpu_r,
        FLOAT_T *idata_cpu_i, int num_elements);

// Display CPU array
#define DISPLAY_ARRAY_SIZE      10
    void
dispCPUArray(FLOAT_T *X, int N);
    void
dispCPUArray(FLOAT_T *X, int N, int startIdx, int dispLength);

#if 0   // GPU device functions should be put in another .CU files.
// Display GPU array
    void
dispGPUArray(FLOAT_T *X, int N);
    void
dispGPUArray(FLOAT_T *X, int N, int startIdx, int dispLength);
    __global__ void
dispGPUArrayKernel(FLOAT_T *X, int N, int startIdx, int dispLength);
#endif

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

#endif // TOOLS_H

