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

    File Name   [tools.cpp]

    Synopsis    [This file defines the necessary macros and helper functions
        used in other files.]

    Description [See the corresponding header file for more information.]

    Revision    [0.1; Initial build; Yue Zhuo, BIOE UIUC;
                 Xiao-Long Wu, ECE UIUC]
    Revision    [0.1.1; Calculating FLOPS, Code cleaning;
                 Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Further optimization, Code cleaning, Adding more comments;
                 Xiao-Long Wu, ECE UIUC, Jiading Gai, Beckman Institute]
    Date        [10/27/2010]

*****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// Standard libraries
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

// CUDA library
#include <cutil.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>

// XCPPLIB libraries
#include <xcpplib_process.h>
#include <xcpplib_types.h>
#include <xcpplib_typesGpu.cuh>

// MRI project related files
#include <tools.h>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Namespace used                                                           */
/*---------------------------------------------------------------------------*/

using namespace std;
using namespace xcpplib;

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

// ==================== isPowerOfTwo ====================
    bool 
isPowerOfTwo(int n)
{
    return ((n & (n - 1)) == 0);
}

// ==================== getLeastPowerOfTwo ====================
// Get the least number of power two that is greater than value.
// E.g., value = 8, p = 8.
//       value = 12, p = 16.
//       value = 16, p = 16.
//       value = 18, p = 32.
    int 
getLeastPowerOfTwo(const int value)
{
    int num = value;
    int exp = 0;
    frexp((float) num, &exp);
    int p = (int) pow(2.0, exp);

    // frexp may generate larger value
    if (num == (int) pow(2.0, exp - 1)) {
        p = (int) pow(2.0, exp - 1);
    }

    return p;
}

// ==================== duplicate1dArray ====================
    void
duplicate1dArray(FLOAT_T *output, FLOAT_T *input, int num_elements)
{
    makeSure(output, __FILE__, __LINE__);
    memcpy(output, input, num_elements * sizeof(FLOAT_T));
}
#if 0
void duplicate_array(FLOAT_T **output, FLOAT_T *input, int num_elements)
{
    *output = (FLOAT_T *) malloc(num_elements * sizeof(FLOAT_T));
    memcpy(*output, input, num_elements * sizeof(FLOAT_T));
/*
    for(int i=0; i<num_elements; i++)
    {
        (*output)[i]=input[i];
    }
 */
}
#endif

// ==================== getNRMSD ====================
    FLOAT_T
getNRMSD(
    const FLOAT_T *idata_r, const FLOAT_T *idata_i,
    const FLOAT_T *idata_cpu_r, const FLOAT_T *idata_cpu_i,
    const int num_elements)
{
    FLOAT_T error = 0.0;
    FLOAT_T gpu_mag = 0.0;
    for (int i = 0; i < num_elements; i++) {
        gpu_mag += sqrt(pow(idata_r[i], 2) + pow(idata_i[i], 2));
        error += pow(sqrt(pow(idata_r[i], 2) + pow(idata_i[i], 2)) - sqrt(pow(idata_cpu_r[i], 2) + pow(idata_cpu_i[i], 2)), 2);
    }
    error = sqrt(error) / (num_elements * gpu_mag);
    return error;
}

// ==================== getNRMSE ====================
    double
getNRMSE(
    const FLOAT_T *idata_r, const FLOAT_T *idata_i,
    const FLOAT_T *idata_cpu_r, const FLOAT_T *idata_cpu_i,
    const int num_elements)
{
    double error = 0.0;
    double cpu_mag = 0.0;
    for (int i = 0; i < num_elements; i++) {
        cpu_mag += pow(idata_cpu_r[i], 2) + pow(idata_cpu_i[i], 2);
        error += pow(sqrt(pow(idata_r[i], 2) + pow(idata_i[i], 2)) - sqrt(pow(idata_cpu_r[i], 2) + pow(idata_cpu_i[i], 2)), 2);
    }
    error = sqrt(error / cpu_mag);
    return error;
}

// ==================== get_MSE ====================
FLOAT_T get_MSE(FLOAT_T *idata_r, FLOAT_T *idata_i, FLOAT_T *idata_cpu_r, FLOAT_T *idata_cpu_i, int num_elements)
{
    FLOAT_T error = 0.0;
    for (int i = 0; i < num_elements; i++) {
        error += pow(sqrt(pow(idata_r[i], 2) + pow(idata_i[i], 2)) - sqrt(pow(idata_cpu_r[i], 2) + pow(idata_cpu_i[i], 2)), 2);
    }
    error = sqrt(error);
    return error;
}

// ==================== dispCPUArray ====================
void dispCPUArray(FLOAT_T *X, int N)
{
    printf("\n---\n");
    for (int i = 0; i < min(N, DISPLAY_ARRAY_SIZE); i++) {
        printf("%f, ", X[i]);
    }
    printf("\n---\n");
}

void dispCPUArray(FLOAT_T *X, int N, int startIdx, int dispLength)
{
    printf("\n---\n");
    for (int i = startIdx; i < min(N, startIdx + dispLength); i++) {
        printf("%f, ", X[i]);
    }
    printf("\n---\n");
}

#if 0   // GPU device functions should be put in another .CU files.
// ==================== dispGPUArray ====================
#ifdef __DEVICE_EMULATION__
void dispGPUArray(FLOAT_T *X, int N)
{
    dim3 dimGrid(1, 1);
    dim3 dimBlock(1, 1);
    dispGPUArrayKernel <<< dimGrid, dimBlock >>> (X, N, 0, DISPLAY_ARRAY_SIZE);
}

void dispGPUArray(FLOAT_T *X, int N, int startIdx, int dispLength)
{
    dim3 dimGrid(1, 1);
    dim3 dimBlock(1, 1);
    dispGPUArrayKernel <<< dimGrid, dimBlock >>> (X, N, startIdx, dispLength);
}

__global__ void dispGPUArrayKernel(FLOAT_T *X, int N, int startIdx, int dispLength)
{
    printf("\n---\n");
    for (int i = startIdx; i < min(N, startIdx + dispLength); i++) {
        printf("%f, ", X[i]);
    }
    printf("\n---\n");
}
#else
#endif
#endif

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

