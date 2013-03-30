/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */
 
#ifndef _CUTIL_INLINE_H_
#define _CUTIL_INLINE_H_

#include <cuda.h>
#include <cutil.h>
#include <cuda_runtime_api.h>

#include <cutil_inline_bankchecker.h>
#include <cutil_inline_runtime.h>
#include <cutil_inline_drvapi.h>

inline void print_NVCC_min_spec(const char *sSDKsample, const char *sNVCCReq, const char *sDriverReq)
{
    printf("CUDA %d.%02d Toolkit built this project.\n", CUDART_VERSION/1000, (CUDART_VERSION%100));
    printf("  [ %s ] requirements:\n", sSDKsample);
    printf(" -> CUDA %s Toolkit\n"  , sNVCCReq);
    printf(" -> %s NVIDIA Display Driver.\n", sDriverReq);
}

#define ALIGN_OFFSET(offset, alignment) offset = (offset + (alignment-1)) & ~((alignment-1))


#endif // _CUTIL_INLINE_H_
