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

    File Name   [recon.cu]

    Synopsis    [Toeplitz's CG Solver.]

    Description []

    Revision    [1.0; Initial build; Sam S. Stone, ECE UIUC]
    Revision    [2.0; Code extension; Jiading Gai, Beckman Institute UIUC and 
                Xiao-Long Wu, ECE UIUC]
    Date        [03/23/2011]

*****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

#include <cufft.h>
//#include <cuda_runtime.h>
//#include <cuda_pthreads.h>
#include <cutil.h>
//#include <mcuda.h>
#include <assert.h>

// XCPPLIB libraries
#include <xcpplib_process.h>
#include <xcpplib_types.h>
#include <structures.h>

// Project header files
#include <utils.h>
#include <DHWD2dGpu_toeplitz.cuh>
#include <DHWD3dGpu_toeplitz.cuh>
#include <Dhori2dGpu_toeplitz.cuh>
#include <Dverti2dGpu_toeplitz.cuh>
#include <Dhori3dGpu_toeplitz.cuh>
#include <Dverti3dGpu_toeplitz.cuh>
#include <Dzaxis3dGpu_toeplitz.cuh>
#include <multiplyGpu_toeplitz.cuh>

#include <smvmGpu.cuh>
#include <pointMultGpu.cuh>
#include <addGpu.cuh>
#include <fftshift.cuh>

/**************************************************************************/
// Sources:
// (1) http://www.cs.toronto.edu/~dross/code/matlab-mex.zip [example of mex for 3d
// arrays]
// (2) http://developer.nvidia.com/object/matlab_cuda.html [2D FFT CUDA plug-in for
// MATLAB]
// (3) http://grove.circa.ufl.edu/matlab_help/techdoc/apiref/mx-c37.html#106737
// [Excellent!! Sparse Matrices]
/**************************************************************************/

// Assumes that num elems is a multiple of 256
#define GPU_DOTPROD_TILE_SIZE 256

// CC: both sources are complex-valued
    __global__ void
CudaDotProdCC(cufftComplex *Src1, cufftComplex *Src2, cufftComplex *Dest) 
{
    int index = blockIdx.y * (gridDim.x * GPU_DOTPROD_TILE_SIZE) + 
                blockIdx.x * GPU_DOTPROD_TILE_SIZE + threadIdx.x;
    float Src1Real = Src1[index] REAL;
    float Src1Imag = Src1[index] IMAG;
    float Src2Real = Src2[index] REAL;
    float Src2Imag = Src2[index] IMAG;
    Dest[index] REAL = Src1Real * Src2Real - Src1Imag * Src2Imag;
    Dest[index] IMAG = Src1Real * Src2Imag + Src1Imag * Src2Real;
}

// RC: Src1 is real-valued, Src2 is complex-valued
    __global__ void
CudaDotProdRC(cufftComplex *Src1, cufftComplex *Src2, cufftComplex *Dest) 
{
    int index = blockIdx.y * (gridDim.x * GPU_DOTPROD_TILE_SIZE) + 
                blockIdx.x * GPU_DOTPROD_TILE_SIZE + threadIdx.x;
    float Src1Real = Src1[index] REAL;
    float Src2Real = Src2[index] REAL;
    float Src2Imag = Src2[index] IMAG;
    Dest[index] REAL = Src1Real * Src2Real;
    Dest[index] IMAG = Src1Real * Src2Imag;
}

// RR: bot sources are real-valued
    __global__ void
CudaDotProdRR(cufftComplex *Src1, cufftComplex *Src2, cufftComplex *Dest) 
{
    int index = blockIdx.y * (gridDim.x * GPU_DOTPROD_TILE_SIZE) + 
                blockIdx.x * GPU_DOTPROD_TILE_SIZE + threadIdx.x;
    float Src1Real = Src1[index] REAL;
    float Src2Real = Src2[index] REAL;
    Dest[index] REAL = Src1Real * Src2Real;
    Dest[index] IMAG = 0.0f;
}

    void 
cuda_dot_prod(matrix_t&src1, matrix_t&src2, matrix_t&dest, int elems) 
{
    dim3 threads(GPU_DOTPROD_TILE_SIZE, 1);
    int xBlocks = elems / GPU_DOTPROD_TILE_SIZE;
    int yBlocks = 1;
    while (xBlocks > 32768) {
        yBlocks <<= 1;
        if (xBlocks % 2) {
            xBlocks >>= 1;
            xBlocks++;
        } else {
            xBlocks >>= 1;
        }
    }
    dim3 blocks(xBlocks, yBlocks);
    #if DEBUG
    printf("Launching cudaDotProd with (%d %d) blocks (%d %d) threads, \
            elems=%d\n", blocks.x, blocks.y, threads.x, threads.y, elems);
    #endif

    if (src1.isComplex) {
        if (src2.isComplex) {
            CudaDotProdCC <<< blocks, threads >>> 
                             (src1.device, src2.device, dest.device);
            dest.isComplex = 1;
            #if DEBUG
            dest.copy_to_host();
            #endif
        } else {
            CudaDotProdRC <<< blocks, threads >>> 
                             (src2.device, src1.device, dest.device);
            dest.isComplex = 1;
            #if DEBUG
            dest.copy_to_host();
            #endif
        }
    } else {
        if (src2.isComplex) {
            CudaDotProdRC <<< blocks, threads >>> 
                             (src1.device, src2.device, dest.device);
            dest.isComplex = 1;
            #if DEBUG
            dest.copy_to_host();
            #endif
        } else {
            CudaDotProdRR <<< blocks, threads >>> 
                             (src1.device, src2.device, dest.device);
            dest.isComplex = 0;
            #if DEBUG
            dest.copy_to_host();
            #endif
        }
    }
}

    __global__ void
CudaDotProdReal(cufftComplex *Src1, cufftComplex *Src2, float *Dest) 
{
    int index = blockIdx.y * (gridDim.x * GPU_DOTPROD_TILE_SIZE) + 
                blockIdx.x * GPU_DOTPROD_TILE_SIZE + threadIdx.x;
    float Src1Real = Src1[index] REAL;
    float Src1Imag = Src1[index] IMAG;
    float Src2Real = Src2[index] REAL;
    float Src2Imag = Src2[index] IMAG;
    Dest[index] = Src1Real * Src2Real + Src1Imag * Src2Imag;
}


// compute real part of dot product
void cuda_dot_prod_real(matrix_t&src1, matrix_t&src2, float *dest, int elems) {
    dim3 threads(GPU_DOTPROD_TILE_SIZE, 1);
    int xBlocks = elems / GPU_DOTPROD_TILE_SIZE;
    int yBlocks = 1;
    while (xBlocks > 32768) {
        yBlocks <<= 1;
        if (xBlocks % 2) {
            xBlocks >>= 1;
            xBlocks++;
        } else {
            xBlocks >>= 1;
        }
    }
    dim3 blocks(xBlocks, yBlocks);

    #if 0 //DEBUG
    printf("Launching cudaDotProd with (%d %d) blocks (%d %d) threads, \
            elems=%d, Src1 %x Src2 %x Dst %x\n", blocks.x, blocks.y, threads.x, 
            threads.y, elems, &src1, &src2, &dest);
    #endif

    CudaDotProdReal <<< blocks, threads >>> (src1.device, src2.device, dest);

    #if DEBUG
//      dest.copy_to_host();
    #endif
}

    void 
cuda_fftshift(matrix_t&src, matrix_t&dest, int rows, int cols, int inverse) 
{
    int cudaRows = 0;
    int cudaCols = 0;
    if (inverse) {
        cudaRows = (int)floor(double(rows / 2));
        cudaCols = (int)floor(double(cols / 2));
    } else {
        cudaRows = (int)ceil(double(rows / 2));
        cudaCols = (int)ceil(double(cols / 2));
    }

    #if DEBUG
    dest.copy_to_host();
    #endif

    dim3 threads(FFTSHIFT_TILE_SIZE_X, FFTSHIFT_TILE_SIZE_Y);
    dim3 blocks(cudaCols / FFTSHIFT_TILE_SIZE_X, cudaRows / FFTSHIFT_TILE_SIZE_Y);

    #if DEBUG
    printf("Launching cudaFFTShift (inverse=%d) with (%d %d) blocks (%d %d) \
           threads, rows=%d cols=%d\n", inverse, blocks.x, blocks.y, threads.x, 
           threads.y, rows, cols);
    #endif

    CudaFFTShift <<< blocks, threads >>> 
                     (src.device, cudaRows, cudaCols, cols, dest.device);

    #if DEBUG
    dest.copy_to_host();
    #endif
    dest.isComplex = src.isComplex;
}

    void 
cuda_fft3shift(
    matrix_t&src, matrix_t&dest, int dimY, 
    int dimX, int dimZ, int inverse) 
{
    int pivotY = 0;
    int pivotX = 0;
    int pivotZ = 0;
    if (inverse) {
        pivotY = (int)floor(double(dimY / 2));
        pivotX = (int)floor(double(dimX / 2));
        pivotZ = (int)floor(double(dimZ / 2));
    } else {
        pivotY = (int)ceil(double(dimY / 2));
        pivotX = (int)ceil(double(dimX / 2));
        pivotZ = (int)ceil(double(dimZ / 2));
    }

    dim3 threads(pivotZ, 1);
    dim3 blocks(pivotY, pivotX);

    #if DEBUG
    printf("Launching CudaFFT3Shift (inverse=%d) with (%d %d) blocks (%d %d) \
    threads, pivot2=%d pivot1=%d, pivot0=%d, dimX=%d, dimZ=%d\n", inverse, blocks.x, 
    blocks.y, threads.x, threads.y, pivotY, pivotX, pivotZ, dimX, dimZ);
    #endif
    CudaFFT3Shift <<< blocks, threads >>> 
                  (src.device, dest.device, pivotY, pivotX, pivotZ, dimX, dimZ);
    #if DEBUG
    dest.copy_to_host();
    #endif

    dest.isComplex = src.isComplex;
}

    __global__ void
CudaOversample(
    cufftComplex *Src, cufftComplex *Dst, int N1, int N2, int N3,
    int midN1, int midN2, int midN3, int pageSize, int rowSize) 
{
    int srcIndex = blockIdx.y * N2 * N3 + blockIdx.x * N3 + threadIdx.x;
    int dstIndex = (midN1 + blockIdx.y) * pageSize + 
                   (midN2 + blockIdx.x) * rowSize  + 
                   (midN3 + threadIdx.x);
    Dst[dstIndex] REAL = Src[srcIndex] REAL;
    Dst[dstIndex] IMAG = Src[srcIndex] IMAG;
}


    void 
cuda_oversample(matrix_t&src, matrix_t&dest, int N1, int N2, int N3) 
{
 /* (N1,N2,N3) is the image size of src; sizeof(dst) = 
    (2*N1,2*N2,2*N3); And: 
 
    dst(midN1:2*N1-midN1,midN2:2*N2-midN2,midN3:2*N3-midN3) = src.
     
    cuda_oversample puts 'src' in the center of 'dst'. 
    Row major layout for both image and threads.  */

    assert(N3 <= 512);
    dim3 threads(N3, 1);
    int yBlocks = N1;
    int xBlocks = N2;
    dim3 blocks(xBlocks, yBlocks);

    int midN1 = N1 - (N1 / 2);
    int midN2 = N2 - (N2 / 2);

    //Jiading GAI
        /*3D - JGAI - BEGIN*/
    int midN3;
    int pageSize;
    int rowSize;
    if(1==N3)
    {
       midN3 = 0;
       pageSize = 2 * N2;
       rowSize = 1;
    }
    else if(1<N3)
    {
       midN3 = N3 - (N3/2);
       pageSize = 2*N2 * 2*N3;
       rowSize = 2*N3;
    }
        /*3D - JGAI - END*/

    CudaOversample <<< blocks, threads >>> 
                      (src.device, dest.device, N1, N2, N3, 
                       midN1, midN2, midN3, pageSize, rowSize);

    #if DEBUG
    dest.copy_to_host();
    #endif

    dest.isComplex = src.isComplex;
}

    __global__ void
CudaUndersampleScale(
    cufftComplex *Src, cufftComplex *Dst, int Ny, int Nx, int Nz,
    int midNy, int midNx, int midNz, int pageSize, int rowSize, float scale) 
{
    int dstIndex = blockIdx.y * Nx * Nz + blockIdx.x * Nz + threadIdx.x;
    int srcIndex = (midNy + blockIdx.y) * pageSize + 
                   (midNx + blockIdx.x) * rowSize  + (midNz + threadIdx.x);
    Dst[dstIndex] REAL = scale * Src[srcIndex] REAL;
    Dst[dstIndex] IMAG = scale * Src[srcIndex] IMAG;
}

    void 
cuda_undersample_scale(
    matrix_t&src, matrix_t&dest, int Ny, int Nx, int Nz, float scale) 
{
    assert(Nz <= 512);
    dim3 threads(Nz, 1);
    int yBlocks = Ny;
    int xBlocks = Nx;
    dim3 blocks(xBlocks, yBlocks);

    int midNy = Ny - (Ny / 2);
    int midNx = Nx - (Nx / 2);

    // Jiading GAI
        /*3D - JGAI - BEGIN*/ 
    int midNz;
    int pageSize;
    int rowSize;
    if(1==Nz)
    {
       midNz = 0;
       pageSize = 2 * Nx;
       rowSize = 1;
    }
    else if(1<Nz)
    {
       midNz = Nz - (Nz/2);
       pageSize = 2*Nx * 2*Nz;
       rowSize = 2*Nz;
    }
        /*3D - JGAI - END*/ 

    CudaUndersampleScale <<< blocks, threads >>> 
                            (src.device, dest.device, Ny, Nx, Nz, midNy, 
                            midNx, midNz, pageSize, rowSize, scale);
    #if DEBUG
    dest.copy_to_host();
    #endif

    dest.isComplex = src.isComplex;
}

#define GPU_ADD_SCALE_TILE_SIZE 256

    __global__ void
CudaAddScaleCC(
    cufftComplex *Src1, cufftComplex *Src2, cufftComplex *Dst, float Src1Scale) 
{
    int index = blockIdx.y * (gridDim.x * GPU_ADD_SCALE_TILE_SIZE) + 
                blockIdx.x * GPU_ADD_SCALE_TILE_SIZE + threadIdx.x;
    Dst[index] REAL = Src1Scale * Src1[index] REAL + Src2[index] REAL;
    Dst[index] IMAG = Src1Scale * Src1[index] IMAG + Src2[index] IMAG;
}

    __global__ void
CudaAddScaleRC(
    cufftComplex *Src1, cufftComplex *Src2, cufftComplex *Dst, float Src1Scale) 
{
    int index = blockIdx.y * (gridDim.x * GPU_ADD_SCALE_TILE_SIZE) + 
                blockIdx.x * GPU_ADD_SCALE_TILE_SIZE + threadIdx.x;
    Dst[index] REAL = Src1Scale * Src1[index] REAL + Src2[index] REAL;
    Dst[index] IMAG = Src2[index] IMAG;
}

    __global__ void
CudaAddScaleCR(
    cufftComplex *Src1, cufftComplex *Src2, cufftComplex *Dst, float Src1Scale) 
{
    int index = blockIdx.y * (gridDim.x * GPU_ADD_SCALE_TILE_SIZE) + 
                blockIdx.x * GPU_ADD_SCALE_TILE_SIZE + threadIdx.x;
    Dst[index] REAL = Src1Scale * Src1[index] REAL + Src2[index] REAL;
    Dst[index] IMAG = Src1Scale * Src1[index] IMAG;
}

    void 
cuda_add_scale(
    matrix_t&src1, matrix_t&src2, matrix_t&dst, float Src1Scale, int elems) 
{
    dim3 threads(GPU_ADD_SCALE_TILE_SIZE, 1);
    int xBlocks = elems / GPU_ADD_SCALE_TILE_SIZE;
    int yBlocks = 1;
    while (xBlocks > 32768) {
        yBlocks <<= 1;
        if (xBlocks % 2) {
            xBlocks >>= 1;
            xBlocks++;
        } else {
            xBlocks >>= 1;
        }
    }
    dim3 blocks(xBlocks, yBlocks);

    if (src1.isComplex) {
        if (src2.isComplex) {
            CudaAddScaleCC <<< blocks, threads >>> 
                              (src1.device, src2.device, dst.device, Src1Scale);
            dst.isComplex = 1;
        } else {
            CudaAddScaleCR <<< blocks, threads >>> 
                              (src1.device, src2.device, dst.device, Src1Scale);
            dst.isComplex = 1;
        }
    } else {
        if (src2.isComplex) {
            CudaAddScaleRC <<< blocks, threads >>> 
                              (src1.device, src2.device, dst.device, Src1Scale);
            dst.isComplex = 1;
        } else {
            printf("Error: Sum of two real-valued vectors not implemented\n"); 
            exit(1);
        }
    }
}

#define GPU_SUB_TILE_SIZE 256

    __global__ void
CudaSubCC(
    cufftComplex *Src1, cufftComplex *Src2, cufftComplex *Dst) 
{
    int index = blockIdx.y * (gridDim.x * GPU_SUB_TILE_SIZE) + 
                blockIdx.x * GPU_SUB_TILE_SIZE + threadIdx.x;
    Dst[index] REAL = Src1[index] REAL - Src2[index] REAL;
    Dst[index] IMAG = Src1[index] IMAG - Src2[index] IMAG;
}

    __global__ void
CudaSubRC(cufftComplex *Src1, cufftComplex *Src2, cufftComplex *Dst) 
{
    int index = blockIdx.y * (gridDim.x * GPU_SUB_TILE_SIZE) + 
                blockIdx.x *  GPU_SUB_TILE_SIZE + threadIdx.x;
    Dst[index] REAL = Src1[index] REAL - Src2[index] REAL;
    Dst[index] IMAG = -Src2[index] IMAG;
}

    __global__ void
CudaSubCR(cufftComplex *Src1, cufftComplex *Src2, cufftComplex *Dst) 
{
    int index = blockIdx.y * (gridDim.x * GPU_SUB_TILE_SIZE) + 
                blockIdx.x * GPU_SUB_TILE_SIZE + threadIdx.x;
    Dst[index] REAL = Src1[index] REAL - Src2[index] REAL;
    Dst[index] IMAG = Src1[index] IMAG;
}

    void 
cuda_sub(
    matrix_t&src1, matrix_t&src2, matrix_t&dst, int elems) 
{
    dim3 threads( GPU_SUB_TILE_SIZE, 1);
    int xBlocks = elems / GPU_SUB_TILE_SIZE;
    int yBlocks = 1;
    while (xBlocks > 32768) {
        yBlocks <<= 1;
        if (xBlocks % 2) {
            xBlocks >>= 1;
            xBlocks++;
        } else {
            xBlocks >>= 1;
        }
    }
    dim3 blocks(xBlocks, yBlocks);

    if (src1.isComplex) {
        if (src2.isComplex) {
            CudaSubCC <<< blocks, threads >>> 
                         (src1.device, src2.device, dst.device);
            dst.isComplex = 1;
        } else {
            CudaSubCR <<< blocks, threads >>> 
                         (src1.device, src2.device, dst.device);
            dst.isComplex = 1;
        }
    } else {
        if (src2.isComplex) {
            CudaSubRC <<< blocks, threads >>> 
                         (src1.device, src2.device, dst.device);
            dst.isComplex = 1;
        } else {
            printf("Error: Difference of two real-valued vectors not implemented\n"); 
            exit(1);
            dst.isComplex = 0;
        }
    }
}

// Summation reduction
// Assume that size(A) is a multiple of SUM_RED_TILE_SIZE.
// The summation algorithm needs 2N elems of shared memory to sum N elems because 
// we don't scan in place. So if each thread block operates on 256 elems, then 
// shared memory can support up to 8 thread blocks. If each thread block operates 
// on 256 elems, then the first scan will reduce the number of elems to 64, 256, or 1K.
// THREADS_PER_BLOCK must be a power of 2. The kernels assume that THREADS_PER_BLOCK
// is a power of 2, which allows them to use logical shifts rather than division and
// multiplication in many cases.

//#define SUM_RED_TPB 128
#define SUM_RED_TPB 128 //Changing it to 32 makes it work for size 384
#define SUM_RED_TILE_SIZE (2 * SUM_RED_TPB)

// cudaTreeSummation: Performs one level of tree-based summation reduction on a
// vector.
//  vec (input):  Input vector
//  vec (output): Output vector
//  numElems (input): Number of input elements
    __global__ void
cudaTreeSummation(float *vec, int numElems) 
{
    // Use shared mem to hold the packed input        subtree nodes (elems 0
    //                 ... 1*SUM_RED_TILE_SIZE-1).
    //                        the packed intermediate subtree nodes (elems
    // SUM_RED_TILE_SIZE ... 2*SUM_RED_TILE_SIZE-2)
    __shared__ float sTemp[2 * SUM_RED_TILE_SIZE - 1];

    // Read tile (SUM_RED_TILE_SIZE values) from vec into shared memory
    {
        int vecIndex = blockIdx.y * (gridDim.x * SUM_RED_TILE_SIZE) + 
                       blockIdx.x * SUM_RED_TILE_SIZE + threadIdx.x;
        if (vecIndex < numElems)
            sTemp[threadIdx.x] = vec[vecIndex];
        else
            sTemp[threadIdx.x] = 0.0f;

        vecIndex += SUM_RED_TPB;
        if (vecIndex < numElems)
            sTemp[threadIdx.x + SUM_RED_TPB] = vec[vecIndex];
        else
            sTemp[threadIdx.x + SUM_RED_TPB] = 0.0f;
    }
    __syncthreads();

    // Compute the sum of all the elements in sTemp.
    // Pack the intermediate sums in sTemp.
    int nextNumElems = SUM_RED_TPB;
    int levelBase = 0;
    int nextLevelBase = SUM_RED_TILE_SIZE;
    for (int stride = 1; stride < SUM_RED_TILE_SIZE; stride <<= 1) {
        if (threadIdx.x < nextNumElems) {
            int levelIndex = levelBase + 2 * threadIdx.x;
            sTemp[nextLevelBase + threadIdx.x] = sTemp[levelIndex] + 
                                                 sTemp[levelIndex + 1];
            levelBase = nextLevelBase;
            nextLevelBase += nextNumElems;
            nextNumElems >>= 1;
        }
        __syncthreads();
    }

    // Write the sum out to memory
    if (threadIdx.x == 0) {
        vec[blockIdx.x] = sTemp[2 * SUM_RED_TILE_SIZE - 2];
    }
}


// float* cudaVec is a CUDA pointer
// cudaVec will be modified by SummationReduction(.)
    float 
SummationReduction(float *cudaVec, int numElems) 
{
    dim3 threads(SUM_RED_TPB, 1);

    // Init
    int curElems = 0;
    float *the_Vec = NULL;//pad cudaVec

    if(isPowerOfTwo(numElems)) {
        curElems = numElems;
        the_Vec = cudaVec;
    }
    else {
        curElems = getLeastPowerOfTwo(numElems);
        CUDA_SAFE_CALL(cudaMalloc((void**)&the_Vec,curElems*sizeof(float)));
        CUDA_SAFE_CALL(cudaMemset((void*)the_Vec,0,curElems*sizeof(float)));
        CUDA_SAFE_CALL(cudaMemcpy(the_Vec,cudaVec,numElems*sizeof(float),cudaMemcpyDeviceToDevice));
    }
    

    // *************** TREE-BASED SUM: MOVE FROM LEAVES TO ROOT ***************
    while (curElems > SUM_RED_TILE_SIZE) {
        assert((curElems % SUM_RED_TILE_SIZE) == 0);

        int xBlocks = curElems / SUM_RED_TILE_SIZE;
        int yBlocks = 1;
        while (xBlocks > 32768) {
            yBlocks <<= 1;
            if (xBlocks % 2) {
                xBlocks >>= 1;
                xBlocks++;
            } else {
                xBlocks >>= 1;
            }
        }
        dim3 blocks(xBlocks, yBlocks);

        #if 0 //DEBUG
        printf("Launching cudaTreeSummation: Blocks (x=%d, y=%d), Threads \
                (x=%d, y=%d), Elems %d, Src %x\n", blocks.x, blocks.y, 
                threads.x, threads.y, curElems, the_Vec);
        #endif

        cudaTreeSummation <<< blocks, threads >>> (the_Vec, curElems);
        curElems = blocks.x * blocks.y;
    }

    float *fSum = (float *)calloc(curElems, sizeof(float));
    cudaMemcpy(fSum, the_Vec, curElems * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int x = 0; x < curElems; x++)
        sum += fSum[x];

    if(fSum)
      free(fSum);
    if(the_Vec!=cudaVec) {
       //If the_Vec points to a different memory 
       //section than cudaVec, then free it.
       CUDA_SAFE_CALL(cudaFree(the_Vec));
    }
    return sum;
}

#define GPU_MULT_TRANS_TPB 256

// We're 'transposing B in place', so set Brows = B.jc, Bcols = B.ir, numRows =
// B.numCols
    __global__ void
CudaMultTransC(
    cufftComplex *a, cufftComplex *c, float *Bvals, 
    int *Brows, int *Bcols, int numRows) 
{
    int rowIndex = blockIdx.y * (gridDim.x * GPU_MULT_TRANS_TPB) + 
                   blockIdx.x * GPU_MULT_TRANS_TPB + threadIdx.x;

    int col_start = Brows[rowIndex];
    int col_stop  = Brows[rowIndex + 1];

    if (col_start != col_stop) {
        float real = 0.0f;
        float imag = 0.0f;

        for (int col = col_start; col < col_stop; col++) {
            float cReal = c[Bcols[col]] REAL;
            float cImag = c[Bcols[col]] IMAG;

            float BReal = Bvals[col];

            real += BReal * cReal;
            imag += BReal * cImag;
        }

        a[rowIndex] REAL = real;
        a[rowIndex] IMAG = imag;
    }
}

    __global__ void
CudaMultTransR(
    cufftComplex *a, cufftComplex *c, float *Bvals, 
    int *Brows, int *Bcols, int numRows) 
{
    int rowIndex = blockIdx.y * (gridDim.x * GPU_MULT_TRANS_TPB) + 
                   blockIdx.x * GPU_MULT_TRANS_TPB + threadIdx.x;

    int col_start = Brows[rowIndex];
    int col_stop  = Brows[rowIndex + 1];

    if (col_start != col_stop) {
        float real = 0.0f;
        for (int col = col_start; col < col_stop; col++) {
            float cReal = c[Bcols[col]] REAL;
            float BReal = Bvals[col];
            real += BReal * cReal;
        }

        a[rowIndex] REAL = real;
        a[rowIndex] IMAG = 0.0f;
    }
}

// TODO: Possible optis
// 1. Change the row->thread mapping so that threads in the same warp tend to operate
// on rows that have roughly the same number of nonzero elements. Then we'll have less 
// divergence in the for loop.
// 2. Divide the B*c operation into chunks, and load the c vector into constant
//    or shared memory to improve locality.

// a = transpose(B) * c
// a, c are column vectors
// B is a sparse matrix
    void 
cuda_mult_trans(matrix_t&a, sparse_matrix_t&B, matrix_t&c)
{
    // Each CUDA thread computes one element of the output vector.
    // Thus, each CUDA thread reads one row of transpose(B) and
    //  one column of c. There is only one column of c, so we
    //  have some data locality. Unfortunately, that data locality
    //  is difficult to exploit.

    // Assumes that the number of elements in the output vector is
    //  a multiple of GPU_MULT_TRANS_TPB
    assert((a.elems % GPU_MULT_TRANS_TPB) == 0);
    assert(!B.isComplex);
    #if DEBUG
    printf("a (%d x %d), B' (%d x %d), c (%d x %d)\n", a.dims[0], a.dims[1], 
           B.numCols, B.numRows, c.dims[0], c.dims[1]);
    #endif
    dim3 threads(GPU_MULT_TRANS_TPB, 1);
    int xBlocks = a.elems / GPU_MULT_TRANS_TPB;
    int yBlocks = 1;
    while (xBlocks > 32768) {
        yBlocks <<= 1;
        if (xBlocks % 2) {
            xBlocks >>= 1;
            xBlocks++;
        } else {
            xBlocks >>= 1;
        }
    }
    dim3 blocks(xBlocks, yBlocks);

    if (c.isComplex) {
        CudaMultTransC <<< blocks, threads >>> 
                          (a.device, c.device, B.devRVals, 
                           B.devJc, B.devIr, B.numCols);
        #if DEBUG
        a.copy_to_host();
        #endif
        a.isComplex = 1;
    } else {
        CudaMultTransR <<< blocks, threads >>> 
                          (a.device, c.device, B.devRVals, 
                           B.devJc, B.devIr, B.numCols);
        #if DEBUG
        a.copy_to_host();
        #endif
        a.isComplex = 0;
    }
}

#define GPU_AGG_OP1_TILE_SIZE 256

    __global__ void
CudaAggOp1CC(
    cufftComplex *Src1, cufftComplex *Src2, 
    cufftComplex *Dst, float *result, float Src1Scale) 
{
    int index = blockIdx.y * (gridDim.x * GPU_AGG_OP1_TILE_SIZE) + 
                blockIdx.x * GPU_AGG_OP1_TILE_SIZE + threadIdx.x;
    Dst[index] REAL = Src1Scale * Src1[index] REAL + Src2[index] REAL;
    Dst[index] IMAG = Src1Scale * Src1[index] IMAG + Src2[index] IMAG;
    result[index] = Dst[index] REAL * Dst[index] REAL + 
                    Dst[index] IMAG * Dst[index] IMAG;
}

    __global__ void
CudaAggOp1RC(
    cufftComplex *Src1, cufftComplex *Src2, 
    cufftComplex *Dst, float *result, float Src1Scale) 
{
    int index = blockIdx.y * (gridDim.x * GPU_AGG_OP1_TILE_SIZE) + 
                blockIdx.x * GPU_AGG_OP1_TILE_SIZE + threadIdx.x;
    Dst[index] REAL = Src1Scale * Src1[index] REAL + Src2[index] REAL;
    Dst[index] IMAG = Src2[index] IMAG;
    result[index] = Dst[index] REAL * Dst[index] REAL + 
                    Dst[index] IMAG * Dst[index] IMAG;
}

    __global__ void
CudaAggOp1CR(
    cufftComplex *Src1, cufftComplex *Src2, 
    cufftComplex *Dst, float *result, float Src1Scale) 
{
    int index = blockIdx.y * (gridDim.x * GPU_AGG_OP1_TILE_SIZE) + 
                blockIdx.x * GPU_AGG_OP1_TILE_SIZE + threadIdx.x;
    Dst[index] REAL = Src1Scale * Src1[index] REAL + Src2[index] REAL;
    Dst[index] IMAG = Src1Scale * Src1[index] IMAG;
    result[index] = Dst[index] REAL * Dst[index] REAL + 
                    Dst[index] IMAG * Dst[index] IMAG;
}

// dst = src1 + src1scale * src2
// dst = real(dst .* dst)
    void 
cuda_agg_op1(
    matrix_t&src1, matrix_t&src2, matrix_t&dst, 
    float *result, float Src1Scale, int elems) 
{
    dim3 threads(GPU_AGG_OP1_TILE_SIZE, 1);
    int xBlocks = elems / GPU_AGG_OP1_TILE_SIZE;
    int yBlocks = 1;
    while (xBlocks > 32768) {
        yBlocks <<= 1;
        if (xBlocks % 2) {
            xBlocks >>= 1;
            xBlocks++;
        } else {
            xBlocks >>= 1;
        }
    }
    dim3 blocks(xBlocks, yBlocks);

    if (src1.isComplex) {
        if (src2.isComplex) {
            CudaAggOp1CC <<< blocks, threads >>> 
                            (src1.device, src2.device, dst.device, 
                             result, Src1Scale);
            dst.isComplex = 1;
        } else {
            CudaAggOp1CR <<< blocks, threads >>> 
                            (src1.device, src2.device, dst.device, 
                             result, Src1Scale);
            dst.isComplex = 1;
        }
    } else {
        if (src2.isComplex) {
            CudaAggOp1RC <<< blocks, threads >>> 
                            (src1.device, src2.device, dst.device, 
                             result, Src1Scale);
            dst.isComplex = 1;
        } else {
            printf("Error: AggOp1 of two real-valued vectors not implemented\n"); 
            exit(1);
        }
    }
}

#define GPU_AGG_OP2_TILE_SIZE 256

// ASSUME FOR NOW THAT W IS REAL-VALUED
    __global__ void
CudaAggOp2C(
    cufftComplex *a, cufftComplex *c, cufftComplex *d, float *Bvals, 
    int *Brows, int *Bcols, int numRows) 
{
    int rowIndex = blockIdx.y * (gridDim.x * GPU_AGG_OP2_TILE_SIZE) + 
                   blockIdx.x * GPU_AGG_OP2_TILE_SIZE + threadIdx.x;

    int col_start = Brows[rowIndex];
    int col_stop  = Brows[rowIndex + 1];

    if (col_start != col_stop) {
        float real = 0.0f;
        float imag = 0.0f;

        for (int col = col_start; col < col_stop; col++) {
            float cReal = c[Bcols[col]] REAL;
            float cImag = c[Bcols[col]] IMAG;

            float BReal = Bvals[col];

            real += BReal * cReal;
            imag += BReal * cImag;
        }

        // dot product
        a[rowIndex] REAL = real * d[rowIndex] REAL;
        a[rowIndex] IMAG = imag * d[rowIndex] REAL;
    }
}

    __global__ void
CudaAggOp2R(
    cufftComplex *a, cufftComplex *c, cufftComplex *d, 
    float *Bvals, int *Brows, int *Bcols, int numRows) 
{
    int rowIndex = blockIdx.y * (gridDim.x * GPU_AGG_OP2_TILE_SIZE) + 
                   blockIdx.x * GPU_AGG_OP2_TILE_SIZE + threadIdx.x;

    int col_start = Brows[rowIndex];
    int col_stop  = Brows[rowIndex + 1];

    if (col_start != col_stop) {
        float real = 0.0f;
        for (int col = col_start; col < col_stop; col++) {
            float cReal = c[Bcols[col]] REAL;
            float BReal = Bvals[col];
            real += BReal * cReal;
        }

        // dot product
        a[rowIndex] REAL = real * d[rowIndex] REAL;
        a[rowIndex] IMAG = 0.0f;
    }
}

// a = transpose(B) * c ; a = a .* d
// a, c, d are column vectors
// B is a sparse matrix
    void 
cuda_agg_op2(matrix_t&a, sparse_matrix_t&B, matrix_t&c, matrix_t&d)
{
    // Each CUDA thread computes one element of the output vector.
    // Thus, each CUDA thread reads one row of transpose(B) and
    //  one column of c. There is only one column of c, so we
    //  have some data locality. Unfortunately, that data locality
    //  is difficult to exploit.

    // Assumes that the number of elements in the output vector is
    //  a multiple of GPU_AGG_OP2_TILE_SIZE
    assert((a.elems % GPU_AGG_OP2_TILE_SIZE) == 0);
    assert(!B.isComplex);
    #ifdef DEBUG
    printf("a (%d x %d), B' (%d x %d), c (%d x %d)\n", a.dims[0], a.dims[1], 
            B.numCols, B.numRows, c.dims[0], c.dims[1]);
    #endif
    dim3 threads(GPU_AGG_OP2_TILE_SIZE, 1);
    int xBlocks = a.elems / GPU_AGG_OP2_TILE_SIZE;
    int yBlocks = 1;
    while (xBlocks > 32768) {
        yBlocks <<= 1;
        if (xBlocks % 2) {
            xBlocks >>= 1;
            xBlocks++;
        } else {
            xBlocks >>= 1;
        }
    }
    dim3 blocks(xBlocks, yBlocks);

    if (c.isComplex) {
        CudaAggOp2C <<< blocks, threads >>> 
                       (a.device, c.device, d.device, B.devRVals, 
                        B.devJc, B.devIr, B.numCols);
        a.isComplex = 1;
    } else {
        CudaAggOp2R <<< blocks, threads >>> 
                       (a.device, c.device, d.device, B.devRVals, 
                        B.devJc, B.devIr, B.numCols);
        a.isComplex = 0;
    }
}

    static void 
inputData_recon(
    char *input_folder_path,
    float&version, int&numK, int&numK_per_coil,
    int&ncoils, int&nslices, int&numX, int&numX_per_coil,
    float *&kx, float *&ky, float *&kz,
    float *&x, float *&y, float *&z,
    float *&fm, float *&t,
    float *&sensi_r, float *&sensi_i,
    float *&phiR, float *&phiI)
{
  	// Test data format version (0.2 or 1.0 higher)
    string test_version_fn = input_folder_path;
    test_version_fn += "/kx.dat";
    FILE *fp0 = fopen(test_version_fn.c_str(),"r");
	if(NULL==fp0) {
		printf("%s not found!\n",test_version_fn.c_str());
		exit(1);
	}
	float the_version = -1.0f;
	//the_version could be 0.2 or 1.0 higher
	if(1!=fread(&the_version,sizeof(float),1,fp0)) {
		printf("Error: fread return value mismatch\n");
	    exit(1);
	}
	fclose(fp0);


    string kz_fn = input_folder_path;
    kz_fn = kz_fn + "/kz.dat";
    string ky_fn = input_folder_path;
    ky_fn = ky_fn + "/ky.dat";
    string kx_fn = input_folder_path;
    kx_fn = kx_fn + "/kx.dat";
    string iz_fn = input_folder_path;
    iz_fn = iz_fn + "/iz.dat";
    string iy_fn = input_folder_path;
    iy_fn = iy_fn + "/iy.dat";
    string ix_fn = input_folder_path;
    ix_fn = ix_fn + "/ix.dat";
    string phiR_fn = input_folder_path;
    phiR_fn = phiR_fn + "/phiR.dat";
    string phiI_fn = input_folder_path;
    phiI_fn = phiI_fn + "/phiI.dat";
    string t_fn = input_folder_path;
    t_fn = t_fn + "/t.dat";
    string fm_fn = input_folder_path;
    fm_fn = fm_fn + "/fm.dat";
    string sensi_r_fn = input_folder_path;
    sensi_r_fn = sensi_r_fn + "/sensi_r.dat";
    string sensi_i_fn = input_folder_path;
    sensi_i_fn = sensi_i_fn + "/sensi_i.dat";

    if(0.2f==the_version) {
      kz = readDataFile_JGAI(kz_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      ky = readDataFile_JGAI(ky_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      kx = readDataFile_JGAI(kx_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);

      //numX_per_coil=numX for all below but sense map.
      z = readDataFile_JGAI(iz_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);
      y = readDataFile_JGAI(iy_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);
      x = readDataFile_JGAI(ix_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);

      phiR = readDataFile_JGAI(phiR_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      phiI = readDataFile_JGAI(phiI_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
 
      t = readDataFile_JGAI(t_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      fm = readDataFile_JGAI(fm_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);

      sensi_r = readDataFile_JGAI(sensi_r_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);
      sensi_i = readDataFile_JGAI(sensi_i_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);
    }
    else {
      kz = readDataFile_JGAI_10(kz_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      ky = readDataFile_JGAI_10(ky_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      kx = readDataFile_JGAI_10(kx_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);

      //numX_per_coil=numX for all below but sense map.
      z = readDataFile_JGAI_10(iz_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);
      y = readDataFile_JGAI_10(iy_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);
      x = readDataFile_JGAI_10(ix_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);

      phiR = readDataFile_JGAI_10(phiR_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      phiI = readDataFile_JGAI_10(phiI_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
 
      t = readDataFile_JGAI_10(t_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      fm = readDataFile_JGAI_10(fm_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);

      sensi_r = readDataFile_JGAI_10(sensi_r_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);
      sensi_i = readDataFile_JGAI_10(sensi_i_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);
    }

    if (1 == ncoils) {
        for (int i = 0; i < (numX_per_coil * ncoils); i++) {
            sensi_r[i] = 1.0f;
            sensi_i[i] = 0.0f;
        }
    }
}

    void 
CudaFhF(
    matrix_t&image, matrix_t&Qf, matrix_t&FhF, 
    matrix_t&result, cufftHandle&plan)
{
    // TODO: IS THIS REALLY NECESSARY?
    result.clear_host();
    result.copy_to_device();

    // Not sure which dimensions really correspond to rows and cols, 
    // but it shouldn't matter in this case because all dimensions are the same.
        /*3D - JGAI - BEGIN*/
    int N1 = Qf.dims[0] / 2;
    int N2 = Qf.dims[1] / 2;
    int N3;
    if (Qf.dims[2]>1)
    {
       N3 = Qf.dims[2] / 2;
    }
    else if(Qf.dims[2]==1)
    {
       N3 = 1;
    }
        /*3D - JGAI - END*/



    // TODO: Aggregating oversample and fftshift would help, 
    // but indexing schemes make it very difficult
    cuda_oversample(image, result, N1, N2, N3);


    // not sure which dimensions really correspond to rows and cols, 
    // but it shouldn't matter in this case because all dimensions are the same
       /*3D -JGAI - BEGIN*/
    if (Qf.dims[2]>1)
    {
       cuda_fft3shift(result, result, Qf.dims[0], Qf.dims[1], Qf.dims[2], 1);
    }
    else if(Qf.dims[2]==1)
    {
       cuda_fftshift(result, result, Qf.dims[0], Qf.dims[1], 1);
    }
       /*3D - JGAI - END*/



    // execute FFT on device
    cufftExecC2C(plan, result.device, result.device, CUFFT_FORWARD);
    result.isComplex = 1;
    #if DEBUG
    result.copy_to_host();
    #endif

    // dot product
    cuda_dot_prod(result, Qf, result, Qf.elems);
    #if DEBUG
    Qf.copy_to_host();
    result.copy_to_host();
    #endif

    // execute IFFT on device
    // NOTE: Result must be scaled by 1/elems. Scaling occurs in
    // cuda_undersample_scale().
    cufftExecC2C(plan, result.device, result.device, CUFFT_INVERSE);
    result.isComplex = 1;
    #if DEBUG
    result.copy_to_host();
    #endif



    // TODO: Aggregating fftshift and undersample_scale would help, but indexing
    // schemes make it very difficult. Not sure which dimensions really correspond 
    // to rows and cols, but it shouldn't matter in this case because all dimensions 
    // are the same.
       /*3D - JGAI - BEGIN*/
    if(1==Qf.dims[2])
    {
       cuda_fftshift(result, result, Qf.dims[0], Qf.dims[1], 0);
    }
    else if(1<Qf.dims[2])
    {
       cuda_fft3shift(result, result, Qf.dims[0], Qf.dims[1], Qf.dims[2], 0);
    }
       /*3D - JGAI - END*/

    cuda_undersample_scale(result, FhF, N1, N2, N3, 1.0 / Qf.elems);

}

#define KERNEL_HADAMARD_THREADS_PER_BLOCK 256
    __global__ void 
Hadamard_prod_GPU_Kernel(
    cufftComplex *result_d, cufftComplex *in1_d,
    float *in2_r_d, float *in2_i_d, int len)
{
    int index = blockIdx.x * KERNEL_HADAMARD_THREADS_PER_BLOCK + threadIdx.x;
    if (index < len) {
        float in1_r = in1_d[index] REAL;
        float in1_i = in1_d[index] IMAG;
        float in2_r = in2_r_d[index];
        float in2_i = in2_i_d[index];
        float tmp_r = in1_r * in2_r - in1_i * in2_i;
        float tmp_i = in1_r * in2_i + in1_i * in2_r;

        result_d[index] REAL = tmp_r;
        result_d[index] IMAG = tmp_i;
    }
}
    void 
Hadamard_prod_GPU(
    cufftComplex *result_d, cufftComplex *in1_d,
    float *in2_r_d, float *in2_i_d, int len)
{
    int HadamardBlocks = len / KERNEL_HADAMARD_THREADS_PER_BLOCK;
    if (len % KERNEL_HADAMARD_THREADS_PER_BLOCK)
        HadamardBlocks++;
    dim3 DimHadamard_Block(KERNEL_HADAMARD_THREADS_PER_BLOCK, 1);
    dim3 DimHadamard_Grid(HadamardBlocks, 1);

    Hadamard_prod_GPU_Kernel <<< DimHadamard_Grid, DimHadamard_Block >>>
    (result_d, in1_d, in2_r_d, in2_i_d, len);
}

// result = a^H .* b, used by SENSE
    __global__ void 
Hadamard_prod_Conj_GPU_Kernel(
    cufftComplex *result_d, float *a_r_d, 
    float *a_i_d,cufftComplex *b_d, int len)
{
    int index = blockIdx.x * KERNEL_HADAMARD_THREADS_PER_BLOCK + threadIdx.x;
    float tmp_r, tmp_i;
    float a_r = a_r_d[index];
    float a_i = a_i_d[index];
    float b_r = b_d[index] REAL;
    float b_i = b_d[index] IMAG;
    if (index < len) {
        tmp_r = a_r * b_r + a_i * b_i;
        tmp_i = a_r * b_i - a_i * b_r;

        result_d[index] REAL = tmp_r;
        result_d[index] IMAG = tmp_i;
    }
}

    void 
Hadamard_prod_Conj_GPU(
    cufftComplex *result_d, float *a_r_d, 
    float *a_i_d, cufftComplex *b_d, int len)
{
    int HadamardBlocks = len / KERNEL_HADAMARD_THREADS_PER_BLOCK;
    if (len % KERNEL_HADAMARD_THREADS_PER_BLOCK)
        HadamardBlocks++;
    dim3 DimHadamard_Block(KERNEL_HADAMARD_THREADS_PER_BLOCK, 1);
    dim3 DimHadamard_Grid(HadamardBlocks, 1);

    Hadamard_prod_Conj_GPU_Kernel <<< DimHadamard_Grid, DimHadamard_Block >>>
    (result_d, a_r_d, a_i_d, b_d, len);
}

// expfm = $e^{i w[n] (\tau l + t0)}, where n = 1,...,N$
#define KERNEL_EXP_FM_THREADS_PER_BLOCK 256
    __global__ static void 
ComputeExpFM_GPU_Kernel(
    float *expfm_r_d, float *expfm_i_d, float *fm_d, 
    float l, float tau, float t0, int len)
{
    int index = blockIdx.x * KERNEL_EXP_FM_THREADS_PER_BLOCK + threadIdx.x;
    if (index < len) {
        float expArg = (fm_d[index] * (tau * l + t0));
        expfm_r_d[index] = cosf(expArg);
        expfm_i_d[index] = sinf(expArg);
    }
}
// expfm = $e^{-1.0f * i * w[n] * (\tau l + t0)}, where n = 1,...,N$
    __global__ static void 
ComputeExpFM_GPU_Conj_Kernel(
    float *expfm_r_d, float *expfm_i_d, float *fm_d, float l, 
    float tau, float t0, int len)
{
    int index = blockIdx.x * KERNEL_EXP_FM_THREADS_PER_BLOCK + threadIdx.x;
    if (index < len) {
        float expArg = (-1.0f * fm_d[index] * (tau * l + t0));
        expfm_r_d[index] = cosf(expArg);
        expfm_i_d[index] = sinf(expArg);
    }
}

    void 
ComputeExpFM_GPU( 
    float *expfm_r_d, float *expfm_i_d, float *fm_d, float l, 
    float tau, float t0, int len, int conj = 0 )
{
    int ExpFMBlocks = len / KERNEL_EXP_FM_THREADS_PER_BLOCK;
    if (len % KERNEL_EXP_FM_THREADS_PER_BLOCK)
        ExpFMBlocks++;
    dim3 DimExpFM_Block(KERNEL_EXP_FM_THREADS_PER_BLOCK, 1);
    dim3 DimExpFM_Grid(ExpFMBlocks, 1);

    if (0 == conj) {
        ComputeExpFM_GPU_Kernel <<< DimExpFM_Grid, DimExpFM_Block >>>
        (expfm_r_d, expfm_i_d, fm_d, l, tau, t0, len);
    } else {
        ComputeExpFM_GPU_Conj_Kernel <<< DimExpFM_Grid, DimExpFM_Block >>>
        (expfm_r_d, expfm_i_d, fm_d, l, tau, t0, len);
    }
}

#define ADD_GPU_THREADS_PER_BLOCKS 256
    __global__ void 
add_GPU_Kernel(
    cufftComplex *result_d, cufftComplex *a_d, cufftComplex *b_d, int len)
{
    int index = blockIdx.x * ADD_GPU_THREADS_PER_BLOCKS + threadIdx.x;
    if (index < len) {
        result_d[index] REAL = a_d[index] REAL + b_d[index] REAL;
        result_d[index] IMAG = a_d[index] IMAG + b_d[index] IMAG;
    }
}
    void 
add_GPU(
    cufftComplex *result_d, cufftComplex *a_d, cufftComplex *b_d, int len)
{
    int addBlocks = len / ADD_GPU_THREADS_PER_BLOCKS;
    if (len % ADD_GPU_THREADS_PER_BLOCKS)
        addBlocks++;
    dim3 DimAdd_Block(ADD_GPU_THREADS_PER_BLOCKS, 1);
    dim3 DimAdd_Grid(addBlocks, 1);

    add_GPU_Kernel <<< DimAdd_Grid, DimAdd_Block >>>
    (result_d, a_d, b_d, len);
}

#define SET_ZERO_THREADS_PER_BLOCK 256
    __global__ void 
setZero_GlobalMemory_Kernel(cufftComplex *input, int len)
{
    int index = blockIdx.x * SET_ZERO_THREADS_PER_BLOCK + threadIdx.x;
    if (index < len) {
        input[index] REAL = 0.0f;
        input[index] IMAG = 0.0f;
    }
}
    void 
setZero_GlobalMemory(cufftComplex *input, int len)
{
    int setZeroBlocks = len / SET_ZERO_THREADS_PER_BLOCK;
    if (len % SET_ZERO_THREADS_PER_BLOCK)
        setZeroBlocks++;
    dim3 DimsetZero_Block(SET_ZERO_THREADS_PER_BLOCK, 1);
    dim3 DimsetZero_Grid(setZeroBlocks, 1);

    setZero_GlobalMemory_Kernel <<< DimsetZero_Grid, DimsetZero_Block >>>
    (input, len);
}

    void 
CudaAhA(
    matrix_t&II, matrix_t&image, matrix_t *Qf, float *expfm_r_d, 
    float *expfm_i_d, float *fm_d, int L, float tau, float t0, 
    matrix_t&AhA, matrix_t&FhF, matrix_t&tmp3, matrix_t&result, 
    cufftHandle&plan)
{
    int conjYes = 1;
    int conjNo = 0;
    int numX_per_single_coil;

       /*3D - JGAI - BEGIN*/ 
    if(1==Qf[0].dims[2]) //2D
    {
       numX_per_single_coil = Qf[0].elems / 4;
    }
    else if(1<Qf[0].dims[2]) // 3D
    {
       numX_per_single_coil = Qf[0].elems / 8;
    }
       /*3D - JGAI - END*/ 

    setZero_GlobalMemory(AhA.device, numX_per_single_coil);

    //ERASEME:
    //CUDA_SAFE_CALL( cudaMemset( (void*)fm_d, 0, numX_per_single_coil * sizeof(float) ) );

    for (int l = 0; l <= ((int) L); l++) {
        ComputeExpFM_GPU(expfm_r_d, expfm_i_d, fm_d, l, tau, 
                         t0, numX_per_single_coil, conjYes);

        Hadamard_prod_GPU(II.device, image.device, expfm_r_d, 
                          expfm_i_d, numX_per_single_coil);

        //aim for more memory efficiency
        Qf[l].alloc_device();
        Qf[l].copy_to_device();

        CudaFhF(II, Qf[l], FhF, result, plan);

        Qf[l].copy_to_host();
        Qf[l].free_device();

        ComputeExpFM_GPU(expfm_r_d, expfm_i_d, fm_d, l, tau, t0, 
                         numX_per_single_coil, conjNo);

        Hadamard_prod_GPU(tmp3.device, FhF.device, expfm_r_d, 
                          expfm_i_d, numX_per_single_coil);

        add_GPU(AhA.device, AhA.device, tmp3.device, numX_per_single_coil);
    }
    AhA.isComplex = 1;
}

    void 
parCudaAhA(
    matrix_t&II, matrix_t&image, matrix_t *Qf, float *expfm_r_d, 
    float *expfm_i_d,float *fm_d, int L, float tau, float t0, 
    matrix_t&AhA, matrix_t&FhF, matrix_t&tmp3, matrix_t&result, 
    cufftHandle&plan, // new parameters for SENSE:
    matrix_t&sensed_Rp, matrix_t&sensed_AhA, matrix_t&par_AhA,
    float *sensi_r_d, float *sensi_i_d, int ncoils)
{
    int numX_per_single_coil;

       /*3D - JGAI - BEGIN*/ 
    if(1==Qf[0].dims[2]) //2D
    {
       numX_per_single_coil = Qf[0].elems / 4;
    }
    else if(1<Qf[0].dims[2]) // 3D
    {
       numX_per_single_coil = Qf[0].elems / 8;
    }
       /*3D - JGAI - END*/ 
    setZero_GlobalMemory(par_AhA.device, numX_per_single_coil);

    for (int c = 0; c < ncoils; c++) {
        Hadamard_prod_GPU(sensed_Rp.device, image.device, 
                          sensi_r_d + c * numX_per_single_coil,
                          sensi_i_d + c * numX_per_single_coil,  
                          numX_per_single_coil);

        CudaAhA(II, sensed_Rp, Qf, expfm_r_d, expfm_i_d, fm_d,
                L, tau, t0, AhA, FhF, tmp3, result, plan);

        Hadamard_prod_Conj_GPU(sensed_AhA.device, sensi_r_d + 
                               c * numX_per_single_coil,
                               sensi_i_d + c * numX_per_single_coil, 
                               AhA.device, numX_per_single_coil);

        add_GPU(par_AhA.device, par_AhA.device, sensed_AhA.device, 
                numX_per_single_coil);
    }
    par_AhA.isComplex = 1;
}

    void 
Computeq(
    float lambda2, matrix_t *Qf, matrix_t&W, matrix_t&p,
    sparse_matrix_t&D, sparse_matrix_t&Dp, sparse_matrix_t&R, matrix_t&q,
    matrix_t&Rp, matrix_t&FhF, matrix_t&tmp1, matrix_t&tmp2, matrix_t&result,
    cufftHandle&plan, matrix_t&II, float *expfm_r_d, float *expfm_i_d, float *fm_d,
    int L, float tau, float t0, matrix_t&AhA, matrix_t&tmp3,
    // new parameters for SENSE:
    matrix_t&sensed_Rp, matrix_t&sensed_AhA, matrix_t&par_AhA,
    float *sensi_r_d, float *sensi_i_d, int ncoils,
    // new parameters for Total Variation
    float *w,
    // new parameters for choosing the regularization between sparse matrix
    // and finite difference. COO format is used to pass the sparse matrix in.
    // The actual computations are done in CSR format. COO2CSR conversion is
    // done inside this Computeq as well.
    const bool enable_regularization, const bool enable_finite_difference,
    const CooMatrix *c )
{
    Rp.clear_host();
    tmp1.clear_host(); tmp1.copy_to_device();
    tmp2.clear_host(); tmp2.isComplex=1;// IMPORTANT: isComplex is VERY confusing as a member.
    FhF.clear_host(); FhF.copy_to_device();

    // TODO: This op leverages the fact that R = R'. Is that always true?
    // Rp = R * p
    //Jiading GAI
    //cuda_mult_trans(Rp, R, p);
    Rp.assign_device(&p);
    p.copy_to_host();
    Rp.copy_to_host();
/*
    #ifdef GEFORCE_8800

    // tmp1 = D * Rp
    Dp.alloc_device(); Dp.copy_to_device();
    cuda_mult_trans(tmp1, Dp, Rp);
    Dp.free_device();

    // tmp1 = W .* tmp1
    W.alloc_device(); W.copy_to_device();
    cuda_dot_prod(tmp1, W, tmp1, W.elems);
    cudaMemcpy(tmp1.host, tmp1.device, W.elems * sizeof(cufftComplex), 
               cudaMemcpyDeviceToHost);
    W.free_device();

    // tmp2 = Dp * tmp1
    D.alloc_device(); D.copy_to_device();
    cuda_mult_trans(tmp2, D, tmp1);
    D.free_device();

    #else

    // tmp1 = D * Rp
    // tmp1 = W .* tmp1
    cuda_agg_op2(tmp1, Dp, Rp, W);
    cudaMemcpy(tmp1.host, tmp1.device, W.elems * sizeof(cufftComplex), 
               cudaMemcpyDeviceToHost);

    // tmp2 = Dp * tmp1
    cuda_mult_trans(tmp2, D, tmp1);

    #endif
*/


    // FhF = CudaFhF(Rp, Qf)
    // Jiading GAI - test it with brute force FhF.
    // CudaFhF2(Rp, Qf, FhF, result, plan);
    //CudaFhF(Rp, Qf, FhF, result, plan);
    parCudaAhA(II, Rp, Qf, expfm_r_d, expfm_i_d, fm_d, L, tau, t0,
               AhA, FhF, tmp3, result, plan,
               sensed_Rp, sensed_AhA, par_AhA,
               sensi_r_d, sensi_i_d, ncoils);


    if (enable_regularization) {
        CsrMatrix c_csr, c_trans_csr;

        // Convert matrix C to CSR format.
        c_csr = mtx2Csr(c->I, c->J, c->V,
                        c->num_rows, c->num_cols, c->num_nonzeros);

        // Convert transposed matrix C_trans to CSR format.
        // In fact, we just need to switch the row and column indices.
        c_trans_csr = mtx2Csr(c->J, c->I, c->V,
                              c->num_cols, c->num_rows, c->num_nonzeros);

        int *c_Ap_d = NULL, *c_Aj_d = NULL;
        FLOAT_T *c_Ax_d = NULL;
        int *c_trans_Ap_d = NULL, *c_trans_Aj_d = NULL;
        FLOAT_T *c_trans_Ax_d = NULL;

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

        mriCopyHostToDevice<int>    (c_Ap_d, c_csr.Ap, num_Ap);
        mriCopyHostToDevice<int>    (c_Aj_d, c_csr.Aj, num_Aj);
        mriCopyHostToDevice<FLOAT_T>(c_Ax_d, c_csr.Ax, num_Ax);
        mriCopyHostToDevice<int>(c_trans_Ap_d, c_trans_csr.Ap, num_trans_Ap);
        mriCopyHostToDevice<int>(c_trans_Aj_d, c_trans_csr.Aj, num_trans_Aj);
        mriCopyHostToDevice<FLOAT_T>(c_trans_Ax_d, c_trans_csr.Ax,
                                      num_trans_Ax);

        int num_i = Rp.elems;
        FLOAT_T *Cf_r_d = NULL;
        FLOAT_T *Cf_i_d = NULL;
        if(Rp.dims[2]==1) {
           Cf_r_d = mriNewGpu<FLOAT_T>(2*num_i);
           Cf_i_d = mriNewGpu<FLOAT_T>(2*num_i);
        }
        else {
           Cf_r_d = mriNewGpu<FLOAT_T>(3*num_i);
           Cf_i_d = mriNewGpu<FLOAT_T>(3*num_i);
        }
        FLOAT_T *CtCf_r    = mriNewCpu<FLOAT_T>(  num_i);
        FLOAT_T *CtCf_i    = mriNewCpu<FLOAT_T>(  num_i);
        FLOAT_T *CtCf_r_d  = mriNewGpu<FLOAT_T>(  num_i);
        FLOAT_T *CtCf_i_d  = mriNewGpu<FLOAT_T>(  num_i);
        FLOAT_T *Rp_r      = mriNewCpu<FLOAT_T>(  num_i);
        FLOAT_T *Rp_i      = mriNewCpu<FLOAT_T>(  num_i);
        FLOAT_T *Rp_r_d    = mriNewGpu<FLOAT_T>(  num_i);
        FLOAT_T *Rp_i_d    = mriNewGpu<FLOAT_T>(  num_i);
         
        Rp.copy_to_host(); 
        for(int i=0;i<num_i;i++)
        {
          Rp_r[i] = Rp.host[i].x;
          Rp_i[i] = Rp.host[i].y;
        }
        mriCopyHostToDevice<FLOAT_T>(Rp_r_d, Rp_r, num_i);
        mriCopyHostToDevice<FLOAT_T>(Rp_i_d, Rp_i, num_i);


        if(Rp.dims[2]==1) {
           smvmGpu(Cf_r_d, Cf_i_d, Rp_r_d, Rp_i_d, c_Ap_d, c_Aj_d, c_Ax_d, 2*num_i);
        } 
        else {
           smvmGpu(Cf_r_d, Cf_i_d, Rp_r_d, Rp_i_d, c_Ap_d, c_Aj_d, c_Ax_d, 3*num_i);
        }

        FLOAT_T *zv_temp = mriNewGpu<FLOAT_T>(num_i);
        cutilSafeCall(
           cudaMemset(zv_temp, 0, num_i*sizeof(FLOAT_T))
        );
        pointMultGpu(Cf_r_d, Cf_i_d, Cf_r_d, Cf_i_d, w, zv_temp, num_i);

        pointMultGpu(Cf_r_d+num_i, Cf_i_d+num_i, Cf_r_d+num_i, 
                     Cf_i_d+num_i, w, zv_temp, num_i);

        
        if(Rp.dims[2]>1) {
           pointMultGpu(Cf_r_d+2*num_i, Cf_i_d+2*num_i, Cf_r_d+2*num_i, 
                        Cf_i_d+2*num_i, w, zv_temp, num_i);
        }

        smvmGpu(CtCf_r_d, CtCf_i_d, Cf_r_d, Cf_i_d, c_trans_Ap_d,
                c_trans_Aj_d, c_trans_Ax_d, num_i);
        
        // lambda2 is fd_penalizer on the Brute Force side. 
        addGpu(CtCf_r_d, CtCf_i_d, zv_temp, zv_temp, CtCf_r_d, 
               CtCf_i_d, lambda2, num_i);


        mriCopyDeviceToHost<FLOAT_T>(CtCf_r, CtCf_r_d, num_i);
        mriCopyDeviceToHost<FLOAT_T>(CtCf_i, CtCf_i_d, num_i);
        for(int i=0;i<num_i;i++)
        {
          tmp2.host[i].x = CtCf_r[i]; 
          tmp2.host[i].y = CtCf_i[i]; 
        }
        tmp2.copy_to_device();


        mriDeleteGpu(zv_temp);

        mriDeleteGpu(c_Ap_d);
        mriDeleteGpu(c_Aj_d);
        mriDeleteGpu(c_Ax_d);

        mriDeleteGpu(c_trans_Ap_d);
        mriDeleteGpu(c_trans_Aj_d);
        mriDeleteGpu(c_trans_Ax_d);

        mriDeleteGpu(Cf_r_d);
        mriDeleteGpu(Cf_i_d);
        mriDeleteGpu(CtCf_r_d);
        mriDeleteGpu(CtCf_i_d);
        mriDeleteGpu(Rp_r_d);
        mriDeleteGpu(Rp_i_d);
        
        mriDeleteCpu(Rp_r);
        mriDeleteCpu(Rp_i);
        mriDeleteCpu(CtCf_r);
        mriDeleteCpu(CtCf_i);

    } else {
       // s = (D^H w D)p
       // fd_penalizer: regularization weight
       if(Rp.dims[2]==1) {
         DHWD2dGpu(tmp2, Rp, w, lambda2);
       } 
       else {
       //  cout << lambda2;
         DHWD3dGpu(tmp2, Rp, w, lambda2);
       }
    }

    // FhF = FhF + lambda2*tmp2
    cuda_add_scale(tmp2, par_AhA, par_AhA, 1.0f, par_AhA.elems);

    // q = R' * FhF
    // TODO: Is clearing q really necessary? Is there a faster way to do it?
    q.clear_host();
    q.copy_to_device();
    // Jiading GAI
    //cuda_mult_trans(q, R, FhF);
    q.assign_device(&par_AhA);
    q.copy_to_host();
}

//******************************************************************************
// Inputs:
//   int numRestarts - number of outer loop iterations
//   int numIterMax - max number of inner loop iterations
// Data structures:
//   float Q[2*N1][2*N2][2*N3]
//   float W[6340608]
//   float Fhd[N1*N2*N3]
//   float RhFhd[N1*N2*N3]
//   float x[N1*N2*N3]
//   float p[N1*N2*N3]
//   float r[N1*N2*N3]
//   float D[6340608][N1*N2*N3], sparse
//   float Dp[N1*N2*N3][6340608], sparse
//   float R[N1*N2*N3][N1*N2*N3], sparse
//******************************************************************************

// We're taking advantage of the fact that
// (1) R = R'
// (2) D = Dp'
// Justin says those conditions are likely to hold true in most cases.
// The conditions for those conditions to hold true are
// (1) D/R/Dp are real-valued, or
// (2) The 'prime' operation is a Hermitian (transpose + complex-conjugate)

int toeplitz_recon(int toe_argc, char *toe_argv[], 
                   const bool enable_total_variation, 
                   const bool enable_regularization,
                   const bool enable_finite_difference,
                   const bool enable_reuseQ,
                   const CooMatrix *c,
                   float *Fhd_R, float *Fhd_I,
                   float **Q_R, float **Q_I)
{

    float time_segment_num = atof(toe_argv[16]);
    float tau;
    float version;
    int ncoils, nslices;
    int numK_per_coil, numX_per_coil, numX, numK;

    printf("======== Step 3. Start CG solver ========\n");

    matrix_t Q[((int) time_segment_num) + 1],  Qf[((int) time_segment_num) + 1], 
             II, Rp, sensed_Rp, W, Fhd, RhFhd, p, x, r, tmp1, tmp2, tmp3, tmp7, 
             AhA, sensed_AhA, par_AhA, FhF, q, reconBest, result;
    sparse_matrix_t D, Dp, R;

    float *ix, *iy, *iz, *phiR, *phiI, *fm, *sensi_r, *sensi_i;
    float *kx, *ky, *kz, *t;
    float *fm_d, /* *t_d, */ *sensi_r_d, *sensi_i_d, *expfm_r_d, *expfm_i_d;

    // Is Q complex for symmetric trajectories? No. For now, we're not exploiting
    // that. Fix for journal.
    // Is Fhd complex for symmetric trajectories? Yes.

    float alpha       = 0.0f;
    float beta        = 0.0f;
    float beta2       = 0.0f;
    float resBest     = 0.0f;
    float normRhFhd   = 0.0f;
    // For noise = .01: 21.1% error at lambda2 = .0001
    // For noise = .03: 26.1% error at lambda2 = .0002
    float lambda2      = 10.0f; //0.000001f;
    unsigned int tv_num;
    float tol         = 1.0E-16;
    int numRestarts = 0;
    int numIterMax  = 0;

    int N1 = 0;
    int N2 = 0;
    int N3 = 0;
    int symTraj = 0;

    // check for proper number of input and output arguments
    if (toe_argc != 18) {
        printf("Usage: recon N1 N2 N3 numRestarts numIterMax symTraj lambda2 \
                D.file Dp.file R.file W.file Q.file FhD.file data_directory \
                tv_num ntime_segments out.file\n");
        //exit(1);
    }

    inputData_recon(toe_argv[14], version, numK, numK_per_coil, ncoils, nslices, 
                    numX, numX_per_coil, kx, ky, kz, ix, iy, iz, fm, t, sensi_r, 
                    sensi_i, phiR, phiI);

    tau = (time_segment_num > 0.0f) ? ((t[numK_per_coil - 1] - t[0]) / time_segment_num) 
          : (t[numK_per_coil - 1] - t[0]);
    float t0 = t[0];

    //t_d = mriNewGpu<float>(numK_per_coil);
    //mriCopyHostToDevice<float>(t_d, t, numK_per_coil);
    fm_d = mriNewGpu<float>(numX_per_coil);
    mriCopyHostToDevice<float>(fm_d, fm, numX_per_coil);
    sensi_r_d = mriNewGpu<float>(numX_per_coil * ncoils);
    mriCopyHostToDevice<float>(sensi_r_d, sensi_r, numX_per_coil * ncoils);
    sensi_i_d = mriNewGpu<float>(numX_per_coil * ncoils);
    mriCopyHostToDevice<float>(sensi_i_d, sensi_i, numX_per_coil * ncoils);

    expfm_r_d = mriNewGpu<float>(numX_per_coil);
    //mriCopyHostToDevice<float>(expfm_r_d, expfm_r, numX_per_coil);
    expfm_i_d = mriNewGpu<float>(numX_per_coil);
    //mriCopyHostToDevice<float>(expfm_i_d, expfm_i, numX_per_coil);

    N1 = atoi(toe_argv[1]);//N1 = Ny
    N2 = atoi(toe_argv[2]);//N2 = Nx
    N3 = atoi(toe_argv[3]);//N3 = Nz
    numRestarts = atoi(toe_argv[4]);
    numIterMax = atoi(toe_argv[5]);
    symTraj = atoi(toe_argv[6]);
    lambda2 = atof(toe_argv[7]);
    tv_num = atoi(toe_argv[15]);

    // --------------- Initialize host data structures ---------------
    D.init_JGAI(N1 * N2, N1 * N2, 0, 0); //D.read(argv[8]); - deprecated
    D.isComplex = 0;
    Dp.init_JGAI(N1 * N2, N1 * N2, 0, 0); //Dp.read(argv[9]); - deprecated
    Dp.isComplex = 0;

    R.init_JGAI(N1 * N2, N1 * N2, N1 * N2, N1 * N2); //R.read(argv[10]); - deprecated
    R.isComplex = 0;
    for (int i = 0; i < N1 * N2; i++) {
        R.rVals[i] = 1.0;
        R.iVals[i] = 0.0;
    }
    R.jc[0] = 0;
    for (int i = 1; i < (N1 * N2 + 1); i++) {
        R.jc[i] = R.jc[i - 1] + 1;
    }
    for (int i = 0; i < (N1 * N2); i++) {
        R.ir[i] = i;
    }

    W.init2D(N1, N2); //W.read(argv[11]); - deprecated

    for (int l = 0; l <= ((int) time_segment_num); l++) {
            /*3D - JGAI - BEGIN*/
        if(N3>1)
        {
           Q[l].init3D(2*N1, 2*N2, 2*N3);
        }
        else if (1==N3)
        {
           Q[l].init3D(2*N1, 2*N2, 1);
        }
            /*3D - JGAI - END*/
        if (enable_reuseQ) {
            Q[l].read_pack(toe_argv[12], symTraj, l);
        }
        else {//avoid read/write Q matrix from disk
            Q[l].isComplex = 1;
            for (int i = 0; i < Q[l].elems; i++) {
                 Q[l].host[i]REAL = Q_R[l][i];
                 Q[l].host[i]IMAG = Q_I[l][i];
            }
        }
        Qf[l].init(Q[l].numDims, Q[l].dims, Q[l].elems);
    }

    //variables like Fhd are always treated as a 3D data,
    //because 2D data is also 3D with N3=1.
    Fhd.init3D(N1, N2, N3);
    #if 0
    Fhd.read_pack(toe_argv[13], 0);
    #else //avoid read/write from disk.
    Fhd.isComplex = 1;
    for (int i = 0; i < Fhd.elems; i++) {
        Fhd.host[i]REAL = Fhd_R[i];
        Fhd.host[i]IMAG = Fhd_I[i];
    }
    #endif

    x.init3D(N1, N2, N3); // initial guess = zeros(N1*N2*N3, 1)
    reconBest.init(x.numDims, x.dims, x.elems);

    tmp1.init(W.numDims, W.dims, W.elems); // - deprecated
        /*3D - JGAI - BEGIN */
    tmp2.init3D(N1, N2, N3);
    q.init3D(N1, N2, N3);
        /*3D - JGAI - END */
       

    RhFhd.init(Fhd.numDims, Fhd.dims, Fhd.elems);
    p.init(Fhd.numDims, Fhd.dims, Fhd.elems);
    r.init(Fhd.numDims, Fhd.dims, Fhd.elems);
    II.init(Fhd.numDims, Fhd.dims, Fhd.elems);
    Rp.init(Fhd.numDims, Fhd.dims, Fhd.elems);
    sensed_Rp.init(Fhd.numDims, Fhd.dims, Fhd.elems);
    AhA.init(Fhd.numDims, Fhd.dims, Fhd.elems);
    sensed_AhA.init(Fhd.numDims, Fhd.dims, Fhd.elems);
    par_AhA.init(Fhd.numDims, Fhd.dims, Fhd.elems);
    FhF.init(Fhd.numDims, Fhd.dims, Fhd.elems);
    tmp3.init(Fhd.numDims, Fhd.dims, Fhd.elems);

    tmp7.init(q.numDims, q.dims, q.elems);

    // TODO: Why is result 2*N1 by 2*N2 by 2*N3? Why not N1 by N2 by N3?
    result.init(Qf[0].numDims, Qf[0].dims, Qf[0].elems);
    // ---------------------------------------------------------------

    // --------------- Initialize device data structures ---------------
    #ifndef GEFORCE_8800
    W.alloc_device(); W.copy_to_device();
    D.alloc_device(); D.copy_to_device();
    Dp.alloc_device(); Dp.copy_to_device();
    #endif
    for (int l = 0; l <= ((int) time_segment_num); l++) {
    //    Qf[l].alloc_device();
    //    Qf[l].copy_to_device();
    }
    II.alloc_device();
    Rp.alloc_device();
    sensed_Rp.alloc_device();
    FhF.alloc_device(); FhF.copy_to_device();
    AhA.alloc_device(); AhA.copy_to_device(); // Why copy_to_device(.)
    sensed_AhA.alloc_device(); sensed_AhA.copy_to_device();
    par_AhA.alloc_device(); par_AhA.copy_to_device();
    tmp1.alloc_device();
    tmp2.alloc_device();
    tmp3.alloc_device();
    tmp7.alloc_device();
    result.alloc_device();
    q.alloc_device();
    p.alloc_device();
    x.alloc_device(); x.copy_to_device();
    r.alloc_device();
    reconBest.alloc_device();
    // -----------------------------------------------------------------

    float *dotprod_d;
    dotprod_d = mriNewGpu<float>(p.elems);


    cufftHandle plan;
    if(1<Q[0].dims[2]) {//3D
           cufftPlan3d(&plan, Q[0].dims[0], Q[0].dims[1], Q[0].dims[2], CUFFT_C2C);
    }
    else if(1==Q[0].dims[2]) {//2D
           cufftPlan2d(&plan, Q[0].dims[0], Q[0].dims[1], CUFFT_C2C);
    }
    for (int l = 0; l <= ((int) time_segment_num); l++) {
        // Q = ifftshift(Q)
        Q[l].alloc_device();
        Q[l].copy_to_device();

        Qf[l].alloc_device();//aim for memory efficiency
        Qf[l].copy_to_device();
            /*3D - JGAI - BEGIN*/
        if(1<Q[l].dims[2]) // 3D
        {
           cuda_fft3shift(Q[l], Qf[l], Q[l].dims[0], Q[l].dims[1], Q[l].dims[2], 1);
        }
        else if(1==Q[l].dims[2]) // 2D
        {
           cuda_fftshift(Q[l], Qf[l], Q[l].dims[0], Q[l].dims[1], 1);
        }
        Q[l].free_device(); // Q is never accessed again

           /*3D - JGAI - END*/

        cufftExecC2C(plan, Qf[l].device, Qf[l].device, CUFFT_FORWARD);
        Qf[l].isComplex = 1;

        Qf[l].copy_to_host();
        Qf[l].free_device();
    }

    // init reconBest = x
    reconBest.assign_device(&x);

    // RhFhd = R' * Fhd
    R.alloc_device(); R.copy_to_device();
    Fhd.alloc_device(); Fhd.copy_to_device();
    RhFhd.alloc_device();

    // Jiading GAI
    //cuda_mult_trans(RhFhd, R, Fhd);
    RhFhd.assign_device(&Fhd);
    RhFhd.copy_to_host();
   //RhFhd.writeB("RhFhd.file");

    // normRhFhd = real(RhFhd' * RhFhd)
    cuda_dot_prod_real(RhFhd, RhFhd, dotprod_d, RhFhd.elems);
    normRhFhd = SummationReduction(dotprod_d, RhFhd.elems);

    Fhd.free_device(); // Fhd is never accessed again

    // TV - weight matrix
    float *w = mriNewGpu<float>(numX_per_coil);
    float *w_tmp = (float *) malloc(numX_per_coil * sizeof(float));
    for (int i = 0; i < numX_per_coil; i++)
        w_tmp[i] = 1.0f;
    mriCopyHostToDevice(w, w_tmp, numX_per_coil);
    free(w_tmp);

    matrix_t dhori, dverti, dzaxis;
    dhori.init(Fhd.numDims, Fhd.dims, Fhd.elems);
    dverti.init(Fhd.numDims, Fhd.dims, Fhd.elems);
    dhori.alloc_device();  dhori.copy_to_device();
    dverti.alloc_device(); dverti.copy_to_device();
    if(Fhd.dims[2]>1) {
       // 3D derivative
       dzaxis.init(Fhd.numDims, Fhd.dims, Fhd.elems);
       dzaxis.alloc_device();  dzaxis.copy_to_device();
    }
    // END - TV - weight matrix


    unsigned int timerApp = 0;
    (cutCreateTimer(&timerApp));
    (cutStartTimer(timerApp));

    int restart = 0; // no restart on CG
    while (restart < numRestarts) {
        for (int tv_i = 0; tv_i < tv_num; tv_i++) {
        #ifdef DEBUG    
            printf("TV iteration ... # %d and lambda2 = %e\n", tv_i, lambda2);
        #else
            if(enable_total_variation) {
              msg(MSG_PLAIN, "  GPU: TV: %d", tv_i);
              msg(MSG_PLAIN, "\n");
            }
        #endif
            if (tv_i != 0) {
                if(x.dims[2]==1) {
                   Dverti2dGpu(dverti, x);
                   Dhori2dGpu(dhori,  x);
                   multiply2dGpu(w, dhori, dverti);
                } else {
                   Dverti3dGpu(dverti, x);
                   Dhori3dGpu(dhori,  x);
                   Dzaxis3dGpu(dzaxis,  x);
                   multiply3dGpu(w, dhori, dverti, dzaxis);
                }
                 
            }

            Computeq(lambda2, Qf, W, x, D, Dp, R, tmp7, Rp, FhF, tmp1, tmp2, 
                     result, plan, II, expfm_r_d, expfm_i_d, fm_d, time_segment_num,
                     tau, t0, AhA, tmp3, sensed_Rp, sensed_AhA, par_AhA, 
                     sensi_r_d, sensi_i_d, ncoils, w, enable_regularization, 
                     enable_finite_difference, c);

            // r = RhFhd - tmp7
            cuda_sub(RhFhd, tmp7, r, r.elems);


            if (restart == 0) {
                // beta2 = resBest = real(r'*r)
                cuda_dot_prod_real(r, r, dotprod_d, r.elems);
                resBest = SummationReduction(dotprod_d, r.elems);
                beta2 = resBest;
            }

            #ifdef DEBUG
            #else
              msg(MSG_PLAIN, "  GPU: CG: 0.");
            #endif
            int iter = 0;
            // p = r
            p.assign_device(&r);
            while (iter < numIterMax) {
                Computeq(lambda2, Qf, W, p, D, Dp, R, q, Rp, FhF, tmp1, tmp2, 
                         result, plan, II, expfm_r_d, expfm_i_d, fm_d, time_segment_num, 
                         tau, t0, AhA, tmp3, sensed_Rp, sensed_AhA, par_AhA, 
                         sensi_r_d, sensi_i_d, ncoils, w, enable_regularization, 
                         enable_finite_difference, c);

            #ifdef DEBUG
            p.writeB("p.file");
            #endif

                beta = beta2;

                // alpha = beta / real(p'*q)
                float sum = 0.0f;
                cuda_dot_prod_real(p, q, dotprod_d, p.elems);
                sum = SummationReduction(dotprod_d, p.elems);
                alpha = beta / sum;

                #ifdef DEBUG
                  printf("iter = %02d | beta = %e and sum = %e and alpha = %e\n",
                          iter, beta, sum, alpha);
                  x.writeB("x0.file");
                #else
                  msg(MSG_PLAIN, "%d.", iter+1);
                #endif

                

                // x = x + alpha * p
                cuda_add_scale(p, x, x, alpha, x.elems);

                #ifdef DEBUG
                x.writeB("x1.file");
                p.writeB("p.file");
                #endif

                // r = r - alpha * q
                // dotprod_d = r .* r
                cuda_agg_op1(q, r, r, dotprod_d, -alpha, r.elems);

                // beta2 = real(r'*r)
                beta2 = SummationReduction(dotprod_d, r.elems);

                // p = r + (beta2/beta)*p
                cuda_add_scale(p, r, p, beta2 / beta, p.elems);

                if (beta2 < resBest) {
                    resBest = beta2;
                    reconBest.assign_device(&x);
                }

                if (sqrtf(beta2 / normRhFhd) < tol) {
                    break;
                }

                // copy device mem to host mem for inputs to next loop iteration
                cudaMemcpy(p.host, p.device, p.elems * sizeof(cufftComplex), 
                           cudaMemcpyDeviceToHost);

                // Be wary of copying host to device in mid-loop (may overwrite
                // previous iter's results)
                // Investigate each array to determine whether it should be cleared
                // at bottom of loop.
                // Assumptions about variable being cleared are not well-documented.
                iter++;
            }
            msg(MSG_PLAIN, "done.\n");
            if (!enable_total_variation) { // No TV.
                 break;
            }
        } // end of for (tv_i...) 
        restart++;
    }



    (cutStopTimer(timerApp));
    printf("  GPU: CG Execution Time: %f (ms)\n", cutGetTimerValue(timerApp));
    (cutDeleteTimer(timerApp));

    cufftDestroy(plan);
    mriDeleteGpu(dotprod_d);

    // output reconBest
    #ifdef DEBUG
    printf("======== output ========\n");
    #else
      msg(MSG_PLAIN, "  GPU: Exporting results.\n");
    #endif
    float *real, *imag;
    cudaMemcpy(reconBest.host, reconBest.device, reconBest.elems * 
               sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    real = (float *)calloc(reconBest.elems, sizeof(float));
    imag = (float *)calloc(reconBest.elems, sizeof(float));
    for (int i = 0; i < reconBest.elems; i++) {
        real[i] = reconBest.host[i] REAL;
        imag[i] = reconBest.host[i] IMAG;
    }

    FILE *fid = fopen(toe_argv[17], "w");
    fwrite(real, sizeof(float), reconBest.elems, fid);
    fwrite(imag, sizeof(float), reconBest.elems, fid);
    free(real);
    free(imag);
    fclose(fid);

    free(kx);
    free(ky);
    free(kz);
    free(ix);
    free(iy);
    free(iz);
    free(fm);
    free(t);
    free(sensi_r);
    free(sensi_i);
    free(phiR);
    free(phiI);
    //free(expfm_r);
    //free(expfm_i);

    //mriDeleteGpu(t_d);
    mriDeleteGpu(fm_d);
    mriDeleteGpu(sensi_r_d);
    mriDeleteGpu(sensi_i_d);
    mriDeleteGpu(expfm_r_d);
    mriDeleteGpu(expfm_i_d);
    mriDeleteGpu(w);

    return 0;
}
