#ifndef FFTSHIFT_CUH
#define FFTSHIFT_CUH
#include <cufft.h>
// Assumes that each dimension is a multiple of 16
#define FFTSHIFT_TILE_SIZE_X 1
#define FFTSHIFT_TILE_SIZE_Y 1

    __global__ void
CudaFFTShift(cufftComplex *A, int N, int M, int Cols, cufftComplex *Shifted);

    __global__ void
CudaFFT3Shift(cufftComplex *src, cufftComplex *dst, int pivotY, 
              int pivotX, int pivotZ, int dimX, int dimZ);
#endif
