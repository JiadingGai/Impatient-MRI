#include <fftshift.cuh>
#include <utils.h>


    __global__ void
CudaFFTShift(cufftComplex *A, int N, int M, int Cols, cufftComplex *Shifted) 
{
    int col = blockIdx.x * FFTSHIFT_TILE_SIZE_X + threadIdx.x;
    int row = blockIdx.y * FFTSHIFT_TILE_SIZE_Y + threadIdx.y;
    #if USE_BUG // Buggy when A and Shifted pointing to the same memory addr.
    Shifted[(row + N) * Cols + (col + M)] REAL = A[(row) * Cols + (col)] REAL;
    Shifted[(row + N) * Cols + (col + M)] IMAG = A[(row) * Cols + (col)] IMAG;
    Shifted[(row) * Cols + (col)] REAL = A[(row + N) * Cols + (col + M)] REAL;
    Shifted[(row) * Cols + (col)] IMAG = A[(row + N) * Cols + (col + M)] IMAG;
    Shifted[(row + N) * Cols + (col)] REAL = A[(row) * Cols + (col + M)] REAL;
    Shifted[(row + N) * Cols + (col)] IMAG = A[(row) * Cols + (col + M)] IMAG;
    Shifted[(row) * Cols + (col + M)] REAL = A[(row + N) * Cols + (col)] REAL;
    Shifted[(row) * Cols + (col + M)] IMAG = A[(row + N) * Cols + (col)] IMAG;
    #else
    float temp1_r = A[(row) * Cols + (col)] REAL;
    float temp1_i = A[(row) * Cols + (col)] IMAG;
    float temp2_r = A[(row + N) * Cols + (col + M)] REAL;
    float temp2_i = A[(row + N) * Cols + (col + M)] IMAG;
    Shifted[(row + N) * Cols + (col + M)] REAL = temp1_r;
    Shifted[(row + N) * Cols + (col + M)] IMAG = temp1_i;
    Shifted[(row) * Cols + (col)] REAL = temp2_r;
    Shifted[(row) * Cols + (col)] IMAG = temp2_i;
    temp1_r = A[(row) * Cols + (col + M)] REAL;
    temp1_i = A[(row) * Cols + (col + M)] IMAG;
    temp2_r = A[(row + N) * Cols + (col)] REAL;
    temp2_i = A[(row + N) * Cols + (col)] IMAG;
    Shifted[(row + N) * Cols + (col)] REAL = temp1_r;
    Shifted[(row + N) * Cols + (col)] IMAG = temp1_i;
    Shifted[(row) * Cols + (col + M)] REAL = temp2_r;
    Shifted[(row) * Cols + (col + M)] IMAG = temp2_i;
    #endif
}

    __global__ void
CudaFFT3Shift(
    cufftComplex *src, cufftComplex *dst, int pivotY, 
    int pivotX, int pivotZ, int dimX, int dimZ) 
{
    int dY = blockIdx.x;
    int dX = blockIdx.y;
    int dZ = threadIdx.x;

    float temp1_r = 
    src[(dY         ) * dimX * dimZ + (dX         ) * dimZ + (dZ         )] REAL;
    float temp1_i = 
    src[(dY         ) * dimX * dimZ + (dX         ) * dimZ + (dZ         )] IMAG;
    float temp2_r = 
    src[(dY + pivotY) * dimX * dimZ + (dX + pivotX) * dimZ + (dZ + pivotZ)] REAL;
    float temp2_i = 
    src[(dY + pivotY) * dimX * dimZ + (dX + pivotX) * dimZ + (dZ + pivotZ)] IMAG;
 
    dst[(dY + pivotY) * dimX * dimZ + (dX + pivotX) * dimZ + (dZ + pivotZ)] REAL = temp1_r;
    dst[(dY + pivotY) * dimX * dimZ + (dX + pivotX) * dimZ + (dZ + pivotZ)] IMAG = temp1_i; 
    dst[(dY         ) * dimX * dimZ + (dX         ) * dimZ + (dZ         )] REAL = temp2_r; 
    dst[(dY         ) * dimX * dimZ + (dX         ) * dimZ + (dZ         )] IMAG = temp2_i;
 
    temp1_r =
    src[(dY         ) * dimX * dimZ + (dX         ) * dimZ + (dZ + pivotZ)] REAL;
    temp1_i =
    src[(dY         ) * dimX * dimZ + (dX         ) * dimZ + (dZ + pivotZ)] IMAG;
    temp2_r = 
    src[(dY + pivotY) * dimX * dimZ + (dX + pivotX) * dimZ + (dZ         )] REAL;
    temp2_i = 
    src[(dY + pivotY) * dimX * dimZ + (dX + pivotX) * dimZ + (dZ         )] IMAG;
    
    dst[(dY + pivotY) * dimX * dimZ + (dX + pivotX) * dimZ + (dZ         )] REAL = temp1_r;
    dst[(dY + pivotY) * dimX * dimZ + (dX + pivotX) * dimZ + (dZ         )] IMAG = temp1_i;
    dst[(dY         ) * dimX * dimZ + (dX         ) * dimZ + (dZ + pivotZ)] REAL = temp2_r;
    dst[(dY         ) * dimX * dimZ + (dX         ) * dimZ + (dZ + pivotZ)] IMAG = temp2_i;

    temp1_r = 
    src[(dY         ) * dimX * dimZ + (dX + pivotX) * dimZ + (dZ         )] REAL;
    temp1_i = 
    src[(dY         ) * dimX * dimZ + (dX + pivotX) * dimZ + (dZ         )] IMAG;
    temp2_r = 
    src[(dY + pivotY) * dimX * dimZ + (dX         ) * dimZ + (dZ + pivotZ)] REAL;
    temp2_i =
    src[(dY + pivotY) * dimX * dimZ + (dX         ) * dimZ + (dZ + pivotZ)] IMAG;

    dst[(dY + pivotY) * dimX * dimZ + (dX         ) * dimZ + (dZ + pivotZ)] REAL = temp1_r;
    dst[(dY + pivotY) * dimX * dimZ + (dX         ) * dimZ + (dZ + pivotZ)] IMAG = temp1_i;
    dst[(dY         ) * dimX * dimZ + (dX + pivotX) * dimZ + (dZ         )] REAL = temp2_r;
    dst[(dY         ) * dimX * dimZ + (dX + pivotX) * dimZ + (dZ         )] IMAG = temp2_i;

    temp1_r = 
    src[(dY         ) * dimX * dimZ + (dX + pivotX) * dimZ + (dZ + pivotZ)] REAL;
    temp1_i =
    src[(dY         ) * dimX * dimZ + (dX + pivotX) * dimZ + (dZ + pivotZ)] IMAG;
    temp2_r = 
    src[(dY + pivotY) * dimX * dimZ + (dX         ) * dimZ + (dZ         )] REAL;
    temp2_i = 
    src[(dY + pivotY) * dimX * dimZ + (dX         ) * dimZ + (dZ         )] IMAG;

    dst[(dY + pivotY) * dimX * dimZ + (dX         ) * dimZ + (dZ         )] REAL = temp1_r;
    dst[(dY + pivotY) * dimX * dimZ + (dX         ) * dimZ + (dZ         )] IMAG = temp1_i;
    dst[(dY         ) * dimX * dimZ + (dX + pivotX) * dimZ + (dZ + pivotZ)] REAL = temp2_r;
    dst[(dY         ) * dimX * dimZ + (dX + pivotX) * dimZ + (dZ + pivotZ)] IMAG = temp2_i;

}

