#ifndef GRIDDING_UTILS_CUH
#define GRIDDING_UTILS_CUH
#include <cufft.h>

   void 
deinterleave_data2d(
   cufftComplex *src, float *outR_d, float *outI_d, 
   int imageX, int imageY);

   void 
deinterleave_data3d(
   cufftComplex *src, float *outR_d, float *outI_d, 
   int imageX, int imageY, int imageZ);

  void
deapodization2d(
  cufftComplex *dst,cufftComplex *src,  
  int imageX, int imageY, 
  float kernelWidth, float beta, float gridOS);

  void
deapodization3d(
  cufftComplex *dst,cufftComplex *src,  
  int imageX, int imageY, int imageZ, 
  float kernelWidth, float beta, float gridOS);

   void 
crop_center_region2d(
   cufftComplex *dst, cufftComplex *src, 
   int imageSizeX, int imageSizeY,
   int gridSizeX, int gridSizeY);

   void 
crop_center_region3d(
   cufftComplex *dst, cufftComplex *src, 
   int imageSizeX, int imageSizeY, int imageSizeZ,
   int gridSizeX, int gridSizeY, int gridSizeZ);

   void 
cuda_fft2shift_grid(
   cufftComplex *src, cufftComplex *dst, 
   int dimY, int dimX, int inverse);

   void 
cuda_fft3shift_grid(
   cufftComplex *src, cufftComplex *dst, 
   int dimY, int dimX, int dimZ, int inverse);
#endif
