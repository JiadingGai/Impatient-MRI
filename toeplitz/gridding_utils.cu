#include <gridding_utils.cuh>
#include <fftshift.cuh>
#include <utils.h>
#include <assert.h>

   __global__ static void
Deinterleave_data2d_kernel(
   cufftComplex *src, float *outR_d, float *outI_d, 
   int imageX, int imageY)
{
   int Y = blockIdx.x;
   int X = blockIdx.y;

   int lIndex = X + Y*imageX;
   outR_d[lIndex] = src[lIndex] REAL;
   outI_d[lIndex] = src[lIndex] IMAG;
}


   void 
deinterleave_data2d(
   cufftComplex *src, float *outR_d, float *outI_d, 
   int imageX, int imageY)
{
   dim3 threads(1,1);
   dim3 blocks(imageY,imageX);

   Deinterleave_data2d_kernel<<<blocks,threads>>>
     (src,outR_d,outI_d,imageX,imageY);
}

   __global__ static void
Deinterleave_data3d_kernel(
   cufftComplex *src, float *outR_d, float *outI_d, 
   int imageX, int imageY, int imageZ)
{
   int Z = threadIdx.x;
   int Y = blockIdx.x;
   int X = blockIdx.y;

   int lIndex = Z + X*imageZ + Y*imageZ*imageX;
   outR_d[lIndex] = src[lIndex] REAL;
   outI_d[lIndex] = src[lIndex] IMAG;
}


   void 
deinterleave_data3d(
   cufftComplex *src, float *outR_d, float *outI_d, 
   int imageX, int imageY, int imageZ)
{
   dim3 threads(imageZ,1);
   dim3 blocks(imageY,imageX);

   Deinterleave_data3d_kernel<<<blocks,threads>>>
     (src,outR_d,outI_d,imageX,imageY,imageZ);
}


   __global__ static void
Deapodization2d_kernel(
  cufftComplex *dst, cufftComplex *src,  
  int imageX, int imageY, 
  float kernelWidth, float beta, float gridOS)
{
   /*Justin's gridding code:
     [kernelX kernelY kernelZ] =meshgrid([-Nx/2:Nx/2-1]/Nx,
                                         [-Ny/2:Ny/2-1]/Ny,
                                         [-Nz/2:Nz/2-1]/Nz);
    gridKernel = (sin(sqrt(pi^2*kernelWidth^2*kernelX.^2 - beta^2))./ ...
                      sqrt(pi^2*kernelWidth^2*kernelX.^2 - beta^2)).*
                 (sin(sqrt(pi^2*kernelWidth^2*kernelY.^2 - beta^2))./
                      sqrt(pi^2*kernelWidth^2*kernelY.^2 - beta^2)).*
                 (sin(sqrt(pi^2*kernelWidth^2*kernelZ.^2 - beta^2))./
                      sqrt(pi^2*kernelWidth^2*kernelZ.^2 - beta^2));
   */
   int imageNumElems = imageX * imageY;

   int Y = blockIdx.x;
   int X = blockIdx.y;

   float gridKernelY = float(Y - (imageY/2)) / (float)imageY; 
   float gridKernelX = float(X - (imageX/2)) / (float)imageX;

   float common_exprX = (PI*PI*kernelWidth*kernelWidth*gridKernelX*gridKernelX - beta*beta);
   float common_exprY = (PI*PI*kernelWidth*kernelWidth*gridKernelY*gridKernelY - beta*beta);

   float common_exprX1;
   float common_exprY1;

   if(common_exprX>=0)
      common_exprX1 = (sin(sqrt(common_exprX))/sqrt(common_exprX));
   else
      common_exprX1 = (sinh(sqrt(-1.0f*common_exprX))/sqrt(-1.0f*common_exprX));
 
   if(common_exprY>=0)
      common_exprY1 = (sin(sqrt(common_exprY))/sqrt(common_exprY));
   else
      common_exprY1 = (sinh(sqrt(-1.0f*common_exprY))/sqrt(-1.0f*common_exprY));
   
   float gridKernel =  common_exprX1 * common_exprY1;

   if(gridKernel==gridKernel)
   {
     int common_index = X + Y*imageX;
     float gridOS2 = gridOS * gridOS;
     //dst[common_index] REAL = ((float)imageNumElems * src[common_index]REAL) / gridKernel;
     //dst[common_index] IMAG = ((float)imageNumElems * src[common_index]IMAG) / gridKernel;
     dst[common_index] REAL = (src[common_index]REAL) / gridKernel * (1.0f / gridOS2);
     dst[common_index] IMAG = (src[common_index]IMAG) / gridKernel * (1.0f / gridOS2);
   }
}

  void
deapodization2d(
  cufftComplex *dst,cufftComplex *src,  
  int imageX, int imageY, 
  float kernelWidth, float beta, float gridOS)
{
   assert( (!(imageX%2) && !(imageY%2)) );

   dim3 threads(1,1);
   dim3 blocks(imageY, imageX);

   Deapodization2d_kernel<<<blocks,threads>>>
      (dst,src,imageX,imageY,kernelWidth,beta,gridOS);   
}

   __global__ static void
Deapodization3d_kernel(
  cufftComplex *dst, cufftComplex *src,  
  int imageX, int imageY, int imageZ, 
  float kernelWidth, float beta, float gridOS)
{
   /*Justin's gridding code:
     [kernelX kernelY kernelZ] =meshgrid([-Nx/2:Nx/2-1]/Nx,
                                         [-Ny/2:Ny/2-1]/Ny,
                                         [-Nz/2:Nz/2-1]/Nz);
    gridKernel = (sin(sqrt(pi^2*kernelWidth^2*kernelX.^2 - beta^2))./ ...
                      sqrt(pi^2*kernelWidth^2*kernelX.^2 - beta^2)).*
                 (sin(sqrt(pi^2*kernelWidth^2*kernelY.^2 - beta^2))./
                      sqrt(pi^2*kernelWidth^2*kernelY.^2 - beta^2)).*
                 (sin(sqrt(pi^2*kernelWidth^2*kernelZ.^2 - beta^2))./
                      sqrt(pi^2*kernelWidth^2*kernelZ.^2 - beta^2));
   */
   int imageNumElems = imageX * imageY * imageZ;

   int Z = threadIdx.x;
   int Y = blockIdx.x;
   int X = blockIdx.y;

   float gridKernelZ = (float(Z) - ((float)imageZ/2.0f)) / (float)imageZ;
   float gridKernelY = (float(Y) - ((float)imageY/2.0f)) / (float)imageY; 
   float gridKernelX = (float(X) - ((float)imageX/2.0f)) / (float)imageX;

   float common_exprX = (PI*PI*kernelWidth*kernelWidth*gridKernelX*gridKernelX - beta*beta);
   float common_exprY = (PI*PI*kernelWidth*kernelWidth*gridKernelY*gridKernelY - beta*beta);
   float common_exprZ = (PI*PI*kernelWidth*kernelWidth*gridKernelZ*gridKernelZ - beta*beta);

   float common_exprX1;
   float common_exprY1;
   float common_exprZ1;

   if(common_exprX>=0)
      common_exprX1 = (sin(sqrt(common_exprX))/sqrt(common_exprX));
   else
      common_exprX1 = (sinh(sqrt(-1.0f*common_exprX))/sqrt(-1.0f*common_exprX));
 
   if(common_exprY>=0)
      common_exprY1 = (sin(sqrt(common_exprY))/sqrt(common_exprY));
   else
      common_exprY1 = (sinh(sqrt(-1.0f*common_exprY))/sqrt(-1.0f*common_exprY));
   
   if(common_exprZ>=0)
      common_exprZ1 = (sin(sqrt(common_exprZ))/sqrt(common_exprZ));
   else
      common_exprZ1 = (sinh(sqrt(-1.0f*common_exprZ))/sqrt(-1.0f*common_exprZ));

   float gridKernel =  common_exprX1 * common_exprY1 * common_exprZ1;

   if(gridKernel==gridKernel)
   {
     int common_index = Z + X*imageZ + Y*imageZ*imageX;
     float gridOS3 = gridOS * gridOS * gridOS;
     //dst[common_index] REAL = ((float)imageNumElems*src[common_index]REAL) / (gridKernel);
     //dst[common_index] IMAG = ((float)imageNumElems*src[common_index]IMAG) / (gridKernel);
     dst[common_index] REAL = (src[common_index]REAL) / gridKernel * (1.0f / gridOS3);
     dst[common_index] IMAG = (src[common_index]IMAG) / gridKernel * (1.0f / gridOS3);
   }
}

  void
deapodization3d(
  cufftComplex *dst,cufftComplex *src,  
  int imageX, int imageY, int imageZ, 
  float kernelWidth, float beta, float gridOS)
{
   assert( (!(imageX%2) && !(imageY%2) && !(imageZ%2)) );

   dim3 threads(imageZ,1);
   dim3 blocks(imageY, imageX);

   Deapodization3d_kernel<<<blocks,threads>>>
      (dst,src,imageX,imageY,imageZ,kernelWidth,beta, gridOS);   
}

   __global__ static void 
CropCenterRegion2d_kernel(
   cufftComplex *dst, cufftComplex *src,  
   int imageSizeX, int imageSizeY, 
   int gridSizeX, int gridSizeY)
{
   int dY_dst = blockIdx.x;
   int dX_dst = blockIdx.y;

   int offsetY = (int)(((float)gridSizeY / 2.0f) - ((float)imageSizeY / 2.0f));
   int offsetX = (int)(((float)gridSizeX / 2.0f) - ((float)imageSizeX / 2.0f));

   int dY_src = dY_dst + offsetY;
   int dX_src = dX_dst + offsetX;

   int common_index_dst = dY_dst*imageSizeX + dX_dst;
   int common_index_src = dY_src*gridSizeX  + dX_src;

   dst[common_index_dst] REAL = src[common_index_src] REAL;
   dst[common_index_dst] IMAG = src[common_index_src] IMAG;
}

   void 
crop_center_region2d(
   cufftComplex *dst, cufftComplex *src,
   int imageSizeX, int imageSizeY, 
   int gridSizeX, int gridSizeY)
{
  /* (gridSizeX,gridSizeY) is the size of 'src' */
  assert( (!(gridSizeX%2) && !(gridSizeY%2) ) );
  assert( (!(imageSizeX%2) && !(imageSizeY%2) ) );

  dim3 threads(1, 1);
  dim3 blocks(imageSizeY, imageSizeX);

  CropCenterRegion2d_kernel<<<blocks,threads>>>
      (dst,src,imageSizeX,imageSizeY,gridSizeX,gridSizeY);
}

   __global__ static void 
CropCenterRegion3d_kernel(
   cufftComplex *dst, cufftComplex *src,  
   int imageSizeX, int imageSizeY, int imageSizeZ,
   int gridSizeX, int gridSizeY, int gridSizeZ)
{
   int dY_dst = blockIdx.x;
   int dX_dst = blockIdx.y;
   int dZ_dst = threadIdx.x;

   int offsetY = (int)(((float)gridSizeY / 2.0f) - ((float)imageSizeY / 2.0f));
   int offsetX = (int)(((float)gridSizeX / 2.0f) - ((float)imageSizeX / 2.0f));
   int offsetZ = (int)(((float)gridSizeZ / 2.0f) - ((float)imageSizeZ / 2.0f));

   int dY_src = dY_dst + offsetY;
   int dX_src = dX_dst + offsetX;
   int dZ_src = dZ_dst + offsetZ;

   int common_index_dst = dY_dst*imageSizeX*imageSizeZ + dX_dst*imageSizeZ + dZ_dst;
   int common_index_src = dY_src*gridSizeX*gridSizeZ   + dX_src*gridSizeZ  + dZ_src;

   dst[common_index_dst] REAL = src[common_index_src] REAL;
   dst[common_index_dst] IMAG = src[common_index_src] IMAG;
}

   void 
crop_center_region3d(
   cufftComplex *dst, cufftComplex *src, 
   int imageSizeX, int imageSizeY, int imageSizeZ,
   int gridSizeX, int gridSizeY, int gridSizeZ)
{
  /* (gridSizeX,gridSizeY,gridSizeZ) is the size of 'src' */
  assert( (!(gridSizeX%2) && !(gridSizeY%2) && !(gridSizeZ%2)) );
  assert( (!(imageSizeX%2) && !(imageSizeY%2) && !(imageSizeZ%2)) );

  dim3 threads(imageSizeZ, 1);
  dim3 blocks(imageSizeY, imageSizeX);

  CropCenterRegion3d_kernel<<<blocks,threads>>>
      (dst, src, imageSizeX, imageSizeY, imageSizeZ, 
       gridSizeX, gridSizeY, gridSizeZ);
}

   void 
cuda_fft2shift_grid(
   cufftComplex *src, cufftComplex *dst, 
   int dimY, int dimX, int inverse)
{  
   //(dimX,dimY) is the size of 'src'
   
   int pivotY = 0;
   int pivotX = 0;
   if(inverse) {
     pivotY = (int)floor(float(dimY / 2));
     pivotX = (int)floor(float(dimX / 2));
   } else {
     pivotY = (int)ceil(float(dimY / 2));
     pivotX = (int)ceil(float(dimX / 2));
   }

    dim3 threads(FFTSHIFT_TILE_SIZE_X, FFTSHIFT_TILE_SIZE_Y);
    dim3 blocks(pivotX / FFTSHIFT_TILE_SIZE_X, pivotX / FFTSHIFT_TILE_SIZE_Y);

    CudaFFTShift<<<blocks, threads>>> 
            (src, pivotY, pivotX, dimX, dst);
}

   void 
cuda_fft3shift_grid(
   cufftComplex *src, cufftComplex *dst, 
   int dimY, int dimX, int dimZ, int inverse)
{  
   //(dimX,dimY,dimZ) is the size of 'src'
   
   int pivotY = 0;
   int pivotX = 0;
   int pivotZ = 0;
   if(inverse) {
     pivotY = (int)floor(float(dimY / 2));
     pivotX = (int)floor(float(dimX / 2));
     pivotZ = (int)floor(float(dimZ / 2));
   } else {
     pivotY = (int)ceil(float(dimY / 2));
     pivotX = (int)ceil(float(dimX / 2));
     pivotZ = (int)ceil(float(dimZ / 2));
   }

    dim3 threads(pivotZ, 1);
    dim3 blocks(pivotY, pivotX);

    CudaFFT3Shift<<<blocks, threads>>> 
            (src, dst, pivotY, pivotX, pivotZ, dimX, dimZ);
}


