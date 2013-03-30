#include <stdio.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cufft.h>

#include <utils.h>
#include "UDTypes.h"

#define TILE 64
#define LOG_TILE 6

__constant__ int gridSize_c[3];
__constant__ int imageSize_c[3];
__constant__ int Nxy_grid_c;
__constant__ float gridOS_c;
__constant__ float kernelWidth_c;

///*From Numerical Recipes in C, 2nd Edition
__device__ static float bessi0(float x)
{
// /*
    float ax,ans;
    float y;

    ax = fabs(x);    
    if (ax < 3.75)
    {
        y=x/3.75;
        y=y*y;
        ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492+y*(0.2659732+
            y*(0.360768e-1+y*0.45813e-2)))));
    }
    else
    {
        y=3.75/ax;
        ans=(__expf(ax)*rsqrt(ax))*(0.39894228+y*(0.1328592e-1+y*(0.225319e-2+
             y*(-0.157565e-2+y*(0.916281e-2+y*(-0.2057706e-1+y*(0.2635537e-1+
             y*(-0.1647633e-1+y*0.392377e-2))))))));
    }
    return ans;
// */
}
// */

// t0 is there b/c t.dat does not start with 0.0f.
static    __device__ 
float hanning_d(float tm, float tau, float l, float t0)
{
// /*
    float taul = tau * l;
    float result;
    float x = tm - taul - t0;
    if ( fabs(x) < tau ) {
        result = 0.5f + 0.5f * __cosf(PI * (x) / tau);
    } else {
        result = 0.0f;
    }
    //FIXME:
    //result = 1.0f;
    return result;
// */
}

///*
__global__ void binning_kernel1 (unsigned int n, ReconstructionSample* Sample_g, unsigned int* numPts_g, 
                                unsigned int binsize, unsigned int gridNumElems)
{
//If your GPU has no atomic support, then binning_kernel1 is defined
//to be an empty kernel function. Meanwhile, its job is re-assigned to
//CPU, which is handled by binning_kernel1_CPU(.) in gridding.cpp.
//FIXME:not an elegant solution for selecting between sm_10 and sm_11
#ifndef CUDA_NO_SM_11_ATOMIC_INTRINSICS
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  ReconstructionSample pt;
  unsigned int binIdx;
  unsigned int cap;
  if (idx < n){
    pt = Sample_g[idx];
    pt.kX = (gridOS_c)*(pt.kX+((float)imageSize_c[0])/2.0f);
    pt.kY = (gridOS_c)*(pt.kY+((float)imageSize_c[1])/2.0f);

    if(1!=imageSize_c[2])
    {
      pt.kZ = (gridOS_c)*(pt.kZ+((float)imageSize_c[2])/2.0f);
    }
    else
    {
      pt.kZ = 0.0f;
    }

    binIdx = (unsigned int)(pt.kZ)*Nxy_grid_c + (unsigned int)(pt.kY)*gridSize_c[0] + (unsigned int)(pt.kX);     
    if (numPts_g[binIdx]<binsize){
      cap = atomicAdd(numPts_g+binIdx, 1);
      if (cap >= binsize){
        atomicSub(numPts_g+binIdx, 1);
      }
    }
  }
#endif
}

   __global__ void 
binning_kernel2(unsigned int n, ReconstructionSample* Sample_g, unsigned int* binStartAddr_g, 
                unsigned int* numPts_g, unsigned int binsize, samplePtArray SampleArray_g, 
                unsigned int* CPUbinSize, unsigned int* CPUbin)
{
//If your GPU has no atomic support, then binning_kernel2 is defined
//to be an empty kernel function. Meanwhile, its job is re-assigned to
//CPU, which is handled by binning_kernel2_CPU(.) in gridding.cpp.
//FIXME:not an elegant solution for selecting between sm_10 and sm_11
#ifndef CUDA_NO_SM_11_ATOMIC_INTRINSICS
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  ReconstructionSample pt;
  unsigned int binIdx;
  unsigned int cap;

  if (idx < n){
    pt = Sample_g[idx];
    pt.kX = (gridOS_c)*(pt.kX+((float)imageSize_c[0])/2.0f);
    pt.kY = (gridOS_c)*(pt.kY+((float)imageSize_c[1])/2.0f);
 
    if(1!=imageSize_c[2])
    {
      pt.kZ = (gridOS_c)*(pt.kZ+((float)imageSize_c[2])/2.0f);
    }
    else
    {
      pt.kZ = 0.0f;
    }

    binIdx = (unsigned int)(pt.kZ)*Nxy_grid_c + (unsigned int)(pt.kY)*gridSize_c[0] + (unsigned int)(pt.kX);
    if (numPts_g[binIdx]<binsize){
      cap = atomicAdd(numPts_g+binIdx, 1);
      if (cap < binsize){
        float2 data;
        data.x = pt.real;
        data.y = pt.imag;

        float2 loc1;
        loc1.x = pt.kX;
        loc1.y = pt.kY;

        float2 loc2;
        loc2.x = pt.kZ;
        loc2.y = pt.sdc;

        float2 loc3;
		loc3.x = pt.t;
		//loc3.y = pt.dummy;

        SampleArray_g.data[binStartAddr_g[binIdx]+cap] = data;
        SampleArray_g.loc1[binStartAddr_g[binIdx]+cap] = loc1;
        SampleArray_g.loc2[binStartAddr_g[binIdx]+cap] = loc2;
        SampleArray_g.loc3[binStartAddr_g[binIdx]+cap] = loc3;
      } else {
        cap = atomicAdd(CPUbinSize, 1);
        CPUbin[cap]=idx;
      }
    } else {
      cap = atomicAdd(CPUbinSize, 1);
      CPUbin[cap]=idx;
    }
  }
#endif
}
// */

// 2D gridding on GPU
   __global__ void 
gridding_GPU_2D(samplePtArray sampleArray_g, unsigned int* start_g, 
             float *t_d, float l, float tau,
             float2* gridData_g, float* sampleDensity_g, float beta)
{
  float gridOS = gridOS_c;
  float _1_gridOS = 1.0f / gridOS;
  float kernelWidth = kernelWidth_c;
  float _1_kernelWidth = 1.0f / kernelWidth;
  float _4_kernelWidth2 = 4.0f * _1_kernelWidth * _1_kernelWidth;

  __shared__ float real_s[TILE];
  __shared__ float imag_s[TILE];
  __shared__ float shiftedkx_s[TILE];
  __shared__ float shiftedky_s[TILE];
  __shared__ float shiftedkz_s[TILE];
  __shared__ float sdc_s[TILE];
  __shared__ float t_s[TILE];


  int Nx = imageSize_c[0];
  int Ny = imageSize_c[1];

  const int flatIdx = threadIdx.y*blockDim.x+threadIdx.x;

  // figure out starting point of the tile
  //const int z0 = (4*blockDim.z)*(blockIdx.y/(gridSize_c[1]/blockDim.y));
  //const int y0 = blockDim.y*(blockIdx.y%(gridSize_c[1]/blockDim.y));
  const int y0 = blockIdx.y*blockDim.y;
  const int x0 = blockIdx.x*blockDim.x;

  const int X = x0+threadIdx.x;
  const int Y = y0+threadIdx.y;
  //const int Z = z0+threadIdx.z;
  //const int Z1 = Z+blockDim.z;
  //const int Z2 = Z1+blockDim.z;
  //const int Z3 = Z2+blockDim.z;

  float half_kernelWidth = kernelWidth * 0.5f;
  float half_length_grid = half_kernelWidth * (gridOS);
  const int xl = x0-ceil(half_length_grid);//+1);
  const int xL = (xl < 0) ? 0 : xl;
  const int xh = x0+blockDim.x+half_length_grid;//+1;
  const int xH = (xh >= gridSize_c[0]) ? gridSize_c[0]-1 : xh;

  const int yl = y0-ceil(half_length_grid);//+1);
  const int yL = (yl < 0) ? 0 : yl;
  const int yh = y0+blockDim.y+half_length_grid;//+1;
  const int yH = (yh >= gridSize_c[1]) ? gridSize_c[1]-1 : yh;

  //const int zl = z0-ceil(half_length_grid);//+1);
  //const int zL = (zl < 0) ? 0 : zl;
  //const int zh = z0+(4*blockDim.z)+half_length_grid;//+1;
  //const int zH = (zh >= gridSize_c[2]) ? gridSize_c[2]-1 : zh;

  const int idx = Y*gridSize_c[0] + X; 
  //const int idx = Z*Nxy_grid_c + Y*gridSize_c[0] + X;
  //const int idx1 = idx  + blockDim.z * Nxy_grid_c;
  //const int idx2 = idx1 + blockDim.z * Nxy_grid_c;
  //const int idx3 = idx2 + blockDim.z * Nxy_grid_c;

  float2 pt;
  pt.x = 0.0;
  pt.y = 0.0;
  float density = 0.0;

/*
  float2 pt1;
  pt1.x = 0.0;
  pt1.y = 0.0;
  float density1 = 0.0;  

  float2 pt2;
  pt2.x = 0.0;
  pt2.y = 0.0;
  float density2 = 0.0;

  float2 pt3;
  pt3.x = 0.0;
  pt3.y = 0.0;
  float density3 = 0.0;
*/

  //Jiading GAI
  float t0 = t_d[0];

  for (int y = yL; y <= yH; y++){
      const unsigned int* addr = start_g + y*gridSize_c[0];
      const unsigned int start = *(addr+xL);
      const unsigned int end = *(addr+xH+1);
      const unsigned int delta = end-start;
      ///*
      for (int x = 0; x < ((delta+TILE-1)>>LOG_TILE); x++){
        int fIdx = flatIdx+(x<<LOG_TILE);
        __syncthreads();
        if(fIdx < delta){
          //Jiading GAI: (start+fIdx) is k-space GlobalIndex.
          const float2 data = sampleArray_g.data[start+fIdx];
          const float2 loc1 = sampleArray_g.loc1[start+fIdx];
          const float2 loc2 = sampleArray_g.loc2[start+fIdx];
          const float2 loc3 = sampleArray_g.loc3[start+fIdx];
          
          real_s[flatIdx] = data.x;
          imag_s[flatIdx] = data.y;
          shiftedkx_s[flatIdx] = loc1.x;
          shiftedky_s[flatIdx] = loc1.y;
          shiftedkz_s[flatIdx] = loc2.x;
          sdc_s [flatIdx] = loc2.y;
          t_s[flatIdx] = loc3.x;
        }
        __syncthreads();

        const int jh = delta-(x<<LOG_TILE);
        const int jH = (jh > TILE) ? TILE : jh;
        for (int j=0; j< jH; j++){
          if((real_s[j] != 0.0 || imag_s[j] != 0.0) && sdc_s[j] != 0.0){
            const float real_l = real_s[j];
            const float imag_l = imag_s[j];
            const float atm_l = hanning_d(t_s[j], tau, l, t0);

            float distX, kbX, distY, kbY;

            distX = fabs(shiftedkx_s[j]-((float)X)) * (_1_gridOS);//
            float distX2 = distX * distX;
            if(distX<=(half_kernelWidth))
            {
               kbX = bessi0(beta*sqrt(1.0-_4_kernelWidth2*distX2)) * _1_kernelWidth;
               if (kbX!=kbX)//if kbX = NaN
                 kbX=0.0f;
            }
            else
               kbX = 0.0f;
               
           
            distY = fabs(shiftedky_s[j]-((float)Y)) * (_1_gridOS);//
            float distY2 = distY * distY;
            if(distY<=(half_kernelWidth))
            {
               kbY = bessi0(beta*sqrt(1.0-_4_kernelWidth2*distY2))*_1_kernelWidth;
               if (kbY!=kbY)//if kbY = NaN
                  kbY=0.0f;
            }
            else
               kbY = 0.0f;

 
            float w = kbX * kbY;
            pt.x += w*real_l*atm_l;
            pt.y += w*imag_l*atm_l;
//            density += w;

          }
        }
      }// */
  }
  
  if(X < gridSize_c[0] && Y < gridSize_c[1]) {
    gridData_g[idx] = pt;
//    sampleDensity_g[idx] = density;
  }
} //end of 2D gridding on GPU

// 3D gridding on GPU
   __global__ void 
gridding_GPU_3D(samplePtArray sampleArray_g, unsigned int* start_g, 
             float *t_d, float l, float tau,
             float2* gridData_g, float* sampleDensity_g, float beta)
{
  float gridOS = gridOS_c;
  float _1_gridOS = 1.0f / gridOS;
  float kernelWidth = kernelWidth_c;
  float _1_kernelWidth = 1.0f / kernelWidth;
  float _4_kernelWidth2 = 4.0f * _1_kernelWidth * _1_kernelWidth;

  __shared__ float real_s[TILE];
  __shared__ float imag_s[TILE];
  __shared__ float shiftedkx_s[TILE];
  __shared__ float shiftedky_s[TILE];
  __shared__ float shiftedkz_s[TILE];
  __shared__ float sdc_s[TILE];
  __shared__ float t_s[TILE];


  int Nx = imageSize_c[0];
  int Ny = imageSize_c[1];
  int Nz = imageSize_c[2];

  const int flatIdx = threadIdx.z*blockDim.y*blockDim.x+threadIdx.y*blockDim.x+threadIdx.x;

  // figure out starting point of the tile
  const int z0 = (4*blockDim.z)*(blockIdx.y/((gridSize_c[1]+blockDim.y-1)/blockDim.y));
  const int y0 = blockDim.y*(blockIdx.y%((gridSize_c[1]+blockDim.y-1)/blockDim.y));
  const int x0 = blockIdx.x*blockDim.x;
  //const int z0 = (4*blockDim.z)*(blockIdx.y/(gridSize_c[1]/blockDim.y));
  //const int y0 = blockDim.y*(blockIdx.y%(gridSize_c[1]/blockDim.y));
  //const int x0 = blockIdx.x*blockDim.x;

  const int X = x0+threadIdx.x;
  const int Y = y0+threadIdx.y;
  const int Z = z0+threadIdx.z;
  const int Z1 = Z+blockDim.z;
  const int Z2 = Z1+blockDim.z;
  const int Z3 = Z2+blockDim.z;

  float half_kernelWidth = kernelWidth * 0.5f;
  float half_length_grid = half_kernelWidth * (gridOS);
  const int xl = x0-ceil(half_length_grid);//+1);
  const int xL = (xl < 0) ? 0 : xl;
  const int xh = x0+blockDim.x+half_length_grid;//+1;
  const int xH = (xh >= gridSize_c[0]) ? gridSize_c[0]-1 : xh;

  const int yl = y0-ceil(half_length_grid);//+1);
  const int yL = (yl < 0) ? 0 : yl;
  const int yh = y0+blockDim.y+half_length_grid;//+1;
  const int yH = (yh >= gridSize_c[1]) ? gridSize_c[1]-1 : yh;

  const int zl = z0-ceil(half_length_grid);//+1);
  const int zL = (zl < 0) ? 0 : zl;
  const int zh = z0+(4*blockDim.z)+half_length_grid;//+1;
  const int zH = (zh >= gridSize_c[2]) ? gridSize_c[2]-1 : zh;

  const int idx = Z*Nxy_grid_c + Y*gridSize_c[0] + X;
  const int idx1 = idx  + blockDim.z * Nxy_grid_c;
  const int idx2 = idx1 + blockDim.z * Nxy_grid_c;
  const int idx3 = idx2 + blockDim.z * Nxy_grid_c;

  float2 pt;
  pt.x = 0.0;
  pt.y = 0.0;
  float density = 0.0;

  float2 pt1;
  pt1.x = 0.0;
  pt1.y = 0.0;
  float density1 = 0.0;  

  float2 pt2;
  pt2.x = 0.0;
  pt2.y = 0.0;
  float density2 = 0.0;

  float2 pt3;
  pt3.x = 0.0;
  pt3.y = 0.0;
  float density3 = 0.0;

  //Jiading GAI
  float t0 = t_d[0];


  for (int z = zL; z <= zH; z++){
    for (int y = yL; y <= yH; y++){
      const unsigned int* addr = start_g+z*Nxy_grid_c+ y*gridSize_c[0];
      const unsigned int start = *(addr+xL);
      const unsigned int end = *(addr+xH+1);
      const unsigned int delta = end-start;
      ///*
      for (int x = 0; x < ((delta+TILE-1)>>LOG_TILE); x++){
        int fIdx = flatIdx+(x<<LOG_TILE);
        __syncthreads();
        if(fIdx < delta){

          if(X < gridSize_c[0] && Y < gridSize_c[1] && Z < gridSize_c[2]) {
              //Jiading GAI: (start+fIdx) is k-space GlobalIndex.
              const float2 data = sampleArray_g.data[start+fIdx];
              const float2 loc1 = sampleArray_g.loc1[start+fIdx];
              const float2 loc2 = sampleArray_g.loc2[start+fIdx];
              const float2 loc3 = sampleArray_g.loc3[start+fIdx];

              real_s[flatIdx] = data.x;
              imag_s[flatIdx] = data.y;
              shiftedkx_s[flatIdx] = loc1.x;
              shiftedky_s[flatIdx] = loc1.y;
              shiftedkz_s[flatIdx] = loc2.x;
              sdc_s [flatIdx] = loc2.y;
              t_s[flatIdx] = loc3.x;
          }
        }
        __syncthreads();

        const int jh = delta-(x<<LOG_TILE);
        const int jH = (jh > TILE) ? TILE : jh;
        for (int j=0; j< jH; j++){
          if((real_s[j] != 0.0 || imag_s[j] != 0.0) && sdc_s[j] != 0.0){
            const float real_l = real_s[j];
            const float imag_l = imag_s[j];
            const float atm_l = hanning_d(t_s[j], tau, l, t0);

            float distX, kbX, distY, kbY, distZ, kbZ;

            distX = fabs(shiftedkx_s[j]-((float)X)) * (_1_gridOS);//
            float distX2 = distX * distX;
            if(distX<=(half_kernelWidth))
            {
               kbX = bessi0(beta*sqrt(1.0-_4_kernelWidth2*distX2)) * _1_kernelWidth;
               if (kbX!=kbX)//if kbX = NaN
                 kbX=0.0f;
            }
            else
               kbX = 0.0f;
               
           
            distY = fabs(shiftedky_s[j]-((float)Y)) * (_1_gridOS);//
            float distY2 = distY * distY;
            if(distY<=(half_kernelWidth))
            {
               kbY = bessi0(beta*sqrt(1.0-_4_kernelWidth2*distY2)) * _1_kernelWidth;
               if (kbY!=kbY)//if kbY = NaN
                  kbY=0.0f;
            }
            else
               kbY = 0.0f;


            // Update output
            distZ = fabs(shiftedkz_s[j]-((float)Z)) * (_1_gridOS);//
            float distZ2 = distZ * distZ;
            if(distZ<=(half_kernelWidth))
            {
               kbZ = bessi0(beta*sqrt(1.0-_4_kernelWidth2*distZ2)) * _1_kernelWidth;
               if (kbZ!=kbZ)//if kbZ = NaN
                  kbZ=0.0f;
            }
            else
               kbZ = 0.0f;
 
            float w = kbX * kbY * kbZ;
            pt.x += w*real_l*atm_l;
            pt.y += w*imag_l*atm_l;
//            density += w;



            // Update output1
            distZ = fabs(shiftedkz_s[j]-((float)Z1)) * (_1_gridOS);//
            distZ2 = distZ * distZ;
            if(distZ<=(half_kernelWidth))
            {
               kbZ = bessi0(beta*sqrt(1.0-_4_kernelWidth2*distZ2)) * _1_kernelWidth;
               if (kbZ!=kbZ)//if kbZ = NaN
                   kbZ=0.0f;
            }
            else
               kbZ = 0.0f;

            w = kbX * kbY * kbZ;
            pt1.x += w*real_l*atm_l;
            pt1.y += w*imag_l*atm_l;
//            density1 += w;


            // Update output2
            distZ = fabs(shiftedkz_s[j]-((float)Z2)) * (_1_gridOS);//
            distZ2 = distZ * distZ;
            if(distZ<=(half_kernelWidth))
            {
               kbZ = bessi0(beta*sqrt(1.0-_4_kernelWidth2*distZ2)) * _1_kernelWidth;
               if (kbZ!=kbZ)//if kbZ = NaN
                   kbZ=0.0f;
            }
            else
               kbZ = 0.0f;

            w = kbX * kbY * kbZ;
            pt2.x += w*real_l*atm_l;
            pt2.y += w*imag_l*atm_l;
//            density2 += w;

            // Update output3
            distZ = fabs(shiftedkz_s[j]-((float)Z3)) * (_1_gridOS);//
            distZ2 = distZ * distZ;
            if(distZ<=(half_kernelWidth))
            {
               kbZ = bessi0(beta*sqrt(1.0-_4_kernelWidth2*distZ2)) * _1_kernelWidth;
               if (kbZ!=kbZ)//if kbZ = NaN
                   kbZ=0.0f;
            }
            else
               kbZ = 0.0f;

            w = kbX * kbY * kbZ;
            pt3.x += w*real_l*atm_l;
            pt3.y += w*imag_l*atm_l;
//            density3 += w;
          }
        }
      }// */
    }
  }

  if(X < gridSize_c[0] && Y < gridSize_c[1] && Z < gridSize_c[2]) {
      gridData_g[idx] = pt;
//      sampleDensity_g[idx] = density;

      if(Z1 < gridSize_c[2]) {
          gridData_g[idx1] = pt1;
//          sampleDensity_g[idx1] = density1;
      }

      if(Z2 < gridSize_c[2]) {
          gridData_g[idx2] = pt2;
//          sampleDensity_g[idx2] = density2;
      }

      if(Z3 < gridSize_c[2]) {
          gridData_g[idx3] = pt3;
//          sampleDensity_g[idx3] = density3;
      }
  }
}
