#include <stdio.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "WKFUtils.h"
#include "UDTypes.h"

#include "scanLargeArray.h"
#include "GPU_kernels.cu"
#include "CPU_kernels.h"
#include "gridding.h"

#include <string.h>

#define BLOCKSIZE 512

int compare (const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}

   void 
CUDA_interface(unsigned int n, parameters params, ReconstructionSample* sample, 
               float* LUT, int sizeLUT, 
               float *t, float *t_d, float l, float tau, float beta,
               cufftComplex* gridData, float* sampleDensity)
{
  /* Obtain device property b/c 1.0 hardware 
     does not support atomic operations - JGAI*/  
  int current_deviceID = -1;
  CUDA_SAFE_CALL(cudaGetDevice(&current_deviceID));
  cudaDeviceProp prop;
  CUDA_SAFE_CALL( cudaGetDeviceProperties(&prop,current_deviceID) );



  // The 9800 GTX doesn't support zero copy. The only compute 1.1 
  // devices that support zero copy are the MCP79 family of 
  // integrated GPUs (9300M,9400M,Ion). 
  //REF:http://forums.nvidia.com/index.php?showtopic=192361
  unsigned int no_zerocopy = 1;
  if(prop.deviceOverlap==1)
      no_zerocopy = 0;
  unsigned int no_atomic = 0;
  if(prop.major==1 && prop.minor==0)
     no_atomic = 1;//enable atomic if the hardware supports it.

  /* Initializing all variables */

  int Nx = params.imageSize[0];// image size
  int Ny = params.imageSize[1];
  int Nz = params.imageSize[2];

  int Nx_grid = params.gridSize[0];
  int Ny_grid = params.gridSize[1];
  int Nz_grid = params.gridSize[2];
  int Nxy_grid = Nx_grid * Ny_grid;
  int gridNumElems = Nx_grid * Ny_grid * Nz_grid;

  dim3 dims;
  if(Nz==1) // 2D
  {
     //size of a gridding block on the GPU
     dims.x = 8;
     dims.y = 8;//8x8x1
  }
  else
  {
     //size of a gridding block on the GPU
     dims.x = 8;
     dims.y = 4;
     dims.z = 2; // 8x4x2 
  }

  int npad = 0;
  if (n % 64 != 0){
    npad = 64 - (n%64);
  }

  /* Declarations of host data structures */
  cufftComplex* gridData_CPU = NULL;
  float* sampleDensity_CPU = NULL;
  int* indices_CPU = NULL;

  /* Declarations of device data structures */
  ReconstructionSample *Sample_d  = NULL;
  float* Sample_sd = NULL;
  float* Sample_sh = NULL;//used only when no_atomic=1
  float2* gridData_d = NULL;
  float* sampleDensity_d = NULL;
  samplePtArray sampleArray_d;
  samplePtArray sampleArray_h;//used only when no_atomic=1
  unsigned int* numPts_d = NULL;
  unsigned int* numPts = NULL;//used only when no_atomic=1
  unsigned int* start_d = NULL;
  unsigned int* start = NULL;
  unsigned int* CPUbinSize_d = NULL;
  unsigned int* CPUbin_d = NULL;
  int CPUbin_size = 0;
  int *CPUbin = NULL;
  

  /* Allocating device memory */
  wkf_timerhandle timer;
  timer = wkf_timer_create();
  wkf_timer_start(timer);

  CUDA_SAFE_CALL(cudaMalloc((void**)&Sample_sd, (n+npad)*sizeof(ReconstructionSample)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&start_d, (gridNumElems+1)*sizeof(unsigned int)));
  start = (unsigned int*) calloc((gridNumElems+1), sizeof(unsigned int));
  CUDA_SAFE_CALL(cudaMalloc((void**)&Sample_d, n*sizeof(ReconstructionSample)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&numPts_d, (gridNumElems+1)*sizeof(unsigned int)));  
  CUDA_SAFE_CALL(cudaMalloc((void**)&CPUbin_d, n*sizeof(unsigned int)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&CPUbinSize_d, sizeof(unsigned int)));
  if(no_zerocopy==0) {
      //Your GPU supports overlap of data transfer and kernel execution
      CUDA_SAFE_CALL(cudaMallocHost((void**)&CPUbin,n*sizeof(int)));
  }
  else {
      CUT_SAFE_MALLOC(CPUbin = (int*) calloc(n,sizeof(int)));
  }
  

  wkf_timer_stop(timer);
  #if DEBUG_MODE
  printf("  Time taken for malloc is %f\n", wkf_timer_time(timer));
  #endif
  wkf_timer_start(timer);

  /* Transfering data from Host to Device */
  CUDA_SAFE_CALL(cudaMemset(start_d, 0, (gridNumElems+1)*sizeof(unsigned int)));
  CUDA_SAFE_CALL(cudaMemset(numPts_d, 0, (gridNumElems+1)*sizeof(unsigned int)));
  CUDA_SAFE_CALL(cudaMemset(CPUbinSize_d, 0, sizeof(unsigned int)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(gridSize_c, params.gridSize, 3*sizeof(int), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(imageSize_c, params.imageSize, 3*sizeof(int), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(gridOS_c, &params.gridOS, sizeof(int), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(kernelWidth_c, &params.kernelWidth, sizeof(float), 0));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(Nxy_grid_c, &Nxy_grid, sizeof(int), 0));
  CUDA_SAFE_CALL(cudaMemcpy(Sample_d, sample, n*sizeof(ReconstructionSample), cudaMemcpyHostToDevice));

  sampleArray_d.data = (float2*)(Sample_sd);
  sampleArray_d.loc1 = (float2*)(Sample_sd+2*(n+npad));
  sampleArray_d.loc2 = (float2*)(Sample_sd+4*(n+npad));
  sampleArray_d.loc3 = (float2*)(Sample_sd+6*(n+npad));

     
  if(1==no_atomic)//JGAI - no_atomic: do everything on CPU side
  {
    CUT_SAFE_MALLOC(numPts = (unsigned int*) calloc((gridNumElems+1), sizeof(unsigned int)));
    CUT_SAFE_MALLOC(Sample_sh = (float *) malloc((n+npad)*sizeof(ReconstructionSample)));

    sampleArray_h.data = (float2*)(Sample_sh);
    sampleArray_h.loc1 = (float2*)(Sample_sh+2*(n+npad));
    sampleArray_h.loc2 = (float2*)(Sample_sh+4*(n+npad));
    sampleArray_h.loc3 = (float2*)(Sample_sh+6*(n+npad));
  }

  wkf_timer_stop(timer);
  #if DEBUG_MODE
  printf("  Time taken for initial allocation and data transfer is %f\n", wkf_timer_time(timer));
  #endif
  wkf_timer_start(timer);

  dim3 block1 (512);
  dim3 grid1 ((n+511)/512);

  
  if(0==no_atomic)//GPU has atomic operations
  {
     binning_kernel1<<<grid1, block1>>>(n, Sample_d, start_d, params.binsize, gridNumElems);
     CUT_CHECK_ERROR("Unable to launch binning kernel\n");
  }
  else//JGAI - no_atomic: do everything on CPU side
  {
     binning_kernel1_CPU(n, sample, start, params);
  }
 
  if (params.sync){
    CUDA_SAFE_CALL(cudaThreadSynchronize());
  }

  wkf_timer_stop(timer);
  #if DEBUG_MODE
  printf("  Time taken for binning1 is %f\n", wkf_timer_time(timer));
  #endif
  wkf_timer_start(timer);

  if(0==no_atomic)//GPU has atomic operations
  {
     scanLargeArray(gridNumElems+1, start_d);
     CUDA_SAFE_CALL(cudaMemcpy(start, start_d, (gridNumElems+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost));
  }
  else//JGAI - no_atomic: do everything on CPU side
  {
     scanLargeArray_CPU((gridNumElems+1),start);
  }

  if (params.sync){
    CUDA_SAFE_CALL(cudaThreadSynchronize());
  }

  wkf_timer_stop(timer);
  #if DEBUG_MODE
  printf("  Time taken for scanning is %f\n", wkf_timer_time(timer));
  #endif
  wkf_timer_start(timer);

  if(0==no_atomic)//Your GPU supports atomic operations.
  {
     binning_kernel2<<<grid1,block1>>>
                    (n, Sample_d, start_d, numPts_d, params.binsize, 
                    sampleArray_d, CPUbinSize_d, CPUbin_d);
     CUT_CHECK_ERROR("Unable to launch reorder kernel\n");
  }
  else//JGAI - no_atomic: do everything on CPU side
  {
     binning_kernel2_CPU(n, sample, start, numPts, params, 
                         sampleArray_h, CPUbin_size, CPUbin);
  }


  if (params.sync){
    CUDA_SAFE_CALL(cudaThreadSynchronize());
  }

  wkf_timer_stop(timer);
  #if DEBUG_MODE
  printf("  Time taken for binning2 is %f\n", wkf_timer_time(timer));
  #endif
  wkf_timer_start(timer);

  CUDA_SAFE_CALL(cudaFree(Sample_d));
  CUDA_SAFE_CALL(cudaFree(numPts_d));

  if(0==no_atomic)//GPU has atomic operations
  {
    CUDA_SAFE_CALL(cudaMemcpy(&CPUbin_size, CPUbinSize_d, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(CPUbin, CPUbin_d, CPUbin_size*sizeof(unsigned int), cudaMemcpyDeviceToHost));
  }
  else//JGAI - no_atomic: do everything on CPU side
  {
    CUDA_SAFE_CALL( cudaMemcpy(Sample_sd, Sample_sh, 
                    (n+npad)*sizeof(ReconstructionSample), 
                    cudaMemcpyHostToDevice) );


    CUDA_SAFE_CALL( cudaMemcpy(start_d, start, (gridNumElems+1)*sizeof(unsigned int), 
                    cudaMemcpyHostToDevice) );
  }

  free(start);
  CUDA_SAFE_CALL(cudaFree(CPUbin_d));
  CUDA_SAFE_CALL(cudaFree(CPUbinSize_d));

  wkf_timer_stop(timer);
  #if DEBUG_MODE
  printf("  Time taken for copying CPU load back is %f\n", wkf_timer_time(timer));
  #endif
  wkf_timer_start(timer);

  CUDA_SAFE_CALL(cudaMalloc((void**)&gridData_d, gridNumElems*sizeof(float2)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&sampleDensity_d, gridNumElems*sizeof(float)));

  CUDA_SAFE_CALL(cudaMemset(gridData_d, 0, gridNumElems*sizeof(float2)));
  CUDA_SAFE_CALL(cudaMemset(sampleDensity_d, 0, gridNumElems*sizeof(float)));

  if(Nz==1)
  {
    dim3 block2 (dims.x,dims.y);
    dim3 grid2 ((Nx_grid+dims.x-1)/dims.x, (Ny_grid+dims.y-1)/dims.y);

    gridding_GPU_2D<<<grid2, block2>>>(sampleArray_d, start_d, t_d, l, tau, gridData_d, sampleDensity_d, beta);
    CUT_CHECK_ERROR("Unable to launch gridding kernel\n");
  }
  else
  {
    dim3 block2 (dims.x,dims.y,dims.z);
    //dim3 grid2 (Nx_grid/dims.x, (Ny_grid*Nz_grid)/(4*dims.y*dims.z));
    int grid_dim_y = (Ny_grid+dims.y-1)/dims.y;
    int grid_dim_z = (Nz_grid+4*dims.z-1)/(4*dims.z);
    dim3 grid2 ((Nx_grid+dims.x-1)/dims.x, grid_dim_y*grid_dim_z);

    
    gridding_GPU_3D<<<grid2, block2>>>(sampleArray_d, start_d, t_d, l, tau, gridData_d, sampleDensity_d, beta);
    CUT_CHECK_ERROR("Unable to launch gridding kernel\n");
  }


  if (params.sync){
    CUDA_SAFE_CALL(cudaThreadSynchronize());
  }

  wkf_timer_stop(timer);
  #if DEBUG_MODE
  printf("  Time taken for GPU computing is %f\n", wkf_timer_time(timer));
  #endif
  wkf_timer_start(timer); 

  qsort(CPUbin, CPUbin_size, sizeof(int), compare);
  int num = 0;
  if(Nz==1)
  {
    num = gridding_CPU_2D(n, params, sample, CPUbin, CPUbin_size, LUT, sizeLUT, 
                    t, l, tau, &gridData_CPU, &sampleDensity_CPU, &indices_CPU);
  }
  else
  {
    num = gridding_CPU_3D(n, params, sample, CPUbin, CPUbin_size, LUT, sizeLUT, 
                     t, l, tau, &gridData_CPU, &sampleDensity_CPU, &indices_CPU);
  }

  if (no_zerocopy==0) {
      CUDA_SAFE_CALL(cudaFreeHost(CPUbin));
  }
  else {
      free(CPUbin);
  }

  wkf_timer_stop(timer);
  #if DEBUG_MODE
  printf("  Time taken for CPU computing is %f\n", wkf_timer_time(timer));
  #endif
  wkf_timer_start(timer);

  CUDA_SAFE_CALL(cudaMemcpy(sampleDensity, sampleDensity_d, gridNumElems*sizeof(float),cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(gridData, gridData_d, gridNumElems*sizeof(float2),cudaMemcpyDeviceToHost));

  wkf_timer_stop(timer);
  #if DEBUG_MODE
  printf("  Time taken for transfering data back is %f\n", wkf_timer_time(timer));
  #endif
  wkf_timer_start(timer);

  for (int i=0; i< num; i++){
    gridData[indices_CPU[i]].x += gridData_CPU[i].x;
    gridData[indices_CPU[i]].y += gridData_CPU[i].y;
    sampleDensity[indices_CPU[i]] += sampleDensity_CPU[i];
  }

 ///*
 // re-arrange dimensions and output
 // Nady uses: x->y->z
 // IMPATIENT uses: z->x->y
 // So we need to convert from (x->y->z)-order to (z->x->y)-order
 // Note that: the following re-arranging code works for both 2D and 3D cases.
 cufftComplex *gridData_reorder;
 CUT_SAFE_MALLOC(gridData_reorder = (cufftComplex*) calloc(gridNumElems, sizeof(cufftComplex)));

 if(Nz==1)
 {
    for(int x=0;x<params.gridSize[0];x++)
    for(int y=0;y<params.gridSize[1];y++)
    {
       int lindex_nady = x + y*params.gridSize[0];
       int lindex_impatient = y + x*params.gridSize[1];
   
       gridData_reorder[lindex_impatient] = gridData[lindex_nady];
    }
 }
 else
 {
    for(int x=0;x<params.gridSize[0];x++)
    for(int y=0;y<params.gridSize[1];y++)
    for(int z=0;z<params.gridSize[2];z++)
    {
       int lindex_nady = x + y*params.gridSize[0] + z*params.gridSize[0]*params.gridSize[1];
       int lindex_impatient = z + x*params.gridSize[2] + y*params.gridSize[0]*params.gridSize[2];
   
       gridData_reorder[lindex_impatient] = gridData[lindex_nady];
    }
 }

 memcpy((void*)gridData,(void*)gridData_reorder,gridNumElems*sizeof(cufftComplex));
 free(gridData_reorder);
 // */

 if (gridData_CPU != NULL){
    free(gridData_CPU);
 }

 if (indices_CPU != NULL){
    free(indices_CPU);
 }

 if (sampleDensity_CPU != NULL){
    free(sampleDensity_CPU);
 }

 if (Sample_sh != NULL){
    free(Sample_sh);
 }

 if (numPts != NULL) {
    free(numPts);
 }

  wkf_timer_stop(timer);
  #if DEBUG_MODE
  printf("  Time taken for Merging is %f\n", wkf_timer_time(timer));
  #endif
  wkf_timer_start(timer);

  CUDA_SAFE_CALL(cudaFree(start_d));
  CUDA_SAFE_CALL(cudaFree(gridData_d));
  CUDA_SAFE_CALL(cudaFree(sampleDensity_d));
  CUDA_SAFE_CALL(cudaFree(Sample_sd));

  wkf_timer_stop(timer);
  #if DEBUG_MODE
  printf("  Time taken for free is %f\n", wkf_timer_time(timer));
  #endif
  wkf_timer_destroy(timer);

  return;
}
