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

    File Name   [computeQ.cmem.cu]

    Synopsis    [ CUDA code for creating the Q data structure for fast 
                  convolution-based Hessian multiplication for arbitrary 
                  k-space trajectories.
                ]

    Description [Inputs:
                    kx - VECTOR of kx values, same length as ky and kz
                    ky - VECTOR of ky values, same length as kx and kz
                    kz - VECTOR of kz values, same length as kx and ky
                    x  - VECTOR of x values, same length as y and z
                    y  - VECTOR of y values, same length as x and z
                    z  - VECTOR of z values, same length as x and y
                   phi - VECTOR of the Fourier transform of the spatial basis
                         function, evaluated at [kx, ky, kz].  Same length as 
                         kx, ky, and kz.
                 ]

    Revision    [1.0; Initial build; Sam S. Stone, ECE UIUC]
    Revision    [2.0; Code extension; Jiading Gai, Beckman Institute UIUC and 
                Xiao-Long Wu, ECE UIUC]
    Date        [03/23/2011]

*****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <stdio.h>
#include <math.h>
#include <cutil.h>
#include <assert.h>
#include <string.h>

// XCPPLIB libraries
#include <xcpplib_global.h>
#include <xcpplib_process.h>

// Project header files
#include <tools.h>
#include <structures.h>
#include <utils.h>
#include <gridding.h>
#include <utils.h>
#include <CUDA_interface.h>
#include <UDTypes.h>
#include <WKFUtils.h>
#include <CPU_kernels.h>
#include <fftshift.cuh>
#include <cufft.h>
#include <gridding_utils.cuh>

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

#ifdef DEBUG
    #warning "Debug mode is enabled."
    #define ERROR_CHECK  true
#else
    #define ERROR_CHECK  false
#endif

#define KERNEL_PHI_MAG_THREADS_PER_BLOCK 256
#define KERNEL_MARSHALL_SCALE_TPB 256
#define KERNEL_Q_THREADS_PER_BLOCK 128
#define KERNEL_Q_K_ELEMS_PER_GRID 512
#define KERNEL_Q_X_ELEMS_PER_THREAD 1


struct kValues_Q {
    float Kx;
    float Ky;
    float Kz;
    float PhiMag;
};

// Cannot use 4096 elems per array per grid - execution fails.
__constant__ static __device__ kValues_Q c[KERNEL_Q_K_ELEMS_PER_GRID];


// t0 is there b/c t.dat does not start with 0.0f.
  __host__ __device__ static
float hanning_d(float tm, float tau, float l, float t0)
{
    float taul = tau * l;
    float result;
    if ( abs(tm - taul - t0) < tau ) {
        result = 0.5f + 0.5f * cosf(PI * (tm - taul - t0) / tau);
    } else {
        result = 0.0f;
    }
    //FIXME:
    //result = 1.0f;
    return result;
}


    __global__ void
ComputePhiMag_GPU(
    float *phiR, float *phiI, float *phiMag, int numK) 
{
    int indexK = blockIdx.y * (gridDim.x * KERNEL_PHI_MAG_THREADS_PER_BLOCK) 
               + blockIdx.x * KERNEL_PHI_MAG_THREADS_PER_BLOCK + threadIdx.x;
    if (indexK < numK) {
        float real = phiR[indexK];
        float imag = phiI[indexK];
        phiMag[indexK] = real * real + imag * imag;
    }
}

    __global__ void
MarshallScaleGPU( 
    int numK, float *kx, float *ky, 
    float *kz, float *phiMag, kValues_Q *kVals ) 
{
    int indexK = blockIdx.y * (gridDim.x * KERNEL_MARSHALL_SCALE_TPB) 
               + blockIdx.x * KERNEL_MARSHALL_SCALE_TPB + threadIdx.x;
    if (indexK < numK) {
        kVals[indexK].Kx = PIx2 * kx[indexK];
        kVals[indexK].Ky = PIx2 * ky[indexK];
        kVals[indexK].Kz = PIx2 * kz[indexK];
        kVals[indexK].PhiMag = phiMag[indexK];
    }
}

    __global__ void
ComputeQ_GPU(int numK, int numX, int kGlobalIndex,
             float *x, float *y, float *z,
             float *t, float l, float tau,
             float *Qr, float *Qi) 
{
    float sX;
    float sY;
    float sZ;
    float sQr;
    float sQi;
    float t0 = t[0];

    // Determine the element of the X arrays computed by this thread
    int xIndex = blockIdx.y * (gridDim.x * KERNEL_Q_THREADS_PER_BLOCK) 
               + blockIdx.x * KERNEL_Q_THREADS_PER_BLOCK + threadIdx.x;
    if (xIndex < numX) {
        // Read block's X values from global mem to shared mem
        sX = x[xIndex];
        sY = y[xIndex];
        sZ = z[xIndex];
        sQr = Qr[xIndex];
        sQi = Qi[xIndex];

        // Loop over all elements of K in constant mem to compute a partial value for
        // X.
        for ( int kIndex = 0; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && 
              (kGlobalIndex < numK); kIndex++, kGlobalIndex++) 
        {
            float expArg = c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ;
            float atm = hanning_d(t[kGlobalIndex], tau, l, t0);
            sQr += atm * c[kIndex].PhiMag *cos(expArg);
            sQi += atm * c[kIndex].PhiMag *sin(expArg);
        }

        Qr[xIndex] = sQr;
        Qi[xIndex] = sQi;
    }

}

    static void 
inputData_Q(
    const char *input_folder_path,
    float&version, int&numK, int&numK_per_coil,
    int&ncoils, int&nslices, int&numX, int&numX_per_coil,
    float *&kx, float *&ky, float *&kz,
    float *&x, float *&y, float *&z,
    float *&t,
    float *&phiR, float *&phiI)
{
  	//Step 0. Test data format version (0.2 or 1.0 higher)
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

    //Step 1. Read input data files
    string kz_fn = input_folder_path;
    kz_fn = kz_fn + "/kz.dat";
    string ky_fn = input_folder_path;
    ky_fn = ky_fn + "/ky.dat";
    string kx_fn = input_folder_path;
    kx_fn = kx_fn + "/kx.dat";
    string t_fn = input_folder_path;
    t_fn = t_fn + "/t.dat";
    string iz_fn = input_folder_path;
    iz_fn = iz_fn + "/izQ.dat";
    string iy_fn = input_folder_path;
    iy_fn = iy_fn + "/iyQ.dat";
    string ix_fn = input_folder_path;
    ix_fn = ix_fn + "/ixQ.dat";
    string phiR_fn = input_folder_path;
    phiR_fn = phiR_fn + "/phiR.dat";
    string phiI_fn = input_folder_path;
    phiI_fn = phiI_fn + "/phiI.dat";

    if(0.2f==the_version) {
      kz = readDataFile_JGAI(kz_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      ky = readDataFile_JGAI(ky_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      kx = readDataFile_JGAI(kx_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      t = readDataFile_JGAI(t_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      z = readDataFile_JGAI(iz_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);
      y = readDataFile_JGAI(iy_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);
      x = readDataFile_JGAI(ix_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);
      phiR = readDataFile_JGAI(phiR_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      phiI = readDataFile_JGAI(phiI_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
    }
    else {
      kz = readDataFile_JGAI_10(kz_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      ky = readDataFile_JGAI_10(ky_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      kx = readDataFile_JGAI_10(kx_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      t = readDataFile_JGAI_10(t_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      z = readDataFile_JGAI_10(iz_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);
      y = readDataFile_JGAI_10(iy_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);
      x = readDataFile_JGAI_10(ix_fn.c_str(), version, numX_per_coil, ncoils, nslices, numX);
      phiR = readDataFile_JGAI_10(phiR_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      phiI = readDataFile_JGAI_10(phiI_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
    }
}

    static void 
outputData(
    const char *fName, float **Qr_gpu, float **Qi_gpu, int numX, float L) 
{
    FILE *fid = fopen(fName, "w");
    for (int l = 0; l <= ((int) L); l++) {

        if(numX!=fwrite(Qr_gpu[l], sizeof(float), numX, fid))
        {
          printf("Error: fwrite error at line %d, file %s.\n",__LINE__,__FILE__);
          exit(1);
        }

        if(numX!=fwrite(Qi_gpu[l], sizeof(float), numX, fid))
        {
          printf("Error: fwrite error at line %d, file %s.\n",__LINE__,__FILE__);
          exit(1);
        }
    }
    fclose(fid);
}


    static void 
createDataStructs(
    int numK_per_coil, int numX_per_coil, float *&phiMag)
{
    phiMag = (float * ) calloc(numK_per_coil, sizeof(float));
}

    void 
computePhiMag_GPU( 
    int numK, float *phiR_d, float *phiI_d, 
    float *phiMag_d, float *&phiMag_gpu) 
{
    unsigned int timerComputePhiMag = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timerComputePhiMag));
    CUT_SAFE_CALL(cutStartTimer(timerComputePhiMag));

    int phiMagxBlocks = numK / KERNEL_PHI_MAG_THREADS_PER_BLOCK;
    int phiMagyBlocks = 1;
    if (numK % KERNEL_PHI_MAG_THREADS_PER_BLOCK)
        phiMagxBlocks++;

    while (phiMagxBlocks > 32768) {
        phiMagyBlocks *= 2;
        if (phiMagxBlocks % 2) {
            phiMagxBlocks /= 2;
            phiMagxBlocks++;
        } else {
            phiMagxBlocks /= 2;
        }
    }
    
    dim3 DimPhiMagBlock(KERNEL_PHI_MAG_THREADS_PER_BLOCK, 1);
    dim3 DimPhiMagGrid(phiMagxBlocks, phiMagyBlocks);
    #ifdef DEBUG
    printf("Launch PhiMag Kernel on GPU: Blocks (%d, %d), Threads Per \
            Block %d\n", phiMagxBlocks, phiMagyBlocks, KERNEL_PHI_MAG_THREADS_PER_BLOCK);
    #endif
    ComputePhiMag_GPU <<< DimPhiMagGrid, DimPhiMagBlock >>> 
                         (phiR_d, phiI_d, phiMag_d, numK);

    mriCopyDeviceToHost(phiMag_gpu, phiMag_d, numK);

    #ifdef DEBUG
    CUT_CHECK_ERROR("ComputePhiMagGPU failed!\n");
    CUT_SAFE_CALL(cutStopTimer(timerComputePhiMag));
    printf( "Time to compute PhiMag on GPU: %f (s)\n", 
            cutGetTimerValue(timerComputePhiMag) / 1000.0 );
    CUT_SAFE_CALL(cutDeleteTimer(timerComputePhiMag));
    #endif
}

    void 
MarshallScale_GPU( 
    int numK, float *kx_d, float *ky_d, float *kz_d, 
    float *phiMag_d, kValues_Q *&kVals) 
{
    unsigned int timerMarshallScale = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timerMarshallScale));
    CUT_SAFE_CALL(cutStartTimer(timerMarshallScale));

    kValues_Q *kVals_d = mriNewGpu<kValues_Q>(numK);

    int blocks_x = numK / KERNEL_MARSHALL_SCALE_TPB;
    int blocks_y = 1;
    if (numK % KERNEL_MARSHALL_SCALE_TPB)
        blocks_x++;

    while (blocks_x > 32768) {
        blocks_y *= 2;
        if (blocks_x % 2) {
            blocks_x /= 2;
            blocks_x++;
        } else {
            blocks_x /= 2;
        }
    }

    dim3 dimBlock(KERNEL_MARSHALL_SCALE_TPB, 1);
    dim3 dimGrid(blocks_x, blocks_y);
    #ifdef DEBUG
    printf("Launch MarshallScale Kernel on GPU: Blocks (%d, %d), Threads \
            Per Block %d\n", blocks_x, blocks_y, KERNEL_MARSHALL_SCALE_TPB);
    #endif
    MarshallScaleGPU <<< dimGrid, dimBlock >>> 
             (numK, kx_d, ky_d, kz_d, phiMag_d, kVals_d);

    mriCopyDeviceToHost<kValues_Q>(kVals, kVals_d, numK);
    mriDeleteGpu<kValues_Q>(kVals_d);

    #ifdef DEBUG
    CUT_CHECK_ERROR("MarshallScale failed!\n");
    CUT_SAFE_CALL(cutStopTimer(timerMarshallScale));
    printf("Time to marshall and scale data on GPU: %f (s)\n", 
           cutGetTimerValue(timerMarshallScale) / 1000.0);
    CUT_SAFE_CALL(cutDeleteTimer(timerMarshallScale));
    #endif
}

    void 
computeQ_GPU_Grid(
    int numK_per_coil, int numX_per_coil,
    float *kx, float *ky, float *kz,
    //Actual MR image size, Q matrix is twice as large:
    int Nx, int Ny, int Nz,
    float *t, float *t_d, float l, float tau,
    float gridOS,
    float *Qr_d, float *Qi_d,
    float *&Qr_gpu, float *&Qi_gpu) 
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


    int Nx_Q = (int)(2.0f*Nx);//(Nx,Ny,Nz) is MR image size.
    int Ny_Q = (int)(2.0f*Ny);//(Nx_Q,Ny_Q,Nz_Q) is Q matrix size.
    int Nz_Q = (1==Nz)?Nz:((int)(2.0f*Nz));


///*
    // grid_size in xy-axis has to be divisible-by-two:
    //       (required by the cropImageRegion)
    // grid_size in z-axis has to be devisible-by-four:
    //       (required by the function gridding_GPU_3D(.))
    if(1==Nz_Q) {        
        //round grid size (xy-axis) to the next divisible-by-two.
        gridOS = 2.0f * ceil((gridOS * (float)Nx_Q) / 2.0f) / (float) Nx_Q;
    }
    else {
        //round grid size (z-axis) to the next divisible-by-four.
        gridOS = 4.0f * ceil((gridOS * (float)Nz_Q) / 4.0f) / (float) Nz_Q;
    }
// */

    float kernelWidth = 4.0f;
    #if 0
    float beta = 18.5547;
    #else
    /*
     *  Based on Eqn. (5) of Beatty's gridding paper:
     *  "Rapid Gridding Reconstruction With a Minimal Oversampling Ratio"
     *
     *  Note that Beatty use their kernel width to be equal twice the window
     *  width parameter used by Jackson et al. 
     */ 
     float beta = PI * sqrt( (gridOS - 0.5f) * (gridOS - 0.5f) * 
                             (kernelWidth * kernelWidth*4.0f) / 
                             (gridOS * gridOS) - 0.8f 
                           );
     //printf("beta=%f\n",beta);
     //exit(1);
     #endif

    parameters params;
    params.sync=0;
    params.binsize=128;

    params.useLUT = 0;
    params.kernelWidth = kernelWidth;
    params.gridOS = gridOS;
    params.imageSize[0] = Nx_Q;//params.imageSize is the Q matrix dimensions
    params.imageSize[1] = Ny_Q;//params.gridSize is gridOS time larger than the Q matrix dimensions.
    params.imageSize[2] = Nz_Q;
    params.gridSize[0]  = (ceil)(gridOS*(float)Nx_Q);
    params.gridSize[1]  = (ceil)(gridOS*(float)Ny_Q);
    if(params.gridSize[0]%2)
        params.gridSize[0] += 1;
    if(params.gridSize[1]%2)
        params.gridSize[1] += 1;
    params.gridSize[2]  = (Nz_Q==1)?Nz_Q:((ceil)(gridOS*(float)Nz_Q));// 2D or 3D
    params.numSamples = numK_per_coil;

    ReconstructionSample* samples; //Input Data    
    float* LUT; //use look-up table for faster execution on CPU (intermediate data)
    unsigned int sizeLUT; //set in the function calculateLUT (intermediate data)

    cufftComplex* gridData; //Output Data
    float* sampleDensity; //Output Data
   
    if (no_zerocopy==0) {
        CUDA_SAFE_CALL(
            cudaMallocHost((void**)&samples, params.numSamples*sizeof(ReconstructionSample))
        );
    }
    else {
        samples = (ReconstructionSample*) malloc(params.numSamples*sizeof(ReconstructionSample));
    }
    if (samples == NULL){
      printf("ERROR: Unable to allocate memory for input data\n");
      exit(1);
    }

    unsigned int n =  params.numSamples;
    // 
    for(int i=0; i<params.numSamples; i++){
      if( abs(kx[i])>(Nx/2.0f) ||
          abs(ky[i])>(Ny/2.0f) ||
          abs(kz[i])>(Nz/2.0f) 
        ) {
        printf("\nError:k-space trajectory out of range [-N/2,N/2]:\n      gridding requires that k-space should be contained within the winodw -N/2 to N/2.\n");
        exit(1);
     }
      else {

         samples[i].kX = 2.0f * kx[i];
         samples[i].kY = 2.0f * ky[i];
         samples[i].kZ = 2.0f * kz[i];
 
         samples[i].real = 1.0f;// In the context of gridding Q matrix, k-space 
         samples[i].imag = 0.0f;// data is not needed, only k-trajectory
 
         samples[i].sdc = 1.0f;
         samples[i].t = t[i];
      }
      
    }

    if (params.useLUT){
      //Generating Look-Up Table
      calculateLUT(beta, params.kernelWidth, LUT, sizeLUT);
    }

    int gridNumElems  = params.gridSize[0] * 
                        params.gridSize[1] * 
                        params.gridSize[2] ;

    // Actual Q matrix dimensions
    int imageNumElems = params.imageSize[0] * 
                        params.imageSize[1] * 
                        params.imageSize[2] ;

  
    if (no_zerocopy==0) {
        CUDA_SAFE_CALL(cudaMallocHost((void**)&gridData, gridNumElems*sizeof(cufftComplex)));
        CUDA_SAFE_CALL(cudaMallocHost((void**)&sampleDensity, gridNumElems*sizeof(float)));
    }
    else {
        gridData = (cufftComplex*) malloc(gridNumElems*sizeof(cufftComplex));
        sampleDensity = (float*) malloc(gridNumElems*sizeof(float));
    }
    if (sampleDensity == NULL || gridData == NULL){
       printf("ERROR: Unable to allocate memory for output data\n");
       exit(1);
    }
      // Have to set 'gridData' and 'sampleDensity' to zero.
      // Because they will be involved in accumulative operations
      // inside gridding functions.
    for(int i=0;i<gridNumElems;i++)
    {
      gridData[i].x = 0.0f;
      gridData[i].y = 0.0f;
      
      sampleDensity[i] = 0.0f;
    }

  
    // Gridding Running - CUDA version
    wkf_timerhandle timer = wkf_timer_create();
    wkf_timer_start(timer);

    //Interface function to GPU implementation of gridding
    #if 1// Gridding with GPU (synergy with CPU)
    CUDA_interface(n, params, samples, LUT, sizeLUT, t, t_d,
                   l, tau, beta, gridData, sampleDensity);

    #else // Gridding with CPU - gold
      if(Nz==1)
      {
         gridding_Gold_2D(n, params, samples, LUT, sizeLUT, t, l, tau, 
                          gridData, sampleDensity);
      }
      else
      {
         gridding_Gold_3D(n, params, samples, LUT, sizeLUT, t, l, tau, 
                          gridData, sampleDensity);
      }
    #endif


    // Copy "gridData" from CPU to GPU
    cufftComplex *gridData_d;
    CUDA_SAFE_CALL(cudaMalloc((void**)&gridData_d,gridNumElems*sizeof(cufftComplex)));
    CUDA_SAFE_CALL(cudaMemcpy(gridData_d, gridData, gridNumElems*sizeof(cufftComplex), cudaMemcpyHostToDevice));

    // ifftshift(gridData):
    if(Nz==1)
    {
      cuda_fft2shift_grid(gridData_d,gridData_d,params.gridSize[0],
                          params.gridSize[1], 1);
    }
    else
    {
      cuda_fft3shift_grid(gridData_d,gridData_d,params.gridSize[0],
                          params.gridSize[1],params.gridSize[2],1);
    }
    // ifftn(gridData):
    cufftHandle plan;
    if(Nz==1)
    {
       CUFFT_SAFE_CALL( cufftPlan2d(&plan, params.gridSize[0], 
                                    params.gridSize[1], CUFFT_C2C) );
    }
    else
    {
       CUFFT_SAFE_CALL( cufftPlan3d(&plan, params.gridSize[0], params.gridSize[1],
                                    params.gridSize[2], CUFFT_C2C) );
    }
           /* Inverse transform 'gridData_d' in place. */
    CUFFT_SAFE_CALL( cufftExecC2C(plan, gridData_d, gridData_d, CUFFT_INVERSE) );
    CUFFT_SAFE_CALL(cufftDestroy(plan));


    // fftshift(gridData):
    if(Nz==1)
    {
       cuda_fft2shift_grid(gridData_d,gridData_d, params.gridSize[0], 
                           params.gridSize[1],0);
    }
    else
    {
       cuda_fft3shift_grid(gridData_d,gridData_d, params.gridSize[0], 
                           params.gridSize[1],params.gridSize[2],0);
    }

  //Jiading GAI - DEBUG
#if 0
CUDA_SAFE_CALL(cudaMemcpy(gridData, gridData_d, gridNumElems*sizeof(cufftComplex), cudaMemcpyDeviceToHost));
float *output_r = (float*) calloc (gridNumElems, sizeof(float));
float *output_i = (float*) calloc (gridNumElems, sizeof(float));

int lindex = 0;
for(int x=0;x<params.gridSize[0];x++)
for(int y=0;y<params.gridSize[1];y++)
for(int z=0;z<params.gridSize[2];z++)
{
   output_r[lindex] = gridData[lindex].x;
   output_i[lindex] = gridData[lindex].y;
   lindex++;
}

outputData("/home/UIUC/jgai/Desktop/FhD_nady.file",&output_r,&output_i,gridNumElems,0);
//outputData("./FhD.file",output_r,output_i,gridNumElems);
free(output_r);
free(output_i);
exit(1);
#endif 


    // crop the center region of the "image".
    cufftComplex *gridData_crop_d;
    CUDA_SAFE_CALL(cudaMalloc((void**)&gridData_crop_d,imageNumElems*sizeof(cufftComplex)));
    if(Nz==1)
    {
           crop_center_region2d(gridData_crop_d, gridData_d, params.imageSize[0], 
                                params.imageSize[1], params.gridSize[0], params.gridSize[1]);
    }
    else
    {
       crop_center_region3d(gridData_crop_d, gridData_d, 
                            params.imageSize[0], params.imageSize[1], params.imageSize[2],
                            params.gridSize[0],params.gridSize[1],params.gridSize[2]);
    }

    // deapodization
    //FIXME:gridSize or imageSize, that's a question!?
    if(Nz==1)
    {
        deapodization2d(gridData_crop_d, gridData_crop_d, params.imageSize[0], 
                        params.imageSize[1], kernelWidth, beta, params.gridOS);

    }
    else
    {
        deapodization3d(gridData_crop_d, gridData_crop_d, params.imageSize[0], 
                        params.imageSize[1], params.imageSize[2], kernelWidth, beta, params.gridOS);
    }


//Jiading GAI - DEBUG
#if 0 
deinterleave_data3d(gridData_crop_d, Qr_d, Qi_d, params.imageSize[0], params.imageSize[1], params.imageSize[2]);
   

float *output_r = (float*) calloc (imageNumElems, sizeof(float));
float *output_i = (float*) calloc (imageNumElems, sizeof(float));

CUDA_SAFE_CALL(cudaMemcpy(output_r, Qr_d, imageNumElems*sizeof(float), cudaMemcpyDeviceToHost));
CUDA_SAFE_CALL(cudaMemcpy(output_i, Qi_d, imageNumElems*sizeof(float), cudaMemcpyDeviceToHost));

outputData("/home/UIUC/jgai/Desktop/Q_2D.file",&output_r,&output_i,imageNumElems,0);
free(output_r);
free(output_i);
exit(1);
#endif


    // Copy results from gridData_crop_d to outR_d and outI_d
    // gridData_crop_d is cufftComplex, interleaving
    // De-interleaving the data from cufftComplex to outR_d-and-outI_d
    if(Nz==1)
    {
       deinterleave_data2d(gridData_crop_d, Qr_d, Qi_d, 
                           params.imageSize[0], params.imageSize[1]);
    }
    else
    {
       deinterleave_data3d(gridData_crop_d, Qr_d, Qi_d, params.imageSize[0], 
                           params.imageSize[1], params.imageSize[2]);
    }


    mriCopyDeviceToHost(Qr_gpu, Qr_d, numX_per_coil);
    mriCopyDeviceToHost(Qi_gpu, Qi_d, numX_per_coil);
   
    wkf_timer_stop(timer);
    #if DEBUG_MODE
    printf("  Total CUDA runtime is %f\n", wkf_timer_time(timer));
    #endif
    wkf_timer_destroy(timer);


    if (params.useLUT){
       free(LUT);
    }

    if (no_zerocopy==0) {
        CUDA_SAFE_CALL(cudaFreeHost(samples));
        CUDA_SAFE_CALL(cudaFreeHost(gridData));
        CUDA_SAFE_CALL(cudaFreeHost(sampleDensity));
    }
    else {
        free(samples);
        free(gridData);
        free(sampleDensity);
    }

    CUDA_SAFE_CALL(cudaFree(gridData_crop_d));
    CUDA_SAFE_CALL(cudaFree(gridData_d));
 
}

    void 
computeQ_GPU(
    int numK, int numX,
    float *x_d, float *y_d, float *z_d,
    kValues_Q *kVals,
    float *t_d, float l, float tau,
    float *Qr_d, float *Qi_d,
    float *&Qr_gpu, float *&Qi_gpu) 
{
    unsigned int timerComputeGPU = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timerComputeGPU));
    CUT_SAFE_CALL(cutStartTimer(timerComputeGPU));

    int QGrids = numK / KERNEL_Q_K_ELEMS_PER_GRID;
    if (numK % KERNEL_Q_K_ELEMS_PER_GRID)
        QGrids++;
    int QxBlocks = numX / (KERNEL_Q_THREADS_PER_BLOCK * 
                           KERNEL_Q_X_ELEMS_PER_THREAD);
    int QyBlocks = 1;
    if (numX % (KERNEL_Q_THREADS_PER_BLOCK * KERNEL_Q_X_ELEMS_PER_THREAD))
        QxBlocks++;
    while (QxBlocks > 32768) {
        QyBlocks *= 2;
        if (QxBlocks % 2) {
            QxBlocks /= 2;
            QxBlocks++;
        } else {
            QxBlocks /= 2;
        }
    }
    dim3 DimQBlock(KERNEL_Q_THREADS_PER_BLOCK, 1);
    dim3 DimQGrid(QxBlocks, QyBlocks);
#ifdef DEBUG
    printf("Launch GPU Kernel: Grids %d, Blocks Per Grid (%d, %d), \
            Threads Per Block (%d, %d), K Elems Per Thread %d, X Elems \
            Per Thread %d\n",QGrids, DimQGrid.x, DimQGrid.y, DimQBlock.x, 
            DimQBlock.y, KERNEL_Q_K_ELEMS_PER_GRID, KERNEL_Q_X_ELEMS_PER_THREAD);
#endif
    for (int QGrid = 0; QGrid < QGrids; QGrid++) {
        // unsigned int timerGridGPU = 0;
        // CUT_SAFE_CALL(cutCreateTimer(&timerGridGPU));
        // CUT_SAFE_CALL(cutStartTimer(timerGridGPU));

        // Put the tile of K values into constant mem
        int QGridBase = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;
        kValues_Q *kValsTile = kVals + QGridBase;
        int numElems = MIN(KERNEL_Q_K_ELEMS_PER_GRID, numK - QGridBase);
        cudaMemcpyToSymbol(c, kValsTile, numElems * sizeof(kValues_Q), 0);

        ComputeQ_GPU <<< DimQGrid, DimQBlock >>> 
        (numK, numX, QGridBase, x_d, y_d, z_d, t_d, l, tau, Qr_d, Qi_d);
        CUT_CHECK_ERROR("ComputeQGPU failed!\n");

        // CUT_SAFE_CALL(cutStopTimer(timerGridGPU));
        // printf("Time to compute grid %d on GPU: %f (s)\n", QGrid,
        // cutGetTimerValue(timerGridGPU) / 1000.0);
        // CUT_SAFE_CALL(cutDeleteTimer(timerGridGPU));
    }

    mriCopyDeviceToHost(Qr_gpu, Qr_d, numX);
    mriCopyDeviceToHost(Qi_gpu, Qi_d, numX);
    #ifdef DEBUG
    CUT_SAFE_CALL(cutStopTimer(timerComputeGPU));
    printf("Time to compute Q on GPU: %f (s)\n", cutGetTimerValue(timerComputeGPU) / 1000.0);
    CUT_SAFE_CALL(cutDeleteTimer(timerComputeGPU));
    #endif
}



    int
toeplitz_computeQ_GPU(const char *data_directory, const float ntime_segments, 
                      int Nx, int Ny, int Nz, const char *Q_full_filename,
                      float **Qr_gpu, float **Qi_gpu, 
                      const bool enable_direct, const bool enable_gridding,
                      float gridOS,
                      const bool enable_writeQ)
{   
    /*(Nx,Ny,Nz) is the actual MR image size.*/

    float tau;
    float version = 0.2f;
    // 'numX' and 'numK' are not used, only dummy variables for 'inputData_Q(.)'.
    // Q matrices are a per_coil quantity, so SENSE does not make thru this level.
    int numX, numK, numX_per_coil, numK_per_coil, ncoils, nslices;

    float *kx, *ky, *kz, *t;
    float *x, *y, *z, *phiR, *phiI;
    float *x_d, *y_d, *z_d, *phiR_d, *phiI_d, *t_d;
    float *phiMag_gpu /*, **Qr_gpu, **Qi_gpu*/;
    float *phiMag_d;

    #ifdef DEBUG
    unsigned int timerApp = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timerApp));
    CUT_SAFE_CALL(cutStartTimer(timerApp));
    #endif

    //#ifdef DEBUG
    //printf("======== Step 1. Compute Q matrices: Q[%d] ========\n",static_cast<int>(ntime_segments+1));
    //#else
    //printf("======== Step 1. Compute Q matrices: Q[%d] ========\n",static_cast<int>(ntime_segments+1));
    //#endif

    /* Read in data */
    // After 'inputData_Q(.)', numX_per_coil is the Q matrix 
    // dimension, not the actual MR image, b/c it is ixQ.dat
    // that gets read in by 'inputData_Q(.)'.
    inputData_Q( data_directory, version, numK, numK_per_coil, ncoils, nslices, 
                 numX, numX_per_coil, kx, ky, kz, x, y, z, t, phiR, phiI );

    tau = (ntime_segments > 0.0f) ? ((t[numK_per_coil - 1] - t[0]) / ntime_segments) 
                : (t[numK_per_coil - 1] - t[0]);

    unsigned int timerKernel = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timerKernel));
    CUT_SAFE_CALL(cutStartTimer(timerKernel));

    /* Create GPU data structures */
    createDataStructs(numK_per_coil, numX_per_coil, phiMag_gpu);

    #ifdef DEBUG
    printf("Start GPU computation!\n");
    #endif

    /* Setup GPU memory */
    phiR_d   = mriNewGpu<float>(numK_per_coil);
    phiI_d   = mriNewGpu<float>(numK_per_coil);
    phiMag_d = mriNewGpu<float>(numK_per_coil);
    t_d      = mriNewGpu<float>(numK_per_coil);
    x_d      = mriNewGpu<float>(numX_per_coil);
    y_d      = mriNewGpu<float>(numX_per_coil);
    z_d      = mriNewGpu<float>(numX_per_coil);
    mriCopyHostToDevice<float>(phiR_d, phiR, numK_per_coil);
    mriCopyHostToDevice<float>(phiI_d, phiI, numK_per_coil);
    mriCopyHostToDevice<float>(phiMag_d, phiMag_gpu, numK_per_coil);
    mriCopyHostToDevice<float>(t_d, t, numK_per_coil);
    mriCopyHostToDevice<float>(x_d, x, numX_per_coil);
    mriCopyHostToDevice<float>(y_d, y, numX_per_coil);
    mriCopyHostToDevice<float>(z_d, z, numX_per_coil);

    // Zero out initial values of Qr and Qi.
    // GPU views these arrays as initialized (cleared) accumulators.
    #if 0 // very memory inefficient
    float **Qr_d, **Qi_d;
    Qr_d = (float **) malloc((((int) ntime_segments) + 1) * sizeof(float *));
    Qi_d = (float **) malloc((((int) ntime_segments) + 1) * sizeof(float *));

    for (int l = 0; l <= ((int) ntime_segments); l++) {
        Qr_d[l] = mriNewGpu<float>(numX_per_coil);
        Qi_d[l] = mriNewGpu<float>(numX_per_coil);
        mriCopyHostToDevice<float>(Qr_d[l], Qr_gpu[l], numX_per_coil);
        mriCopyHostToDevice<float>(Qi_d[l], Qi_gpu[l], numX_per_coil);
    }
    #else
    float *Qr_d, *Qi_d;
    Qr_d = mriNewGpu<float>(numX_per_coil);
    Qi_d = mriNewGpu<float>(numX_per_coil);
    #endif

    /* Pre-compute the values of phiMag on the GPU */
    computePhiMag_GPU(numK_per_coil, phiR_d, phiI_d, phiMag_d, phiMag_gpu);

    unsigned int timerSetupK = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timerSetupK));
    CUT_SAFE_CALL(cutStartTimer(timerSetupK));

    kValues_Q *kVals = (kValues_Q *)calloc(numK_per_coil, sizeof(kValues_Q));

    float *kx_d, *ky_d, *kz_d;
    kx_d = mriNewGpu<float>(numK_per_coil);
    ky_d = mriNewGpu<float>(numK_per_coil);
    kz_d = mriNewGpu<float>(numK_per_coil);
    mriCopyHostToDevice<float>(kx_d, kx, numK_per_coil);
    mriCopyHostToDevice<float>(ky_d, ky, numK_per_coil);
    mriCopyHostToDevice<float>(kz_d, kz, numK_per_coil);

    MarshallScale_GPU(numK_per_coil, kx_d, ky_d, kz_d, phiMag_d, kVals);

    mriDeleteGpu<float>(kx_d);
    mriDeleteGpu<float>(ky_d);
    mriDeleteGpu<float>(kz_d);

    #ifdef DEBUG
    CUT_SAFE_CALL(cutStopTimer(timerSetupK));
    printf("Time to setup K: %f (s)\n", cutGetTimerValue(timerSetupK) / 1000.0);
    CUT_SAFE_CALL(cutDeleteTimer(timerSetupK));
    #endif

    #ifdef DEBUG
    #else
      msg(MSG_PLAIN, "  GPU: Q: ");
    #endif


    for (int l = 0; l <= ((int) ntime_segments); l++) {
        
        /* Compute Qreal and Qimag on the GPU */
        if(enable_direct) {
           computeQ_GPU(numK_per_coil, numX_per_coil,
                        x_d, y_d, z_d, kVals,
                        t_d, l, tau,
                        Qr_d, Qi_d, Qr_gpu[l], Qi_gpu[l]);
        } else { 
           computeQ_GPU_Grid(numK_per_coil, numX_per_coil, 
                             kx, ky, kz, Nx, Ny, Nz, t, t_d, l, tau,
                             gridOS,
                             Qr_d, Qi_d, Qr_gpu[l], Qi_gpu[l]);
        }

        #ifdef DEBUG
        #else
          msg(MSG_PLAIN, "[%d].",l);
        #endif
    }        

    msg(MSG_PLAIN,"\n");
    free(kVals);

    /* Release memory on GPU */
    mriDeleteGpu<float>(x_d);    
    mriDeleteGpu<float>(y_d);    
    mriDeleteGpu<float>(z_d);    
    mriDeleteGpu<float>(t_d);
    #if 0 //very memory INEFFICIENT
    for (int l = 0; l <= ((int) ntime_segments); l++) {
        mriDeleteGpu<float>(Qr_d[l]);
        mriDeleteGpu<float>(Qi_d[l]);
    }
    free(Qr_d);
    free(Qi_d);
    #else
    mriDeleteGpu<float>(Qr_d);
    mriDeleteGpu<float>(Qi_d);
    #endif
    mriDeleteGpu<float>(phiI_d);
    mriDeleteGpu<float>(phiR_d);
    mriDeleteGpu<float>(phiMag_d);

    CUT_SAFE_CALL(cutStopTimer(timerKernel));
    printf("  GPU: Q Kernel Time: %f (ms)\n", cutGetTimerValue(timerKernel));
    CUT_SAFE_CALL(cutDeleteTimer(timerKernel));

    #ifdef DEBUG
    printf("======== output ========\n");
    #else
    printf("  GPU: Exporting results.\n");
    #endif
    /* Write Q to file */
    if(enable_writeQ)
    {
       outputData(Q_full_filename, Qr_gpu, Qi_gpu, numX_per_coil, ntime_segments);
    }

    free(kx);
    free(ky);
    free(kz);
    free(x);
    free(y);
    free(z);
    free(t);
    free(phiR);
    free(phiI);
    free(phiMag_gpu);

    #ifdef DEBUG
    CUT_SAFE_CALL(cutStopTimer(timerApp));
    printf("GPU App Time: %f (s)\n", cutGetTimerValue(timerApp) / 1000.0);
    CUT_SAFE_CALL(cutDeleteTimer(timerApp));
    #endif

    return 0;
}
