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

    File Name   [computeFH.cmem.cu]

    Synopsis    [CUDA code for creating the FH  data structure for fast 
                 convolution-based Hessian multiplication for arbitrary 
                 k-space trajectories.]

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

#include <stdio.h>
#include <math.h>
#include <cutil.h>
#include <assert.h>

// Project header files
#include <tools.h>
#include <structures.h>
#include <utils.h>
#include <gridding.h>
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
#define KERNEL_RHO_PHI_THREADS_PER_BLOCK 256
#define KERNEL_MARSHALL_SCALE_TPB 256
#define KERNEL_FH_THREADS_PER_BLOCK  320
#define KERNEL_FH_K_ELEMS_PER_GRID  2048
#define KERNEL_FH_X_ELEMS_PER_THREAD 1

struct kTrajectory {
    float Kx;
    float Ky;
    float Kz;
    float t;
};

struct kValues_FH {
    float RhoPhiR;
    float RhoPhiI;
};

// Cannot use 4096 elems per array per grid - execution fails.
__constant__ static __device__ kValues_FH cV[KERNEL_FH_K_ELEMS_PER_GRID];
__constant__ static __device__ kTrajectory cT[KERNEL_FH_K_ELEMS_PER_GRID];

__global__ void
ComputeRhoPhiGPU(int numK,
                 float *phiR, float *phiI,
                 float *dR, float *dI,
                 float *realRhoPhi, float *imagRhoPhi) {
    int indexK = blockIdx.y * (gridDim.x * KERNEL_RHO_PHI_THREADS_PER_BLOCK) 
               + blockIdx.x * KERNEL_RHO_PHI_THREADS_PER_BLOCK + threadIdx.x;
    if (indexK < numK) {
        float rPhiR = phiR[indexK];
        float rPhiI = phiI[indexK];
        float rDR = dR[indexK];
        float rDI = dI[indexK];
        realRhoPhi[indexK] = rPhiR * rDR + rPhiI * rDI;
        imagRhoPhi[indexK] = rPhiR * rDI - rPhiI * rDR;
    }
}

__global__ void
MarshallScaleGPU_kTrajectory(int numK_per_coil, float *kx, float *ky, float *kz,
                             float *t, kTrajectory *kTraj)
{
    int indexK = blockIdx.y * (gridDim.x * KERNEL_MARSHALL_SCALE_TPB) 
               + blockIdx.x * KERNEL_MARSHALL_SCALE_TPB + threadIdx.x;
    if (indexK < numK_per_coil) {
        kTraj[indexK].Kx = PIx2 * kx[indexK];
        kTraj[indexK].Ky = PIx2 * ky[indexK];
        kTraj[indexK].Kz = PIx2 * kz[indexK];
        kTraj[indexK].t = t[indexK];
        // NO need to multiply kTraj[indexK] with PIx2 !
    }
}
__global__ void
MarshallScaleGPU_kValues_FH(int numK, float *realRhoPhi, float *imagRhoPhi, 
                         kValues_FH *kVals) 
{
    int indexK = blockIdx.y * (gridDim.x * KERNEL_MARSHALL_SCALE_TPB) 
               + blockIdx.x * KERNEL_MARSHALL_SCALE_TPB + threadIdx.x;
    if (indexK < numK) {
        kVals[indexK].RhoPhiR = realRhoPhi[indexK];
        kVals[indexK].RhoPhiI = imagRhoPhi[indexK];
    }
}
// Brute Force way of computing (F^H d) on GPU
    __global__ void
ComputeFH_GPU_BF(
    int numK_per_coil, int numX_per_coil, int kGlobalIndex,
    float *x, float *y, float *z,
    float *t_d, float *fm_d,
    float *outR, float *outI)
{
    float sX;
    float sY;
    float sZ;
    float sOutR;
    float sOutI;

    // Determine the element of the X arrays computed by this thread
    int xIndex = blockIdx.y * (gridDim.x * KERNEL_FH_THREADS_PER_BLOCK) 
               + blockIdx.x * KERNEL_FH_THREADS_PER_BLOCK + threadIdx.x;
    if (xIndex < numX_per_coil) {
        // Read block's X values from global mem to shared mem
        sX = x[xIndex];
        sY = y[xIndex];
        sZ = z[xIndex];
        sOutR = outR[xIndex];
        sOutI = outI[xIndex];
        float a_fm_value = fm_d[xIndex];

        // Loop over all elements of K in constant mem to compute a partial 
        // value for X.
        for (  int kIndex = 0; (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && 
               (kGlobalIndex < numK_per_coil); kIndex++, kGlobalIndex++  ) 
        {
           float expArg = (cT[kIndex].Kx * sX +
                           cT[kIndex].Ky * sY +
                           cT[kIndex].Kz * sZ +
                           cT[kIndex].t  * a_fm_value);

           float cosArg = cos(expArg);
           float sinArg = sin(expArg);

           sOutR += (cV[kIndex].RhoPhiR * cosArg - cV[kIndex].RhoPhiI * sinArg);
           sOutI += (cV[kIndex].RhoPhiI * cosArg + cV[kIndex].RhoPhiR * sinArg);
        }

        outR[xIndex] = sOutR;
        outI[xIndex] = sOutI;
    }
}

    static void 
inputData_FH(
    const char *input_folder_path,
    float version, int&numK, int&numK_per_coil,
    int&ncoils, int&nslices, int&numX, int&numX_per_coil,
    float *&kx, float *&ky, float *&kz,
    float *&x, float *&y, float *&z,
    float *&fm, float *&t,
    float *&phiR, float *&phiI,
    float *&dR, float *&dI,
    float *&sensi_r, float *&sensi_i)
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
    string kdata_r_fn = input_folder_path;
    kdata_r_fn = kdata_r_fn + "/kdata_r.dat";
    string kdata_i_fn = input_folder_path;
    kdata_i_fn = kdata_i_fn + "/kdata_i.dat";

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

      dR = readDataFile_JGAI(kdata_r_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      dI = readDataFile_JGAI(kdata_i_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
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

      dR = readDataFile_JGAI_10(kdata_r_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
      dI = readDataFile_JGAI_10(kdata_i_fn.c_str(), version, numK_per_coil, ncoils, nslices, numK);
    }

    if (1 == ncoils) {
        for (int i = 0; i < (numX_per_coil * ncoils); i++) {
            sensi_r[i] = 1.0f;
            sensi_i[i] = 0.0f;
        }
    }
}

    static void 
outputData( 
    const char *fName, float *outR_gpu, float *outI_gpu, int numX_per_coil)
{
    FILE *fid = fopen(fName, "w");
    fwrite(outR_gpu, sizeof(float), numX_per_coil, fid);
    fwrite(outI_gpu, sizeof(float), numX_per_coil, fid);
    fclose(fid);
}


    static void 
createDataStructs(
    int numK, int numX_per_coil,
    float *&realRhoPhi, float *&imagRhoPhi) 
{
    realRhoPhi = (float * ) calloc(numK, sizeof(float));
    imagRhoPhi = (float * ) calloc(numK, sizeof(float));
}

    void 
computeRhoPhi_GPU(int numK,
    float *phiR_d, float *phiI_d, float *dR_d, float *dI_d,
    float *realRhoPhi_d, float *imagRhoPhi_d,
    float *&realRhoPhi_gpu, float *&imagRhoPhi_gpu)
{
    
    unsigned int timerComputeRhoPhi = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timerComputeRhoPhi));
    CUT_SAFE_CALL(cutStartTimer(timerComputeRhoPhi));

    int rhoPhixBlocks = numK / KERNEL_RHO_PHI_THREADS_PER_BLOCK;
    int rhoPhiyBlocks = 1;
    if (numK % KERNEL_RHO_PHI_THREADS_PER_BLOCK)
        rhoPhixBlocks++;

    while (rhoPhixBlocks > 32768) {
        rhoPhiyBlocks *= 2;
        if (rhoPhixBlocks % 2) {
            rhoPhixBlocks /= 2;
            rhoPhixBlocks++;
        } else {
            rhoPhixBlocks /= 2;
        }
    }


    dim3 DimRhoPhiBlock(KERNEL_RHO_PHI_THREADS_PER_BLOCK, 1);
    dim3 DimRhoPhiGrid(rhoPhixBlocks, rhoPhiyBlocks);
    #ifdef DEBUG
    printf("Launch RhoPhi Kernel on GPU: Blocks (%d, %d), Threads Per Block %d\n",
           rhoPhixBlocks, rhoPhiyBlocks, KERNEL_RHO_PHI_THREADS_PER_BLOCK);
    #endif
    ComputeRhoPhiGPU <<< DimRhoPhiGrid, DimRhoPhiBlock >>>
    (numK, phiR_d, phiI_d, dR_d, dI_d, realRhoPhi_d, imagRhoPhi_d);

    CUDA_SAFE_CALL(cudaMemcpy(realRhoPhi_gpu, realRhoPhi_d, 
                   numK * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(imagRhoPhi_gpu, imagRhoPhi_d, 
                   numK * sizeof(float), cudaMemcpyDeviceToHost));

    #ifdef DEBUG
    CUT_CHECK_ERROR("ComputeRhoPhiGPU failed!\n");
    CUT_SAFE_CALL(cutStopTimer(timerComputeRhoPhi));
    printf("Time to compute RhoPhi on GPU: %f (s)\n", 
            cutGetTimerValue(timerComputeRhoPhi) / 1000.0);
    CUT_SAFE_CALL(cutDeleteTimer(timerComputeRhoPhi));
    #endif
}

    void 
MarshallScale_GPU( 
    int numK_per_coil, int ncoils, float *kx_d, float *ky_d, 
    float *kz_d, float *t_d, float *rhoPhiR_d, float *rhoPhiI_d, 
    kValues_FH *&kVals, kTrajectory *&kTraj )
{
    unsigned int timerMarshallScale = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timerMarshallScale));
    CUT_SAFE_CALL(cutStartTimer(timerMarshallScale));

    int numK = numK_per_coil * ncoils;

    kValues_FH *kVals_d = mriNewGpu<kValues_FH>(numK);
    kTrajectory *kTraj_d = mriNewGpu<kTrajectory>(numK_per_coil);

    int blocks_kVals_x = numK / KERNEL_MARSHALL_SCALE_TPB;
    int blocks_kVals_y = 1;
    if (numK % KERNEL_MARSHALL_SCALE_TPB)
        blocks_kVals_x++;

    while (blocks_kVals_x > 32768) {
        blocks_kVals_y *= 2;
        if (blocks_kVals_x % 2) {
            blocks_kVals_x /= 2;
            blocks_kVals_x++;
        } else {
            blocks_kVals_x /= 2;
        }
    }



    dim3 dimBlock_kVals(KERNEL_MARSHALL_SCALE_TPB, 1);
    dim3 dimGrid_kVals(blocks_kVals_x, blocks_kVals_y);
    #ifdef DEBUG
    printf("Launch MarshallScale Kernel on GPU: Blocks (%d, %d), Threads \
            Per Block %d\n",blocks_kVals_x, blocks_kVals_y, KERNEL_MARSHALL_SCALE_TPB);
    #endif
    MarshallScaleGPU_kValues_FH <<< dimGrid_kVals, dimBlock_kVals >>> 
                           (numK, rhoPhiR_d, rhoPhiI_d, kVals_d);

    mriCopyDeviceToHost<kValues_FH>(kVals, kVals_d, numK);
    mriDeleteGpu<kValues_FH>(kVals_d);

    int blocks_kTraj_x = numK_per_coil / KERNEL_MARSHALL_SCALE_TPB;
    int blocks_kTraj_y = 1;
    if (numK_per_coil % KERNEL_MARSHALL_SCALE_TPB)
        blocks_kTraj_x++;

    while (blocks_kTraj_x > 32768) {
        blocks_kTraj_y *= 2;
        if (blocks_kTraj_x % 2) {
            blocks_kTraj_x /= 2;
            blocks_kTraj_x++;
        } else {
            blocks_kTraj_x /= 2;
        }
    }


    dim3 dimBlock_kTraj(KERNEL_MARSHALL_SCALE_TPB, 1);
    dim3 dimGrid_kTraj(blocks_kTraj_x, blocks_kTraj_y);
    #ifdef DEBUG
    printf("Launch MarshallScale_kTrajectory Kernel on GPU: Blocks (%d, %d), \
            Threads Per Block %d\n", blocks_kTraj_x, blocks_kTraj_y, KERNEL_MARSHALL_SCALE_TPB);
    #endif
    MarshallScaleGPU_kTrajectory <<< dimGrid_kTraj, dimBlock_kTraj >>> 
                               (numK_per_coil, kx_d, ky_d, kz_d, t_d, kTraj_d);


    mriCopyDeviceToHost<kTrajectory>(kTraj,kTraj_d,numK_per_coil);
    mriDeleteGpu<kTrajectory>(kTraj_d);

    #ifdef DEBUG
    CUT_CHECK_ERROR("MarshallScale failed!\n");
    CUT_SAFE_CALL(cutStopTimer(timerMarshallScale));
    printf("Time to marshall and scale data on GPU: %f (s)\n", 
            cutGetTimerValue(timerMarshallScale) / 1000.0);
    CUT_SAFE_CALL(cutDeleteTimer(timerMarshallScale));
    #endif
}



// Gridding way of computing FhD

    void 
computeFH_GPU_Grid(
    int numK_per_coil, float *kx, float *ky, float *kz,
    float *dR, float *dI, int Nx, int Ny, int Nz,
    float *t, float *t_d, float l, float tau,
    float gridOS,
    float *outR_d, float *outI_d)
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

///*
    // grid_size in xy-axis has to be divisible-by-two:
    //       (required by the cropImageRegion)
    // grid_size in z-axis has to be devisible-by-four:
    //       (required by the function gridding_GPU_3D(.))
    if(1==Nz) {        
        //round grid size (xy-axis) to the next divisible-by-two.
        gridOS = 2.0f * ceil((gridOS * (float)Nx) / 2.0f) / (float) Nx;
    }
    else {
        //round grid size (z-axis) to the next divisible-by-four.
        gridOS = 4.0f * ceil((gridOS * (float)Nz) / 4.0f) / (float) Nz;
    }
// */

    parameters params;
    params.sync=0;
    params.binsize=128;

    params.useLUT = 0;
    params.kernelWidth = kernelWidth;
    params.gridOS = gridOS;
    params.imageSize[0] = Nx;//gridSize is gridOS times larger than imageSize.
    params.imageSize[1] = Ny;
    params.imageSize[2] = Nz;
    params.gridSize[0]  = (ceil)(gridOS*(float)Nx);
    params.gridSize[1]  = (ceil)(gridOS*(float)Ny);
    if(params.gridSize[0]%2)//3D case, gridOS is adjusted on the z dimension:
        params.gridSize[0] += 1;//That why we need to make sure here that the xy 
    if(params.gridSize[1]%2)//dimensions have even sizes.
        params.gridSize[1] += 1;
    params.gridSize[2]  = (Nz==1)?Nz:((ceil)(gridOS*(float)Nz));// 2D or 3D
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

          samples[i].kX = kx[i];
          samples[i].kY = ky[i];
          samples[i].kZ = kz[i];
 
          samples[i].real = dR[i];
          samples[i].imag = dI[i];
 
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
      cuda_fft2shift_grid(gridData_d, gridData_d, params.gridSize[0], 
                          params.gridSize[1], 0);
    }
    else
    {
      cuda_fft3shift_grid(gridData_d, gridData_d, params.gridSize[0], 
                          params.gridSize[1], params.gridSize[2],0);
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

outputData("/home/UIUC/jgai/Desktop/FhD_nady.file",output_r,output_i,gridNumElems);
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
      crop_center_region2d(gridData_crop_d, gridData_d, 
                           params.imageSize[0], params.imageSize[1],
                           params.gridSize[0], params.gridSize[1]);
    }
    else
    {
      crop_center_region3d(gridData_crop_d, gridData_d, 
                           params.imageSize[0], params.imageSize[1], params.imageSize[2],
                           params.gridSize[0], params.gridSize[1], params.gridSize[2]);
    }


    // deapodization
    if(Nz==1)
    {
      deapodization2d(gridData_crop_d, gridData_crop_d,
                      Nx, Ny, kernelWidth, beta, params.gridOS);
    }
    else
    {
      deapodization3d(gridData_crop_d, gridData_crop_d,
                      Nx, Ny, Nz, kernelWidth, beta, params.gridOS);
    }


//Jiading GAI - DEBUG
#if 0
deinterleave_data3d(gridData_crop_d, outR_d, outI_d, Nx, Ny, Nz);
   

float *output_r = (float*) calloc (imageNumElems, sizeof(float));
float *output_i = (float*) calloc (imageNumElems, sizeof(float));

CUDA_SAFE_CALL(cudaMemcpy(output_r, outR_d, imageNumElems*sizeof(float), cudaMemcpyDeviceToHost));
CUDA_SAFE_CALL(cudaMemcpy(output_i, outI_d, imageNumElems*sizeof(float), cudaMemcpyDeviceToHost));

outputData("/home/UIUC/jgai/Desktop/projects-mrfil-linux/impatient_v2.0b/mriData/64_64_16_Joe_format/FhD.file",output_r,output_i,imageNumElems);
free(output_r);
free(output_i);
exit(1);
#endif

    // Copy results from gridData_crop_d to outR_d and outI_d
    // gridData_crop_d is cufftComplex, interleaving
    // De-interleaving the data from cufftComplex to outR_d-and-outI_d
    if(Nz==1)
    {
       deinterleave_data2d(gridData_crop_d, outR_d, outI_d, Nx, Ny);
    }
    else
    {
       deinterleave_data3d(gridData_crop_d, outR_d, outI_d, Nx, Ny, Nz);
    }
   
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

// Brute Force way of computing FhD
    void 
computeFH_GPU_BF(
    int numK_per_coil, int numX_per_coil, 
    float *x_d, float *y_d, float *z_d,
    kValues_FH *kVals, kTrajectory *kTraj,
    float *t_d, float *fm_d,
    float *outR_d, float *outI_d)
{
    unsigned int timerComputeGPU = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timerComputeGPU));
    CUT_SAFE_CALL(cutStartTimer(timerComputeGPU));

    int FHGrids = numK_per_coil / KERNEL_FH_K_ELEMS_PER_GRID;
    if (numK_per_coil % KERNEL_FH_K_ELEMS_PER_GRID)
        FHGrids++;
    int FHxBlocks = numX_per_coil / (KERNEL_FH_THREADS_PER_BLOCK * KERNEL_FH_X_ELEMS_PER_THREAD);
    int FHyBlocks = 1;
    if (numX_per_coil % (KERNEL_FH_THREADS_PER_BLOCK * KERNEL_FH_X_ELEMS_PER_THREAD))
        FHxBlocks++;
    while (FHxBlocks > 32768) {
        FHyBlocks *= 2;
        if (FHxBlocks % 2) {
            FHxBlocks /= 2;
            FHxBlocks++;
        } else {
            FHxBlocks /= 2;
        }
    }
    dim3 DimFHBlock(KERNEL_FH_THREADS_PER_BLOCK, 1);
    dim3 DimFHGrid(FHxBlocks, FHyBlocks);

    #ifdef DEBUG
    printf("Launch GPU Kernel: Grids %d, Blocks Per Grid (%d, %d), Threads \
            Per Block (%d, %d), K Elems Per Thread %d, X Elems Per Thread \
            %d\n", FHGrids, DimFHGrid.x, DimFHGrid.y, DimFHBlock.x, DimFHBlock.y, \
            KERNEL_FH_K_ELEMS_PER_GRID, KERNEL_FH_X_ELEMS_PER_THREAD);
    #endif
    for (int FHGrid = 0; FHGrid < FHGrids; FHGrid++) {
        // unsigned int timerGridGPU = 0;
        // CUT_SAFE_CALL(cutCreateTimer(&timerGridGPU));
        // CUT_SAFE_CALL(cutStartTimer(timerGridGPU));

        // Put the tile of K values into constant mem
        int FHGridBase = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;
        kTrajectory *kTrajTile = kTraj + FHGridBase;
        kValues_FH *kValsTile = kVals + FHGridBase;
        int numElems = MIN(KERNEL_FH_K_ELEMS_PER_GRID, numK_per_coil - FHGridBase);
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(cV, kValsTile, numElems * sizeof(kValues_FH), 0));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(cT, kTrajTile, numElems * sizeof(kTrajectory), 0));

        ComputeFH_GPU_BF <<< DimFHGrid, DimFHBlock >>>
        (numK_per_coil, numX_per_coil, FHGridBase, x_d, y_d, z_d, t_d, fm_d, outR_d, outI_d);
        CUT_CHECK_ERROR("ComputeFH_GPU failed!\n");

        // CUT_SAFE_CALL(cutStopTimer(timerGridGPU));
        // printf("Time to compute grid %d on GPU: %f (s)\n", FHGrid,
        // cutGetTimerValue(timerGridGPU) / 1000.0);
        // CUT_SAFE_CALL(cutDeleteTimer(timerGridGPU));
    }
    #ifdef DEBUG
    CUT_SAFE_CALL(cutStopTimer(timerComputeGPU));
    printf("Time to compute FH on GPU: %f (s)\n", cutGetTimerValue(timerComputeGPU) / 1000.0);
    CUT_SAFE_CALL(cutDeleteTimer(timerComputeGPU));
    #endif
}

#define KERNEL_HADAMARD_THREADS_PER_BLOCK 256
// result = a .* b, used by time segmentation
__global__ void Hadamard_prod_GPU_Kernel(float *result_r, float *result_i,
                                         float *a_r_d, float *a_i_d,
                                         float *b_r_d, float *b_i_d, int len)
{
    int index = blockIdx.x * KERNEL_HADAMARD_THREADS_PER_BLOCK + threadIdx.x;
    float tmp_r, tmp_i;
    float a_r = a_r_d[index];
    float a_i = a_i_d[index];
    float b_r = b_r_d[index];
    float b_i = b_i_d[index];
    if (index < len) {
        tmp_r = a_r * b_r - a_i * b_i;
        tmp_i = a_r * b_i + a_i * b_r;

        result_r[index] = tmp_r;
        result_i[index] = tmp_i;
    }
}
void Hadamard_prod_GPU(float *result_r, float *result_i,
                       float *a_r_d, float *a_i_d,
                       float *b_r_d, float *b_i_d, int len)
{
    int HadamardBlocks = len / KERNEL_HADAMARD_THREADS_PER_BLOCK;
    if (len % KERNEL_HADAMARD_THREADS_PER_BLOCK)
        HadamardBlocks++;
    dim3 DimHadamard_Block(KERNEL_HADAMARD_THREADS_PER_BLOCK, 1);
    dim3 DimHadamard_Grid(HadamardBlocks, 1);

    Hadamard_prod_GPU_Kernel <<< DimHadamard_Grid, DimHadamard_Block >>>
    (result_r, result_i, a_r_d, a_i_d, b_r_d, b_i_d, len);
}

// result = a^H .* b, used by SENSE
__global__ void Hadamard_prod_Conj_GPU_Kernel(float *result_r, float *result_i,
                                              float *a_r_d, float *a_i_d,
                                              float *b_r_d, float *b_i_d, int len)
{
    int index = blockIdx.x * KERNEL_HADAMARD_THREADS_PER_BLOCK + threadIdx.x;
    float tmp_r, tmp_i;
    float a_r = a_r_d[index];
    float a_i = a_i_d[index];
    float b_r = b_r_d[index];
    float b_i = b_i_d[index];
    if (index < len) {
        tmp_r = a_r * b_r + a_i * b_i;
        tmp_i = a_r * b_i - a_i * b_r;

        result_r[index] = tmp_r;
        result_i[index] = tmp_i;
    }
}
void Hadamard_prod_Conj_GPU(float *result_r, float *result_i,
                            float *a_r_d, float *a_i_d,
                            float *b_r_d, float *b_i_d, int len)
{
    int HadamardBlocks = len / KERNEL_HADAMARD_THREADS_PER_BLOCK;
    if (len % KERNEL_HADAMARD_THREADS_PER_BLOCK)
        HadamardBlocks++;
    dim3 DimHadamard_Block(KERNEL_HADAMARD_THREADS_PER_BLOCK, 1);
    dim3 DimHadamard_Grid(HadamardBlocks, 1);

    Hadamard_prod_Conj_GPU_Kernel <<< DimHadamard_Grid, DimHadamard_Block >>>
    (result_r, result_i, a_r_d, a_i_d, b_r_d, b_i_d, len);
}

#define ADD_GPU_THREADS_PER_BLOCKS 256
__global__ void add_GPU_Kernel(float* result_r, float* result_i, 
             float* a_r_d, float* a_i_d, 
             float* b_r_d, float* b_i_d, int len)
{
    int index = blockIdx.x * ADD_GPU_THREADS_PER_BLOCKS + threadIdx.x;
    if(index<len)
    {
      result_r[index] = a_r_d[index] + b_r_d[index];
      result_i[index] = a_i_d[index] + b_i_d[index];
    }
}
void add_GPU(float* result_r, float* result_i, 
             float* a_r_d, float* a_i_d, 
             float* b_r_d, float* b_i_d, int len)
{
   int addBlocks = len / ADD_GPU_THREADS_PER_BLOCKS;
   if(len % ADD_GPU_THREADS_PER_BLOCKS)
       addBlocks++;
   dim3 DimAdd_Block(ADD_GPU_THREADS_PER_BLOCKS, 1);
   dim3 DimAdd_Grid(addBlocks, 1);

   add_GPU_Kernel <<<DimAdd_Grid, DimAdd_Block>>>
       (result_r, result_i, a_r_d, a_i_d, b_r_d, b_i_d, len);  
}

#define SET_ZERO_THREADS_PER_BLOCK 256
    __global__ void 
setZero_GlobalMemory_Kernel(float *input, int len)
{
    int index = blockIdx.x * SET_ZERO_THREADS_PER_BLOCK + threadIdx.x;
    if (index < len) {
        input[index] = 0.0f;
    }
}
    void 
setZero_GlobalMemory(float *input, int len)
{
    int setZeroBlocks = len / SET_ZERO_THREADS_PER_BLOCK;
    if (len % SET_ZERO_THREADS_PER_BLOCK)
        setZeroBlocks++;
    dim3 DimsetZero_Block(SET_ZERO_THREADS_PER_BLOCK, 1);
    dim3 DimsetZero_Grid(setZeroBlocks, 1);

    setZero_GlobalMemory_Kernel <<< DimsetZero_Grid, DimsetZero_Block >>>
    (input, len);
}

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

static void 
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
        ComputeExpFM_GPU_Kernel<<<DimExpFM_Grid, DimExpFM_Block>>>
        (expfm_r_d, expfm_i_d, fm_d, l, tau, t0, len);
    } else {
        ComputeExpFM_GPU_Conj_Kernel<<<DimExpFM_Grid, DimExpFM_Block>>>
        (expfm_r_d, expfm_i_d, fm_d, l, tau, t0, len);
    }
}

// Gridding way of computing AhD (plus, time segmentation)
    void 
computeAH_GPU_Grid(
    int numK_per_coil, int c, float *kx, float *ky, float *kz,
    float *dR, float *dI, int Nx, int Ny, int Nz,
    float *t, float *t_d, float *fm_d, int L, float tau, float t0,
    float gridOS,
    float *outR_d, float *outI_d)
{
    //int conjYes = 1;
    int conjNo = 0;
    int numX_per_single_coil = Nx * Ny * Nz;

    setZero_GlobalMemory(outR_d, numX_per_single_coil);
    setZero_GlobalMemory(outI_d, numX_per_single_coil);

    float *tmp_r_d, *tmp_i_d;
    CUDA_SAFE_CALL(cudaMalloc((void**)&tmp_r_d, numX_per_single_coil*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tmp_i_d, numX_per_single_coil*sizeof(float)));

    setZero_GlobalMemory(tmp_r_d, numX_per_single_coil);
    setZero_GlobalMemory(tmp_i_d, numX_per_single_coil);

    float *expfm_r_d, *expfm_i_d;
    CUDA_SAFE_CALL(cudaMalloc((void**)&expfm_r_d, numX_per_single_coil*sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&expfm_i_d, numX_per_single_coil*sizeof(float)));

    for (int l = 0; l <= ((int) L); l++) {
       ///*
       setZero_GlobalMemory(tmp_r_d, numX_per_single_coil);
       setZero_GlobalMemory(tmp_i_d, numX_per_single_coil);

       computeFH_GPU_Grid(numK_per_coil, kx, ky, kz, dR+c*numK_per_coil, 
                          dI+c*numK_per_coil, Nx, Ny, Nz, t, t_d, l, tau, 
                          gridOS,
                          tmp_r_d, tmp_i_d);

       ComputeExpFM_GPU(expfm_r_d, expfm_i_d, fm_d, l, tau, t0, 
                        numX_per_single_coil, conjNo);

       Hadamard_prod_GPU(tmp_r_d, tmp_i_d, tmp_r_d, tmp_i_d, expfm_r_d, 
                         expfm_i_d, numX_per_single_coil);

       add_GPU(outR_d, outI_d, outR_d, outI_d, tmp_r_d, tmp_i_d, numX_per_single_coil);
       // */
    }
    
    CUDA_SAFE_CALL(cudaFree(tmp_r_d));
    CUDA_SAFE_CALL(cudaFree(tmp_i_d));
    CUDA_SAFE_CALL(cudaFree(expfm_r_d));
    CUDA_SAFE_CALL(cudaFree(expfm_i_d));
}


// Gridding way of computing par_AhD
    void 
parcomputeAH_GPU_Grid(
    int numK_per_coil, int numX_per_coil, int ncoils,
    float *kx, float *ky, float *kz,
    float *dR, float *dI, int Nx, int Ny, int Nz,
    float *sensi_r_d, float *sensi_i_d,
    float *fm_d, float *t_d, float *t, int L, float tau, float t0,
    float gridOS,
    float *outR_d, float *outI_d,
    float *&outR_gpu, float *&outI_gpu)
{
    float *tmpR_d, *tmpI_d;
    tmpR_d = mriNewGpu<float>(numX_per_coil);
    tmpI_d = mriNewGpu<float>(numX_per_coil);

    setZero_GlobalMemory(tmpR_d, numX_per_coil);
    setZero_GlobalMemory(tmpI_d, numX_per_coil);

    setZero_GlobalMemory(outR_d, numX_per_coil);
    setZero_GlobalMemory(outI_d, numX_per_coil);

    for (int c = 0; c < ncoils; c++) {
        setZero_GlobalMemory(tmpR_d, numX_per_coil);
        setZero_GlobalMemory(tmpI_d, numX_per_coil);

        computeAH_GPU_Grid(numK_per_coil, c, kx, ky, kz, dR, dI, Nx, Ny, Nz,
                           t, t_d, fm_d, L, tau, t0, gridOS, tmpR_d, tmpI_d);

        Hadamard_prod_Conj_GPU(tmpR_d, tmpI_d, sensi_r_d+c*numX_per_coil, 
                               sensi_i_d+c*numX_per_coil, tmpR_d, tmpI_d, 
                               numX_per_coil);

        add_GPU(outR_d, outI_d, outR_d, outI_d, tmpR_d, tmpI_d, numX_per_coil);
    }

    // copy final results to GPU.
    mriCopyDeviceToHost<float>(outR_gpu, outR_d, numX_per_coil);
    mriCopyDeviceToHost<float>(outI_gpu, outI_d, numX_per_coil);

    mriDeleteGpu<float>(tmpR_d);
    mriDeleteGpu<float>(tmpI_d);
}

// Brute Force way of computing AhD
    void 
computeAH_GPU_BF(
    int numK_per_coil, int numX_per_coil, int c,
    float *x_d, float *y_d, float *z_d,
    kValues_FH *kVals, kTrajectory *kTraj,
    float *t_d, float *fm_d,
    float *outR_d, float *outI_d)
{
    computeFH_GPU_BF(numK_per_coil, numX_per_coil, x_d, y_d, z_d, 
                     kVals+c*numK_per_coil, kTraj, t_d, fm_d, outR_d, outI_d);

}

// Brute Force way of computing par_AhD
    void 
parcomputeAH_GPU_BF(
    int numK_per_coil, int numX_per_coil, int ncoils,
    float *x_d, float *y_d, float *z_d,
    kValues_FH *kVals, kTrajectory *kTraj,
    float *sensi_r_d, float *sensi_i_d,
    float *fm_d, float *t_d,
    float *outR_d, float *outI_d,
    float *&outR_gpu, float *&outI_gpu)
{
    float *tmpR_d, *tmpI_d;
    tmpR_d = mriNewGpu<float>(numX_per_coil);
    tmpI_d = mriNewGpu<float>(numX_per_coil);

    setZero_GlobalMemory(tmpR_d, numX_per_coil);
    setZero_GlobalMemory(tmpI_d, numX_per_coil);

    setZero_GlobalMemory(outR_d, numX_per_coil);
    setZero_GlobalMemory(outI_d, numX_per_coil);

    for(int c = 0; c < ncoils; c++){
      setZero_GlobalMemory(tmpR_d, numX_per_coil);
      setZero_GlobalMemory(tmpI_d, numX_per_coil);

      computeAH_GPU_BF(numK_per_coil, numX_per_coil, c, x_d, y_d, z_d,
                       kVals, kTraj, t_d, fm_d, tmpR_d, tmpI_d);

      Hadamard_prod_Conj_GPU(tmpR_d, tmpI_d, sensi_r_d+c*numX_per_coil, 
                             sensi_i_d+c*numX_per_coil, tmpR_d, tmpI_d, 
                             numX_per_coil);

      add_GPU(outR_d, outI_d, outR_d, outI_d, tmpR_d, tmpI_d, numX_per_coil);
    }

    // copy final results to GPU.
    mriCopyDeviceToHost<float>(outR_gpu, outR_d, numX_per_coil);
    mriCopyDeviceToHost<float>(outI_gpu, outI_d, numX_per_coil);

    mriDeleteGpu<float>(tmpR_d);
    mriDeleteGpu<float>(tmpI_d);
}

    int
toeplitz_computeFH_GPU( const char *data_directory, const float ntime_segments, 
                        int Nx, int Ny, int Nz, const char *Fhd_full_filename, 
                        float *outR_gpu, float *outI_gpu,
                        const bool enable_direct, const bool enable_gridding,
                        float gridOS )
                       
{
    float time_segment_num = ntime_segments;
    float tau;
    float version = 0.2f;
    int numX, numK, numX_per_coil, numK_per_coil, ncoils, nslices;

    float *kx, *ky, *kz, *t;
    float *x, *y, *z, *phiR, *phiI, *fm;
    float *sensi_r, *sensi_i, *dR, *dI;
    float *x_d, *y_d, *z_d, *phiR_d, *phiI_d, *fm_d, *t_d;
    float *sensi_r_d, *sensi_i_d, *dR_d, *dI_d;
    float *realRhoPhi_d, *imagRhoPhi_d, *outI_d, *outR_d;
    float *tmp_r, *tmp_i, *expfm_r, *expfm_i;
    float *tmp_r_d, *tmp_i_d, *expfm_r_d, *expfm_i_d;
    float *realRhoPhi_gpu, *imagRhoPhi_gpu;

    #ifdef DEBUG
    unsigned int timerApp = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timerApp));
    CUT_SAFE_CALL(cutStartTimer(timerApp));
    #endif
    
    #ifdef DEBUG
      printf("======== Step 2. Compute F^H(d)  ========\n");
    #else
      printf("======== Step 2. Compute F^H(d)  ========\n");
    #endif
    /* Read in data */
    inputData_FH( data_directory, version, numK, numK_per_coil, ncoils, nslices, 
                  numX, numX_per_coil, kx, ky, kz, x, y, z, fm, t, phiR, phiI, 
                  dR, dI, sensi_r, sensi_i );

    tau = (time_segment_num > 0.0f) ? ((t[numK_per_coil - 1] - t[0]) / time_segment_num) 
          : (t[numK_per_coil - 1] - t[0]);
    float t0 = t[0];
  

    #ifdef DEBUG
    printf("numX_per_coil = %d,  numK_per_coil = %d, ncoils = %d\n", 
           numX_per_coil, numK_per_coil, ncoils);
    #else

    #endif

    unsigned int timerKernel = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timerKernel));
    CUT_SAFE_CALL(cutStartTimer(timerKernel));

    /* Create GPU data structures */
    createDataStructs( 
      (numK_per_coil * ncoils), numX_per_coil, realRhoPhi_gpu, imagRhoPhi_gpu );
    tmp_r   = (float *) calloc(numX_per_coil, sizeof(float));
    tmp_i   = (float *) calloc(numX_per_coil, sizeof(float));
    expfm_r = (float *) calloc(numX_per_coil, sizeof(float));
    expfm_i = (float *) calloc(numX_per_coil, sizeof(float));


    /* Setup GPU memory */
    phiR_d       = mriNewGpu<float>(numK_per_coil * ncoils);
    phiI_d       = mriNewGpu<float>(numK_per_coil * ncoils);
    sensi_r_d    = mriNewGpu<float>(numX_per_coil * ncoils);
    sensi_i_d    = mriNewGpu<float>(numX_per_coil * ncoils);
    x_d          = mriNewGpu<float>(numX_per_coil);
    y_d          = mriNewGpu<float>(numX_per_coil);
    z_d          = mriNewGpu<float>(numX_per_coil);
    fm_d         = mriNewGpu<float>(numX_per_coil);
    t_d          = mriNewGpu<float>(numK_per_coil);
    dR_d         = mriNewGpu<float>(numK_per_coil * ncoils);
    dI_d         = mriNewGpu<float>(numK_per_coil * ncoils);
    realRhoPhi_d = mriNewGpu<float>(numK_per_coil * ncoils);
    imagRhoPhi_d = mriNewGpu<float>(numK_per_coil * ncoils);
    tmp_r_d      = mriNewGpu<float>(numX_per_coil);
    tmp_i_d      = mriNewGpu<float>(numX_per_coil);
    expfm_r_d    = mriNewGpu<float>(numX_per_coil);
    expfm_i_d    = mriNewGpu<float>(numX_per_coil);
    // Zero out initial values of outR and outI.
    // GPU views these arrays as initialized (cleared) accumulators.
    outR_d       = mriNewGpu<float>(numX_per_coil);
    outI_d       = mriNewGpu<float>(numX_per_coil);

    mriCopyHostToDevice<float>(phiR_d, phiR, numK_per_coil * ncoils);
    mriCopyHostToDevice<float>(phiI_d, phiI, numK_per_coil * ncoils);
    mriCopyHostToDevice<float>(sensi_r_d, sensi_r, numX_per_coil * ncoils);
    mriCopyHostToDevice<float>(sensi_i_d, sensi_i, numX_per_coil * ncoils);
    mriCopyHostToDevice<float>(x_d, x, numX_per_coil);
    mriCopyHostToDevice<float>(y_d, y, numX_per_coil);
    mriCopyHostToDevice<float>(z_d, z, numX_per_coil);
    mriCopyHostToDevice<float>(fm_d, fm, numX_per_coil);
    mriCopyHostToDevice<float>(t_d, t, numK_per_coil);
    mriCopyHostToDevice<float>(dR_d, dR, numK_per_coil * ncoils);
    mriCopyHostToDevice<float>(dI_d, dI, numK_per_coil * ncoils);
    mriCopyHostToDevice<float>(realRhoPhi_d, realRhoPhi_gpu, numK_per_coil * ncoils);
    mriCopyHostToDevice<float>(imagRhoPhi_d, imagRhoPhi_gpu, numK_per_coil * ncoils);
    mriCopyHostToDevice<float>(tmp_r_d, tmp_r, numX_per_coil);
    mriCopyHostToDevice<float>(tmp_i_d, tmp_i, numX_per_coil);
    mriCopyHostToDevice<float>(expfm_r_d, expfm_r, numX_per_coil);
    mriCopyHostToDevice<float>(expfm_i_d, expfm_i, numX_per_coil);
    mriCopyHostToDevice<float>(outR_d, outR_gpu, numX_per_coil);
    mriCopyHostToDevice<float>(outI_d, outI_gpu, numX_per_coil);

    /* Pre-compute the values of rhoPhi on the GPU */
    computeRhoPhi_GPU((numK_per_coil * ncoils), phiR_d, phiI_d, dR_d, dI_d,
                      realRhoPhi_d, imagRhoPhi_d, realRhoPhi_gpu, imagRhoPhi_gpu);
    #ifdef DEBUG
    unsigned int timerSetupK = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timerSetupK));
    CUT_SAFE_CALL(cutStartTimer(timerSetupK));
    #endif

    kTrajectory *kTraj = (kTrajectory *) calloc(numK_per_coil, sizeof(kTrajectory));
    kValues_FH *kVals = (kValues_FH *) calloc((numK_per_coil * ncoils), sizeof(kValues_FH));
    float *kx_d, *ky_d, *kz_d;
    kx_d = mriNewGpu<float>(numK_per_coil);
    ky_d = mriNewGpu<float>(numK_per_coil);
    kz_d = mriNewGpu<float>(numK_per_coil);
    mriCopyHostToDevice<float>(kx_d, kx, numK_per_coil);
    mriCopyHostToDevice<float>(ky_d, ky, numK_per_coil);
    mriCopyHostToDevice<float>(kz_d, kz, numK_per_coil);

    MarshallScale_GPU( numK_per_coil, ncoils, kx_d, ky_d, kz_d, t_d, 
                       realRhoPhi_d, imagRhoPhi_d, kVals, kTraj);

    mriDeleteGpu<float>(kx_d);
    mriDeleteGpu<float>(ky_d);
    mriDeleteGpu<float>(kz_d);

    #ifdef DEBUG
    CUT_SAFE_CALL(cutStopTimer(timerSetupK));
    printf("Time to setup K: %f (s)\n", cutGetTimerValue(timerSetupK) / 1000.0);
    CUT_SAFE_CALL(cutDeleteTimer(timerSetupK));
    #endif

    /* Compute FH on the GPU */
    if(enable_direct) {
    parcomputeAH_GPU_BF(numK_per_coil, numX_per_coil, ncoils,
                        x_d, y_d, z_d, kVals, kTraj,
                        sensi_r_d, sensi_i_d,
                        fm_d, t_d,
                        outR_d, outI_d, outR_gpu, outI_gpu);
    } else {
    parcomputeAH_GPU_Grid(numK_per_coil, numX_per_coil, ncoils,
                          kx, ky, kz, dR, dI, Nx, Ny, Nz, sensi_r_d, 
                          sensi_i_d, fm_d, t_d, t, time_segment_num, tau, t0, gridOS,
                          outR_d, outI_d, outR_gpu, outI_gpu);
    }

    free(kVals);
    free(kTraj);

    /* Release memory on GPU */
    mriDeleteGpu<float>(x_d);
    mriDeleteGpu<float>(y_d);
    mriDeleteGpu<float>(z_d);
    mriDeleteGpu<float>(sensi_r_d);
    mriDeleteGpu<float>(sensi_i_d);
    mriDeleteGpu<float>(phiR_d);
    mriDeleteGpu<float>(phiI_d);
    mriDeleteGpu<float>(dR_d);
    mriDeleteGpu<float>(dI_d);
    mriDeleteGpu<float>(fm_d);
    mriDeleteGpu<float>( t_d);
    mriDeleteGpu<float>(tmp_r_d);
    mriDeleteGpu<float>(tmp_i_d);
    mriDeleteGpu<float>(expfm_r_d);
    mriDeleteGpu<float>(expfm_i_d);
    mriDeleteGpu<float>(outR_d);
    mriDeleteGpu<float>(outI_d);
    mriDeleteGpu<float>(realRhoPhi_d);
    mriDeleteGpu<float>(imagRhoPhi_d);
    

    CUT_SAFE_CALL(cutStopTimer(timerKernel));
    printf("  GPU: F^H(d) Kernel Time: %f (ms)\n", cutGetTimerValue(timerKernel));
    CUT_SAFE_CALL(cutDeleteTimer(timerKernel));
    #ifdef DEBUG
      printf("======== output ========\n");
    #else
      msg(MSG_PLAIN, "  GPU: Exporting results.\n");
    #endif
    /* Write result to file */
    outputData(Fhd_full_filename, outR_gpu, outI_gpu, numX_per_coil);
    

    free(kx);
    free(ky);
    free(kz);
    free(x);
    free(y);
    free(z);
    free(fm);
    free(t);
    free(sensi_r);
    free(sensi_i);
    free(tmp_r);
    free(tmp_i);
    free(expfm_r);
    free(expfm_i);
    free(phiR);
    free(phiI);
    free(dR);
    free(dI);
    free(realRhoPhi_gpu);
    free(imagRhoPhi_gpu);
    //free(outR_gpu);
    //free(outI_gpu);

    #ifdef DEBUG
    CUT_SAFE_CALL(cutStopTimer(timerApp));
    printf("GPU App Time: %f (s)\n", cutGetTimerValue(timerApp) / 1000.0);
    CUT_SAFE_CALL(cutDeleteTimer(timerApp));
    #else
    #endif

    return 0;
}
