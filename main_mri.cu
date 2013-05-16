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

    File Name   [main_mri.cu]

    Synopsis    [Main/Toppest function to launch the MRI program.]

    Description []

    Revision    [0.1; Initial build; Yue Zhuo, BIOE UIUC]
    Revision    [0.1.1; Code cleaning; Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Further optimization, Code cleaning, Adding more comments;
                 Xiao-Long Wu, ECE UIUC, Jiading Gai, Beckman Institute]
    Date        [10/27/2010]

*****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// CUDA libraries
#include <cutil.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <multithreading.h>

// Project header files
#include <xcpplib_process.h>
#include <xcpplib_typesGpu.cuh>
#include <tools.h>
#include <structures.h>
#include <bruteForceCpu.h>
#include <gpuPrototypes.cuh>
#include <computeQ.cmem.cuh>
#include <computeFH.cmem.cuh>
#include <recon.cuh>
#include <utils.h>
#include <main_mri.cuh>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Namespace used                                                           */
/*---------------------------------------------------------------------------*/

using namespace std;

/*---------------------------------------------------------------------------*/
/*  Function prototypes                                                      */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Program version and license information display.]           */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    void
mriSolverProgramHeader(FILE *fp)
{
    char version_info[75] = "\0";
    sprintf(version_info, "%s version %s, release date %s",
        mriSolver_name, mriSolver_version, mriSolver_release_date);

    fprintf(fp,
" +=========================================================================+\n"
" | %-72s|\n"
" | developed by:                                                           |\n"
" |                    IMPACT & MRFIL Research Groups                       |\n"
" |              University of Illinois at Urbana-Champaign                 |\n"
" |                                                                         |\n"
" | (C) Copyright 2010 The Board of Trustees of the University of Illinois. |\n"
" | All rights reserved.                                                    |\n"
" +=========================================================================+\n"
"\n", version_info);
}

    void
mriSolverVersion(FILE *fp)
{
    fprintf(fp, "%s version %s, release date %s.\n",
        mriSolver_name, mriSolver_version, mriSolver_release_date);
}

    void
mriSolverLicense(FILE *fp)
{
    fprintf(fp,
"                                                                            \n"
"                     Illinois Open Source License                           \n"
"                     University of Illinois/NCSA                            \n"
"                         Open Source License                                \n"
"                                                                            \n"
"(C) Copyright 2010 The Board of Trustees of the University of Illinois.     \n"
"All rights reserved.                                                        \n"
"                                                                            \n"
"Developed by:                                                               \n"
"                     IMPACT & MRFIL Research Groups                         \n"
"                University of Illinois, Urbana Champaign                    \n"
"                                                                            \n"
"Permission is hereby granted, free of charge, to any person obtaining a copy\n"
"of this software and associated documentation files (the \"Software\"), to  \n"
"deal with the Software without restriction, including without limitation the\n"
"rights to use, copy, modify, merge, publish, distribute, sublicense, and/or \n"
"sell copies of the Software, and to permit persons to whom the Software is  \n"
"furnished to do so, subject to the following conditions:                    \n"
"                                                                            \n"
"Redistributions of source code must retain the above copyright notice, this \n"
"list of conditions and the following disclaimers.                           \n"
"                                                                            \n"
"Redistributions in binary form must reproduce the above copyright notice,   \n"
"this list of conditions and the following disclaimers in the documentation  \n"
"and/or other materials provided with the distribution.                      \n"
"                                                                            \n"
"Neither the names of the IMPACT Research Group, MRFIL Research Group, the   \n"
"University of Illinois, nor the names of its contributors may be used to    \n"
"endorse or promote products derived from this Software without specific     \n"
"prior written permission.                                                   \n"
"                                                                            \n"
"THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS   \n"
"OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, \n"
"FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE \n"
"CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER \n"
"LIABILITY, WHETHER IN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  \n"
"OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH \n"
"THE SOFTWARE.                                                               \n"
"\n"
    );
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Toeplitz approach: Main function.]                          */
/*                                                                           */
/*  Description [This function contains both CPU and multiple GPU CG solvers.*/
/*      The GPU solver is first executed, followed by the CPU version. The   */
/*      GPU version will make use of multiple GPUs if available on the       */
/*      current machine.]                                                    */
/*                                                                           */
/*  Note        []                                                           */
/*                                                                           */
/*===========================================================================*/

// Kernel debugging macros
#if DEBUG_KERNEL_MSG
    #define DEBUG_TOEPLITZ  true
#else
    #define DEBUG_TOEPLITZ  false
#endif

    bool
toeplitz(
    const string &input_dir, const string &output_dir,
    const int cg_num, const bool enable_gpu, const bool enable_multi_gpu,
    const bool enable_cpu, const bool enable_regularization,
    const bool enable_finite_difference, const float fd_penalizer,
    const bool enable_total_variation, const int tv_num,
    const bool enable_tv_update, const bool enable_toeplitz_direct, 
    const bool enable_toeplitz_gridding, float gridOS_Q, float gridOS_FHD, 
    const float ntime_segments,    const int gpu_id,
    const bool enable_reuseQ, const string reuse_Qlocation,
    const bool enable_writeQ)
{

    // Jiading GAI
    int deviceID = gpu_id;
    int gpuCount = -1;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&gpuCount));
    if(deviceID>=0 && deviceID<gpuCount) {
      CUDA_SAFE_CALL(cudaSetDevice(deviceID));
    }
    else {
      printf("\nError: GPU ID is out of the range [0,%d].\n",gpuCount-1);
      exit(1);
    }
    cudaDeviceProp prop;
    CUDA_SAFE_CALL( cudaGetDeviceProperties(&prop,deviceID) );


    // =======================================================================
    //  Define variables
    // =======================================================================

    // Data dimensions
    FLOAT_T version;
    int num_slices;
    int num_i;                      // Number/Size of image data
    int num_k;                      // Number/Size of k-space data
    int num_coil;                   // Number of coils
    int num_gpus;                   // Number of GPU cards
    //int *num_DataGPU = NULL;        // ??
    //CUTThread *threadID = NULL;
    //CUTThread *thread_dummyID = NULL;

    DataTrajArray ktraj;            // K-space trajectory
    DataTrajArray itraj;            // Image space trajectory
    TArray<FLOAT_T> kdata_r;        // K-space data
    TArray<FLOAT_T> kdata_i;
    TArray<FLOAT_T> idata_r;        // Image space data (initial estimate)
    TArray<FLOAT_T> idata_i;
    TArray<FLOAT_T> sensi_r;
    TArray<FLOAT_T> sensi_i;
    TArray<FLOAT_T> fm;             // Field map
    TArray<FLOAT_T> time;           // Time vector

    CooMatrix c;                    // Sparse matrix for regularization constraint

    // =======================================================================
    //  Timers and FLOP counter
    // =======================================================================

    initMriTimer();

    // =======================================================================
    //  Display program information
    // =======================================================================

    mriSolverProgramHeader(stdout);

    // =======================================================================
    //  Load computation data from the input directory
    // =======================================================================

    // Read data from files
    msg(MSG_PLAIN, "  Reading data from files ... ");
    startMriTimer(getMriTimer()->timer_readFile);


       /* 3D - JGAI - BEGIN*/
    int Ny, Nx, Nz;

    loadInputData(input_dir, version,
                  num_slices, num_k, num_i, num_coil,
                  ktraj, itraj, kdata_r, kdata_i, idata_r, idata_i,
                  sensi_r, sensi_i, fm, time, c, enable_regularization,
                  Nx, Ny, Nz);

    if(version>=1.0f)
    {
      //Do nothing, b/c loadInputData already handle it.
    }
    else
    {
      Nz = 1;//version 0.2 does not support 3D data.
    }
    
    Ny = static_cast<int>( sqrt(num_i/Nz) );
    Nx = Ny;// Now image size in (row,col,depth) is (Ny,Nx,Nz)
       /*3D - JGAI - END*/
    stopMriTimer(getMriTimer()->timer_readFile);
    msg(MSG_PLAIN, "done.\n");


    msg(MSG_PLAIN, "  Which GPU to run on    : %s (ID=%d)\n", prop.name, deviceID);
    if(enable_toeplitz_gridding) {
       msg(MSG_PLAIN, "  Recon method           : Toeplitz with gridding for Q and FHD\n");
       msg(MSG_PLAIN, "  Gridding OS for Q      : %f\n", gridOS_Q);
       msg(MSG_PLAIN, "  Gridding OS for FHD    : %f\n", gridOS_FHD);
    }
    else {
       msg(MSG_PLAIN, "  Recon method           : Toeplitz with direct evaluation for Q and FHD\n");
    }
    if(enable_reuseQ) {
       msg(MSG_PLAIN, "  Reuse Q                : Yes\n");
    }
    else {
       msg(MSG_PLAIN, "  Reuse Q                : No\n");
    }
    if(enable_writeQ) {
       msg(MSG_PLAIN, "  Write Q to disk        : Yes\n");
    }
    else {
       msg(MSG_PLAIN, "  Write Q to disk        : No\n");
    }
    msg(MSG_PLAIN, "  Input folder           : %s\n", input_dir.c_str());
    msg(MSG_PLAIN, "  Output folder          : %s\n", input_dir.c_str());
    msg(MSG_PLAIN, "  Number of coils        : %d\n", num_coil);
    msg(MSG_PLAIN, "  Number of slices       : %d\n", num_slices);
    msg(MSG_PLAIN, "  Image data size (flat) : %d\n", num_i);
    msg(MSG_PLAIN, "  Image data size (yxz)  : (%d,%d,%d)\n", Ny,Nx,Nz);
    msg(MSG_PLAIN, "  K-space data size/coil : %d\n", num_k);
    msg(MSG_PLAIN, "  Number of CG iterations: %d\n", cg_num);
    msg(MSG_PLAIN, "  Number of time segments: %f\n", ntime_segments);
    if (enable_finite_difference) {
        msg(MSG_PLAIN, "  FD penalizer value     : %f\n", fd_penalizer);
    }
    if (enable_total_variation) {
        msg(MSG_PLAIN, "  Number of TV iterations: %d\n", tv_num);
        msg(MSG_PLAIN, "  TV smooth factor value : %f\n", MRI_SMOOTH_FACTOR);
    }

    // =======================================================================
    //  Prepare data for multi-GPU computation
    // =======================================================================

    // Get number of GPUs
    cutilSafeCall(cudaGetDeviceCount(&num_gpus));

    msg(MSG_PLAIN, "  Number of online GPUs  : %d\n", num_gpus);
    msg(MSG_PLAIN, "\n");

    // =======================================================================
    //  Toeplitz - no CPU code available for comparison
    // =======================================================================
    
    if (enable_toeplitz_direct||enable_toeplitz_gridding) {

       int phiR_datasize = num_k*num_coil; // num_k is k space data per coil.
        float *phiR = (float*) calloc( phiR_datasize, sizeof(float) );
        float *phiI = (float*) calloc( phiR_datasize, sizeof(float) );
        for(int i=0;i<phiR_datasize;i++)
        {
           phiR[i] = 1.0f;
           phiI[i] = 0.0f;
        }
        string phiR_filename = input_dir + "/phiR.dat";
        string phiI_filename = input_dir + "/phiI.dat";
        if(version==0.2f)
        {
           writeDataFile_JGAI(phiR_filename.c_str(), version, num_k, num_coil, 1, phiR_datasize, phiR);
           writeDataFile_JGAI(phiI_filename.c_str(), version, num_k, num_coil, 1, phiR_datasize, phiI);
        }
        else
        {
           writeDataFile_JGAI_10(phiR_filename.c_str(), version, Nx, Ny, Nz, num_coil, 1, phiR_datasize, phiR);
           writeDataFile_JGAI_10(phiI_filename.c_str(), version, Nx, Ny, Nz, num_coil, 1, phiR_datasize, phiI);
        }
        free(phiR);
        free(phiI);
        // */


        // Step 1. compute Q's
        int numX_per_coil_Q = 2*Nx * 2*Ny * (Nz==1?1:(2*Nz));
        float **Qr, **Qi;
        Qr = (float **) calloc(((int) ntime_segments) + 1, sizeof(float *));
        Qi = (float **) calloc(((int) ntime_segments) + 1, sizeof(float *));
        for (int l = 0; l <= ((int) ntime_segments); l++) {
            Qr[l] = (float *) calloc(numX_per_coil_Q, sizeof(float));
            Qi[l] = (float *) calloc(numX_per_coil_Q, sizeof(float));
        }


        if(enable_reuseQ) {

            printf("======== Step 1. Compute Q matrices: Q[%d] ========\n",static_cast<int>(ntime_segments+1));
            // reuse Q is enabled, so no need to compute Q for this slice.
            // However, we will do something sanity check below to make
            // sure the Q data structure exists and its file size is consistent
            // with the inputs:
            struct stat stat_Qfile;
            string Q_path_temp = reuse_Qlocation + "/Q_stone.file";
            printf("  Re-using the Q at %s\n", Q_path_temp.c_str());
            if( lstat(Q_path_temp.c_str(), &stat_Qfile) < 0 ) 
            {
                printf("Error: bad directory path that contains the Q file\n");
                printf("Please enter the directory path containing the Q file that you'd like to reuse\n");
                exit(1);
            }
            else
            {
                if(1==Nz) 
                {
                    // correct Q file size in bytes
                    long long correct_Q_size = ((long long) ntime_segments + 1 ) *
                                                ((long long) Nx) *
                                               ((long long) Ny) *
                                               ((long long) Nz) * 4 * 4 * 2;

                    if(correct_Q_size!=stat_Qfile.st_size)
                    {
                       printf("Error: incorrect Q file size\n");
                       exit(1);
                    }

                }
                else
                {
                    // correct Q file size in bytes
                    long long correct_Q_size = ((long long) ntime_segments + 1 ) *
                                              ((long long) Nx) *
                                              ((long long) Ny) *
                                              ((long long) Nz) * 8 * 4 * 2;            
    
                    if(correct_Q_size!=stat_Qfile.st_size)
                    {
                       printf("Error: incorrect Q file size\n");
                       exit(1);
                    }
            
                }
            
            }
        }    
        else {

        printf("======== Step 1. Compute Q matrices: Q[%d] ========\n",static_cast<int>(ntime_segments+1));
        //  Note that:
        //  Data prepartion : Toeplitz needs more data files than 
        //  Brute Force for the computation of Q. They are ixQ.dat, 
        //  iyQ.dat, izQ.dat and phiR.dat, phiI.dat. We are generating 
        //  them now.
        ///*
                // 3D - JGAI - BEGIN 
        float *ixQ=NULL, *iyQ=NULL, *izQ=NULL;
        if(Nz>1) // a 3D slice
        {
           ixQ = (float*) calloc( (2*Nx)*(2*Ny)*(2*Nz), sizeof(float) );
           iyQ = (float*) calloc( (2*Nx)*(2*Ny)*(2*Nz), sizeof(float) );
           izQ = (float*) calloc( (2*Nx)*(2*Ny)*(2*Nz), sizeof(float) );
        
           int index =0;
           for(int y=-Ny;y<Ny;y++)
           for(int x=-Nx;x<Nx;x++)
           for(int z=-Nz;z<Nz;z++)
           {
              iyQ[index] = ((float) y) / ((float) Ny);
              ixQ[index] = ((float) x) / ((float) Nx);
              izQ[index] = ((float) z) / ((float) Nz);
              index++;
           }
        }
        else if(Nz==1) // a 2D slice
        {
           ixQ = (float*) calloc( (2*Nx)*(2*Ny), sizeof(float) );
           iyQ = (float*) calloc( (2*Nx)*(2*Ny), sizeof(float) );
           izQ = (float*) calloc( (2*Nx)*(2*Ny), sizeof(float) );
              
           int index =0;
           for(int y=-Ny;y<Ny;y++)
           for(int x=-Nx;x<Nx;x++)
           {
              iyQ[index] = ((float) y) / ((float) Ny);
              ixQ[index] = ((float) x) / ((float) Nx);
              index++;
           }
        }
        else
        {
           fprintf(stderr,"*** Error at line %d of %s:\n",__LINE__,__FILE__);
           fprintf(stderr,"*** Error: Invalid Z-dimension size in Nz.dat.\n ");
           exit(1);
        }
                // 3D - JGAI - END


        string ixQ_filename = input_dir + "/ixQ.dat";
        string iyQ_filename = input_dir + "/iyQ.dat";
        string izQ_filename = input_dir + "/izQ.dat";

        if(version==0.2f)
        {
           int rows;
           if(1 == Nz) // 2D
           {
             rows = (2*Ny)*(2*Nx);
           }
           else if(1 < Nz) // 3D
           {
             rows = (2*Ny)*(2*Nx)*(2*Nz);
           }

           int ixQ_datasize = rows;
           writeDataFile_JGAI(ixQ_filename.c_str(), version, rows, num_coil, 1, ixQ_datasize, ixQ);
           writeDataFile_JGAI(iyQ_filename.c_str(), version, rows, num_coil, 1, ixQ_datasize, iyQ);
           writeDataFile_JGAI(izQ_filename.c_str(), version, rows, num_coil, 1, ixQ_datasize, izQ);
        }
        else
        {
           int Nx_Q, Ny_Q, Nz_Q;
           if(1==Nz)
           {
             Nz_Q = 1;
           }
           else
           {
             Nz_Q = 2 * Nz;
           }
           Nx_Q = 2 * Nx;
           Ny_Q = 2 * Ny;

           int ixQ_datasize = Nx_Q * Ny_Q * Nz_Q;
           // In ver 2.0 format, if coil number is irrelevant to a file, it must be 
           // set to -1 in writeDataFile_JGAI_10
           writeDataFile_JGAI_10(ixQ_filename.c_str(), version, Nx_Q, Ny_Q, Nz_Q, -1, 1, ixQ_datasize, ixQ);
           writeDataFile_JGAI_10(iyQ_filename.c_str(), version, Nx_Q, Ny_Q, Nz_Q, -1, 1, ixQ_datasize, iyQ);
           writeDataFile_JGAI_10(izQ_filename.c_str(), version, Nx_Q, Ny_Q, Nz_Q, -1, 1, ixQ_datasize, izQ);
        }
        free(ixQ);
        free(iyQ);
        free(izQ);

    
 
        string Q_filename = input_dir + "/Q_stone.file";
        toeplitz_computeQ_GPU ( input_dir.c_str(), ntime_segments, Nx, Ny, Nz, 
                                Q_filename.c_str(), Qr, Qi, enable_toeplitz_direct, 
                                enable_toeplitz_gridding, gridOS_Q, enable_writeQ ); 

        }



        // Step 2. compute Fhd
        int numX_per_coil = Nx * Ny * Nz;
        float *Fhd_outR = (float *) calloc(numX_per_coil, sizeof(float));
        float *Fhd_outI = (float *) calloc(numX_per_coil, sizeof(float));

        string Fhd_filename = input_dir + "/FhD.file";
        toeplitz_computeFH_GPU ( input_dir.c_str(), ntime_segments, 
                                 Nx, Ny, Nz, Fhd_filename.c_str(), 
                                 Fhd_outR, Fhd_outI, enable_toeplitz_direct, 
                                 enable_toeplitz_gridding, gridOS_FHD ); 
        
        // Step 3. recon
        int toe_argc = 18;
        int toe_str_sz = 9999;//FIXME:Get rid of the 9999 upper limit
        char **toe_argv = (char **) malloc( toe_argc*sizeof(char *) );
        for(int i=0;i<toe_argc;i++)
            toe_argv[i] = (char *) malloc( toe_str_sz*sizeof(char) );
 
        sprintf(toe_argv[0],"toeplitz_recon");
        int N1 = Ny; sprintf(toe_argv[1], "%d", N1);
        int N2 = Nx; sprintf(toe_argv[2], "%d", N2);
        int N3 = Nz; sprintf(toe_argv[3], "%d", N3);
        int numRestarts = 1; sprintf(toe_argv[4], "%d", numRestarts);
        int numIterMax = cg_num; sprintf(toe_argv[5], "%d", numIterMax);
        int symTraj = 0; sprintf(toe_argv[6], "%d", symTraj);
        float lambda2 = fd_penalizer; sprintf(toe_argv[7], "%f",lambda2);
        sprintf(toe_argv[8],"D.file"); sprintf(toe_argv[9],"Dp.file");
        sprintf(toe_argv[10],"R.file"); sprintf(toe_argv[11],"W.file");
        /*
         * If the -reuseQ flag is turned, toe_argv[12] stores the location
         * where the to-be-reused Q is saved at.
         */
        sprintf(toe_argv[12],"%s/Q_stone.file",reuse_Qlocation.c_str());

        sprintf(toe_argv[13],"%s/FhD.file",input_dir.c_str());
        sprintf(toe_argv[14],"%s",input_dir.c_str());
        sprintf(toe_argv[15],"%d",tv_num);
        sprintf(toe_argv[16],"%f",ntime_segments);
        sprintf(toe_argv[17],"%s/out.file",input_dir.c_str());

        #ifdef DEBUG
        for(int i=0;i<toe_argc;i++)
        {
            printf("argv[%d] = %s\n", i, toe_argv[i]);
        }
        #endif

        toeplitz_recon(toe_argc, toe_argv, enable_total_variation, 
                       enable_regularization, enable_finite_difference, enable_reuseQ, 
                       &c, Fhd_outR, Fhd_outI, Qr, Qi);
 
        for(int i=0;i<toe_argc;i++)
            free( toe_argv[i] );
        free( toe_argv );
     
        free(Fhd_outR);
        free(Fhd_outI);

        for (int l = 0; l <= ((int) ntime_segments); l++) {
            free(Qr[l]);
            free(Qi[l]);
        }
        free(Qr);
        free(Qi);
    }

    //FIXME: deleteMriTimer() is needed, but this version gives segment fault.
    //deleteMriTimer();
    cudaThreadExit();
    return true;
} // toeplitz

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Brute force approach: Main function.]                       */
/*                                                                           */
/*  Description [This function contains both CPU and multiple GPU CG solvers.*/
/*      The GPU solver is first executed, followed by the CPU version. The   */
/*      GPU version will make use of multiple GPUs if available on the       */
/*      current machine.]                                                    */
/*                                                                           */
/*  Note        []                                                           */
/*                                                                           */
/*===========================================================================*/

// Kernel debugging macros
#if DEBUG_KERNEL_MSG
    #define DEBUG_BRUTEFORCE  true
#else
    #define DEBUG_BRUTEFORCE  false
#endif

    bool
bruteForce(
    const string &input_dir, const string &output_dir,
    const int cg_num, const bool enable_gpu, const bool enable_multi_gpu,
    const bool enable_cpu, const bool enable_regularization,
    const bool enable_finite_difference, const float fd_penalizer,
    const bool enable_total_variation, const int tv_num,
    const bool enable_tv_update,
    const int gpu_id)
{
    // =======================================================================
    //  Define variables
    // =======================================================================

    // Data dimensions
    FLOAT_T version;                  
    int num_slices;
    int num_i;                      // Number/Size of image data
    int num_k;                      // Number/Size of k-space data
    int num_coil;                   // Number of coils
    int num_gpus;                   // Number of GPU cards
    int *num_DataGPU = NULL;        // ??
    CUTThread *threadID = NULL;
    CUTThread *thread_dummyID = NULL;

    DataTrajArray ktraj;            // K-space trajectory
    DataTrajArray itraj;            // Image space trajectory
    TArray<FLOAT_T> kdata_r;        // K-space data
    TArray<FLOAT_T> kdata_i;
    TArray<FLOAT_T> idata_r;        // Image space data (initial estimate)
    TArray<FLOAT_T> idata_i;
    TArray<FLOAT_T> sensi_r;          
    TArray<FLOAT_T> sensi_i;          
    TArray<FLOAT_T> fm;             // Field map
    TArray<FLOAT_T> time;           // Time vector

    CooMatrix c;                    // Sparse matrix for regularization constraint

    // Output filenames
    string output_file_gpu_r;
    string output_file_gpu_i;
    string output_file_cpu_r;
    string output_file_cpu_i;

    // =======================================================================
    //  Timers and FLOP counter
    // =======================================================================

    initMriTimer();

    // =======================================================================
    //  Display program information
    // =======================================================================

    mriSolverProgramHeader(stdout);

    // =======================================================================
    //  Load computation data from the input directory
    // =======================================================================

    // Read data from files
    msg(MSG_PLAIN, "  Reading data from files ... ");
    startMriTimer(getMriTimer()->timer_readFile);
    int Ny, Nx, Nz;

    loadInputData(input_dir, version,
                  num_slices, num_k, num_i, num_coil,
                  ktraj, itraj, kdata_r, kdata_i, idata_r, idata_i,
                  sensi_r, sensi_i, fm, time, c, enable_regularization,
                  Nx, Ny, Nz);

    if(version>=1.0f)
    {
      //Do nothing, b/c loadInputData already handle it.
    }
    else
    {
      Nz = 1;//version 0.2 does not support 3D data.
    }
    Ny = static_cast<int>( sqrt(num_i/Nz) );
    Nx = Ny;// Now image size is (row,col,depth) is (Ny,Nx,Nz)

    output_file_gpu_r = output_dir + "/output_gpu_r.dat";
    output_file_gpu_i = output_dir + "/output_gpu_i.dat";

    stopMriTimer(getMriTimer()->timer_readFile);
    msg(MSG_PLAIN, "done.\n");

    // Image space data (for cpu computation)
    const TArray<FLOAT_T> idata_cpu_r = idata_r;
    const TArray<FLOAT_T> idata_cpu_i = idata_i;

    msg(MSG_PLAIN, "  Recon method           : Brute Force\n");
    msg(MSG_PLAIN, "  Input folder           : %s\n", input_dir.c_str());
    msg(MSG_PLAIN, "  Output folder          : %s\n", output_dir.c_str());
    msg(MSG_PLAIN, "  Number of coils        : %d\n", num_coil);
    msg(MSG_PLAIN, "  Number of slices       : %d\n", num_slices);
    msg(MSG_PLAIN, "  Image data size        : %d\n", num_i);
    msg(MSG_PLAIN, "  Image data size (yxz)  : (%d,%d,%d)\n", Ny,Nx,Nz);
    msg(MSG_PLAIN, "  K-space data size      : %d\n", num_k);
    msg(MSG_PLAIN, "  Number of CG iterations: %d\n", cg_num);
    if (enable_finite_difference) {
        msg(MSG_PLAIN, "  FD penalizer value     : %f\n", fd_penalizer);
    }
    if (enable_total_variation) {
        msg(MSG_PLAIN, "  Number of TV iterations: %d\n", tv_num);
        msg(MSG_PLAIN, "  TV smooth factor value : %f\n", MRI_SMOOTH_FACTOR);
    }
    // =======================================================================
    //  Prepare data for multi-GPU computation
    // =======================================================================

    // Get number of GPUs
    cutilSafeCall(cudaGetDeviceCount(&num_gpus));

    msg(MSG_PLAIN, "  Number of online GPUs  : %d\n", num_gpus);
    msg(MSG_PLAIN, "\n");

    CgSolverData **multislice_data = NULL;
    if (enable_multi_gpu && num_gpus > 1) {
        //Jiading GAI
        //multislice_data = mriNewCpu<CgSolverData*>(num_gpus);
        multislice_data = new CgSolverData*[num_gpus];


        // Calculate number of slices per GPU
        num_DataGPU = mriNewCpu<int>(num_gpus);
        for (int i = 0; i < num_gpus; i++) {
            num_DataGPU[i] = num_slices / num_gpus;
        }
        for (int i = 0; i < num_slices % num_gpus; i++) {
            num_DataGPU[i]++;
        }

        // FIXME: Bugs in this part.
        // Allocate memory for each data structure
        for (int i = 0; i < num_gpus; i++) {
            // Jiading GAI
            //multislice_data[i] = mriNewCpu<CgSolverData>(num_DataGPU[i]);
            multislice_data[i] = new CgSolverData[num_DataGPU[i]];
        }
        threadID = mriNewCpu<CUTThread>(num_gpus);

        int s = 0;
        for (int i = 0; i < num_gpus; i++) {
            for (int j = 0; j < num_DataGPU[i]; j++) {
                if (s < num_slices) {
                    int stepX = s * num_i;
                    int stepK = s * num_k;
                    multislice_data[i][j].num_gpus = num_gpus;
                    multislice_data[i][j].num_slices_total = num_slices;
                    multislice_data[i][j].num_slices_GPU = num_DataGPU[i];
                    multislice_data[i][j].gpu_id = i;
                    multislice_data[i][j].sliceGPU_Idx = j;
                    multislice_data[i][j].slice_id = s;
                    multislice_data[i][j].idata_r = idata_r.array + stepX;
                    multislice_data[i][j].idata_i = idata_i.array + stepX;
                    multislice_data[i][j].kdata_r = kdata_r.array + stepK;
                    multislice_data[i][j].kdata_i = kdata_i.array + stepK;
                    multislice_data[i][j].ktraj = ktraj.array;
                    multislice_data[i][j].itraj = itraj.array;
                    multislice_data[i][j].sensi_r = sensi_r.array;
                    multislice_data[i][j].sensi_i = sensi_i.array;
                    multislice_data[i][j].num_coil = num_coil;
                    multislice_data[i][j].fm = fm.array + stepX;
                    multislice_data[i][j].time = time.array;
                    multislice_data[i][j].c = &c;
                    multislice_data[i][j].cg_num = cg_num;
                    multislice_data[i][j].num_k = num_k;
                    multislice_data[i][j].num_i = num_i;
                    multislice_data[i][j].enable_regularization =
                                          enable_regularization;
                    multislice_data[i][j].enable_finite_difference =
                                          enable_finite_difference;
                    multislice_data[i][j].fd_penalizer = fd_penalizer;
                    multislice_data[i][j].enable_tv_update = enable_tv_update;
                    multislice_data[i][j].output_file_gpu_r = output_file_gpu_r;
                    multislice_data[i][j].output_file_gpu_i = output_file_gpu_i;
                    s++;
                }
            }
        }
    }

    // =======================================================================
    //  Dummy GPU calls to warm up the GPU devices.
    //  The calculated data are not used so the time is not counted.
    // =======================================================================

    if (enable_multi_gpu && num_gpus > 1) {
        thread_dummyID = mriNewCpu<CUTThread>(num_gpus);
        for (int i = 0; i < num_gpus; i++) {
            thread_dummyID[i] = cutStartThread((CUT_THREADROUTINE) mgpuDummy,
                                               (void *)i);
        }
        cutWaitForThreads(thread_dummyID, num_gpus);

    } else {
        FLOAT_T *dummy_A = mriNewGpu<FLOAT_T>(10);
        dim3 dummy_block(10);
        dim3 dummy_grid(1);
        addGpuKernel <<< dummy_grid, dummy_block >>>
                        (dummy_A, dummy_A, dummy_A, dummy_A, dummy_A, dummy_A,
                         1, 10);
        mriDeleteGpu(dummy_A);
    }

    // =======================================================================
    //  GPU-version conjugate gradient solver
    // =======================================================================

    // Check output file permission before execution.

    if (enable_gpu) {
        checkPermission(output_file_gpu_r.c_str(), "w", !DEBUG_MODE);
        checkPermission(output_file_gpu_i.c_str(), "w", !DEBUG_MODE);

        #if ENABLE_DOUBLE_PRECISION
        msg(1, "GPU: Enable double precision floating-point computation.");
        #endif
        startMriTimer(getMriTimer()->timer_Gpu);
        // Single GPU to handle all slices or when only one slice
        if (!enable_multi_gpu || num_slices == 1) {
            for (int slice = 0; slice < num_slices; slice++) {
                msg(1, "GPU: Processing slice %d.", slice);
                int slice_stepX = slice * num_i;
                int slice_stepK = slice * num_k;
                FLOAT_T *cur_idata_r = idata_r.array + slice_stepX;
                FLOAT_T *cur_idata_i = idata_i.array + slice_stepX;

                bruteForceGpu(
                    &cur_idata_r, &cur_idata_i,
                    kdata_r.array + slice_stepK,
                    kdata_i.array + slice_stepK,
                    ktraj.array, itraj.array,
                    fm.array + slice_stepX, time.array,
                    &c, cg_num, num_k, num_i,
                    enable_regularization,
                    enable_finite_difference, fd_penalizer,
                    enable_total_variation, tv_num,
                    sensi_r.array, sensi_i.array, num_coil,
                    enable_tv_update,
                    output_file_gpu_r, output_file_gpu_i, num_slices);
            }
    
        } else { // Use multiple GPU if slice number is greater than 1.
            msg(1, "GPU: Using multiple GPUs for multiple slices.");
            for (int i = 0; i < num_gpus && i < num_slices; i++) {
                threadID[i] = cutStartThread((CUT_THREADROUTINE) bruteForceMgpu,
                                             (void *) multislice_data[i]);
            }
            cutWaitForThreads(threadID, num_gpus);
        }
        stopMriTimer(getMriTimer()->timer_Gpu);
    }

    // =======================================================================
    //  Export GPU results
    // =======================================================================

    if (enable_gpu) {
        msg(1, "GPU: Execution time: %f (ms)",
            cutGetTimerValue(getMriTimer()->timer_bruteForceGpu));
        msg(1, "GPU: Exporting results.\n");
    
        startMriTimer(getMriTimer()->timer_writeFile);
        exportDataCpu(output_file_gpu_r, idata_r, num_slices * num_i);
        exportDataCpu(output_file_gpu_i, idata_i, num_slices * num_i);
        stopMriTimer(getMriTimer()->timer_writeFile);
    }

    // =======================================================================
    //  CPU-version conjugate gradient solver
    // =======================================================================

    // Check output file permission before execution.
    output_file_cpu_r = output_dir + "/output_cpu_r.dat";
    output_file_cpu_i = output_dir + "/output_cpu_i.dat";

    if (enable_cpu) {
        checkPermission(output_file_cpu_r.c_str(), "w", !DEBUG_MODE);
        checkPermission(output_file_cpu_i.c_str(), "w", !DEBUG_MODE);

        #if USE_OPENMP
        msg(1, "CPU: Using OpenMP to speed up the computation with %d threads.",
            omp_get_max_threads());
        #endif
        startMriTimer(getMriTimer()->timer_Cpu);
        for (int slice = 0; slice < num_slices; slice++) {
            msg(1, "CPU: Processing slice %d.", slice);
            int slice_stepX = slice * num_i;
            int slice_stepK = slice * num_k;
            FLOAT_T *cur_idata_cpu_r = idata_cpu_r.array + slice_stepX;
            FLOAT_T *cur_idata_cpu_i = idata_cpu_i.array + slice_stepX;

            bruteForceCpu(
                &cur_idata_cpu_r, &cur_idata_cpu_i,
                kdata_r.array + slice_stepK,
                kdata_i.array + slice_stepK,
                ktraj.array, itraj.array,
                fm.array + slice_stepX, time.array,
                &c, cg_num, num_k, num_i,
                enable_regularization,
                enable_finite_difference, fd_penalizer,
                enable_total_variation, tv_num,
                sensi_r.array, sensi_i.array, num_coil);
        }
        stopMriTimer(getMriTimer()->timer_Cpu);
    }

    // =======================================================================
    //  Export CPU results
    // =======================================================================

    // CPU results
    if (enable_cpu) {
        msg(1, "CPU: Execution time: %f (ms)",
            cutGetTimerValue(getMriTimer()->timer_bruteForceCpu));
        msg(1, "CPU: Exporting results.");

        startMriTimer(getMriTimer()->timer_writeFile);
        exportDataCpu(output_file_cpu_r, idata_cpu_r, num_slices * num_i);
        exportDataCpu(output_file_cpu_i, idata_cpu_i, num_slices * num_i);
        stopMriTimer(getMriTimer()->timer_writeFile);
    }

    // =======================================================================
    //  Compare results
    // =======================================================================

    if (enable_cpu) {
        msg(MSG_PLAIN, "\n  ---------------------------- Results ----------------------------\n\n");
        double error = getNRMSE(idata_r.array, idata_i.array,
                                 idata_cpu_r.array, idata_cpu_i.array,
                                 num_slices * num_i);
        if (error < 0.01f) {
            msg(2, "Test PASSED  (error = %0.8f (%e))", error, error);
        } else {
            msg(2, "Test FAILED  (error = %0.8f (%e))", error, error);
        }
    }

    // =======================================================================
    //  Timer information
    // =======================================================================

    msg(MSG_PLAIN, "\n  -------------------------- Performance --------------------------\n");

    printMriTimer();

    // =======================================================================
    //  Free memory
    // =======================================================================

    // MultiGPU data
    if (enable_multi_gpu && num_gpus > 1) {
        for (int i = 0; i < num_gpus; i++) {
            //Jiading GAI
            delete [] multislice_data[i];
            //mriDeleteCpu(multislice_data[i]);
        }

        //Jiading GAI
        //mriDeleteCpu(multislice_data);
        delete [] multislice_data;

        mriDeleteCpu(num_DataGPU);
        mriDeleteCpu(threadID);
        mriDeleteCpu(thread_dummyID);
    }

    cudaThreadExit();
    return true;
} // bruteForce

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}
