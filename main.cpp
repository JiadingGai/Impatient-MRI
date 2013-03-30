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

    File Name   [main.cpp]

    Synopsis    [Main/Toppest function to launch the program.]

    Description []

    Revision    [0.1; Initial build; Yue Zhuo, BIOE UIUC]
    Revision    [0.1.1; Code cleaning; Xiao-Long Wu, ECE UIUC]
    Date        [03/23/2010]

*****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Project header files
#include "main_mri.cuh"

/*---------------------------------------------------------------------------*/
/*  Namespace used                                                           */
/*---------------------------------------------------------------------------*/

using namespace std;

// For some reason, namespace doesn't work so we disabled all uiuc_mri
// namespaces in source files.
//using namespace uiuc_mri;

/*---------------------------------------------------------------------------*/
/*  Function prototypes                                                      */
/*---------------------------------------------------------------------------*/

    void
programUsage(FILE *fp);

/*---------------------------------------------------------------------------*/
/*  Function definitions                                                     */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Main function of the whole application.]                    */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*  Note        []                                                           */
/*                                                                           */
/*===========================================================================*/

#define CG_NUM_DEFAULT  8
#define TV_NUM_DEFAULT  10
#define FDP_DEFAULT     0.00000 // FDP_DEFALT feature disabled permanently
#define GRIDOS_Q        1.125f
#define GRIDOS_FHD      1.5f
#define NTIME_SEG_DEFAULT 12.0 // has to be a natural number in floating point format

    int
main (int argc, char **argv)
{
    const string prog = mriSolver_name;

    string idir;                          // input directory
    bool ifndef_idir = true;
    string odir;                          // output directory
    bool ifndef_odir = true;

    int   option_cg_num = CG_NUM_DEFAULT; // Set CG iteration number
    bool  option_nogpu = false;           // Disable GPU computation
    bool  option_mgpu = false;            // Enable Multi-GPU computation
    bool  option_cpu = false;             // Enable CPU computation
    bool  option_reg = false;             // Enable regularization
    bool  option_fd = false;              // Enable finite difference
    bool  option_tv = false;              // Enable total variation
    int   option_tv_num = TV_NUM_DEFAULT; // Set TV iteration number
    bool  option_tv_update = false;       // Update output for each TV iteration
    float option_fdp = FDP_DEFAULT;       // Set finite difference penalizer
    bool  option_toeplitz_gridding = false;        // Enable toeplitz code (gridding) - no CPU version
    bool  option_toeplitz_direct = false;        // Enable toeplitz code (gridding) - no CPU version
    float option_ntime_segments = NTIME_SEG_DEFAULT; // Set number of time segments (TOEPLITZ)
    float option_gridOS_Q   = GRIDOS_Q;     // Set gridding oversampling factor (for Q) to default
    float option_gridOS_FHD = GRIDOS_FHD;   // Set gridding oversampling factor (for FHD) to default
	int   option_gpu_id = 0;//Set GPU device ID, default ID=0
	bool  option_reuseQ = false;// set to true to enable the reuse of Q
	string option_reuse_Qlocation;// It contains the location Q to be re-used if option_reuseQ=true
	bool option_writeQ = false;


    bool  result = false;

    /*
     * Getting the file name for log name.
     */

    if (argc > 1) {
        cout<< "mriSolver: Command options: ";
        for (int i = 1; i < argc; i++) {
            cout<< argv[i]<< " ";
        }
        cout<< endl;
    }

    for (int i = 1; i < argc; i++) {
        if (!strcmp("-idir", argv[i])) {
            i++;
            ifndef_idir = false;
            if (argv[i] && argv[i][0] != '-') {
                idir = argv[i];
            } else {
                cerr<< "\nError: Missing input directory name.\n";
                goto _ERROR_mri_main;
            }

        } else if (!strcmp("-gpu_id", argv[i])) {
			i++;
			if (argv[i] && argv[i][0] != '-') {
               option_gpu_id = atoi(argv[i]);
			   if (option_gpu_id < 0) {
				   cerr<< "\nError: invalid GPU device ID.\n";
				   goto _ERROR_mri_main;
			   }
		    } else {
				cerr<< "\nError: Missing gpu ID number.\n";
                goto _ERROR_mri_main;
			}

        } else if (!strcmp("-cg_num", argv[i])) {
            i++;
            if (argv[i] && argv[i][0] != '-') {
                option_cg_num = atoi(argv[i]);
                if (option_cg_num == 0) {
                    cerr<< "\nError: Zero or wrong CG iteration number format.\n";
                    goto _ERROR_mri_main;
                }
            } else {
                cerr<< "\nError: Missing CG iteration number.\n";
                goto _ERROR_mri_main;
            }

        } else if (!strcmp("-odir", argv[i])) {
            i++;
            ifndef_odir = false;
            if (argv[i] && argv[i][0] != '-') {
                odir = argv[i];
            } else {
                cerr<< "\nError: Missing output directory name.\n";
                goto _ERROR_mri_main;
            }

        } else if (!strcmp("-nogpu", argv[i])) {
            option_nogpu = true;

        } else if (!strcmp("-mgpu", argv[i])) {
            option_mgpu = true;

        } else if (!strcmp("-cpu", argv[i])) {
            option_cpu = true;

        } else if (!strcmp("-reg", argv[i])) {
            option_reg = true;

        } else if (!strcmp("-fd", argv[i])) {
            option_fd = true;

        } else if (!strcmp("-toeplitzGridding", argv[i])) {
			option_toeplitz_gridding = true;

			option_nogpu = false;           // Disable GPU computation
            option_mgpu = false;            // Disable Multi-GPU computation
            option_cpu = false;             // Disable CPU computation
            option_toeplitz_direct = false; // Disable toeplitz computation with direct evaluation

		} else if (!strcmp("-toeplitzDirect", argv[i])) {
			option_toeplitz_direct = true;

			option_nogpu = false;           // Disable GPU computation
            option_mgpu = false;            // Disable Multi-GPU computation
            option_cpu = false;             // Disable CPU computation
            option_toeplitz_gridding = false; // Disable toeplitz computation with direct evaluation

		} else if(!strcmp("-reuseQ",argv[i])) {
			i++;
			if( (option_toeplitz_direct || option_toeplitz_gridding) &&
				argv[i] && argv[i][0] != '-') {
				option_reuseQ = true;
                option_reuse_Qlocation = argv[i];
			}
			else {
				cerr<<"\nError: -reuseQ flag requires toeplitz strategy to turn on.\n";
				cerr<<"\nAlso, follwing -reuseQ, please specify the directory where Q resides.\n";
				goto _ERROR_mri_main;
			}
		
	    } else if (!strcmp("-writeQ",argv[i])) {

			option_writeQ = true;
		
		}
		else if (!strcmp("-ntime_segs", argv[i])) { 
			   i++;
			   if( ((option_toeplitz_direct==true) || (option_toeplitz_gridding==true)) && 
                   argv[i] && argv[i][0] != '-' ) 
			   {
				   option_ntime_segments = atof(argv[i]);
				   if(option_ntime_segments<1.0) {
					   cerr<<"\nError: number of time segments has to be a natural number in floating format.\n";
					   goto _ERROR_mri_main;
				   }
			   }
			   else {
			     cerr<< "\nError:-toeplitz flag is not found or invalid number of time segments.\n";
				 goto _ERROR_mri_main;
			   }

		} else if (!strcmp("-gridOS_Q", argv[i])) { 
			   i++;
			   if( ((option_toeplitz_gridding==true)) && 
                   argv[i] && argv[i][0] != '-' ) 
			   {
				   option_gridOS_Q = atof(argv[i]);
				   if(option_gridOS_Q<1.0 || option_gridOS_Q>2.0) {
					   cerr<<"\nError: gridding oversampling factor should be in [1.0,2.0].\n";
					   goto _ERROR_mri_main;
				   }
			   }
			   else {
			     cerr<< "\nError:-toeplitzGridding flag is not found or invalid gridding OS factor (Q).\n";
				 goto _ERROR_mri_main;
			   }

		} else if (!strcmp("-gridOS_FH", argv[i])) { 
			   i++;
			   if( ((option_toeplitz_gridding==true)) && 
                   argv[i] && argv[i][0] != '-' ) 
			   {
				   option_gridOS_FHD = atof(argv[i]);
				   if(option_gridOS_FHD<1.0 || option_gridOS_FHD>2.0) {
					   cerr<<"\nError: gridding oversampling factor should be in [1.0,2.0].\n";
					   goto _ERROR_mri_main;
				   }
			   }
			   else {
			     cerr<< "\nError:-toeplitzGridding flag is not found or invalid gridding OS factor (FHD).\n";
				 goto _ERROR_mri_main;
			   }

		}  else if (!strcmp("-fdp", argv[i])) {
            i++;
            if (argv[i] && argv[i][0] != '-') {
                // FD is enabled automatically when setting the FDP number.
                option_fd = true;
                option_fdp = atof(argv[i]);
            } else {
                cerr<< "\nError: Missing finite difference penalty.\n";
                goto _ERROR_mri_main;
            }

        } else if (!strcmp("-tv", argv[i])) {
            option_tv = true;

        } else if (!strcmp("-tv_num", argv[i])) {
            i++;
            if (argv[i] && argv[i][0] != '-') {
                // TV is enabled automatically when setting the TV number.
                option_tv = true;
                option_tv_num = atoi(argv[i]);
                if (option_tv_num == 0) {
                    cerr<< "\nError: Zero or wrong TV iteration number format.\n";
                    goto _ERROR_mri_main;
                }
            } else {
                cerr<< "\nError: Missing TV iteration number.\n";
                goto _ERROR_mri_main;
            }
        } else if (!strcmp("-tv_update", argv[i])) {
            option_tv = true;
            option_tv_update = true;

        } else if (!strcmp("-version", argv[i])) {
            mriSolverVersion(stdout);
            mriSolverLicense(stdout);
            goto _ERROR_mri_main;

        } else if (!strcmp("-help", argv[i])) {
            programUsage(stderr);
            goto _ERROR_mri_main;

        } else {
            cerr<< "\nError: Unknown option \""<< argv[i]<<"\""<< endl;
            programUsage(stderr);
            goto _ERROR_mri_main;
        }
    }

    /*
     * Checking program arguments
     */

    if (argc == 1) { // No any arguments are given.
        programUsage(stderr);
        goto _ERROR_mri_main;
    }

    if (ifndef_idir) {
        cerr<< "\nError: Missing input directory name.\n";
        goto _ERROR_mri_main;
    }

    if (ifndef_odir) {
        odir = idir + "/output";
        cerr<< prog<< ": Using default output directory name \""
            << odir<< "\"."<< endl;
        // FIXME: Should make a new directory if the directory is not there.
    }

    if (option_nogpu) {
        cout<< prog<< ": Disable the GPU part computation."<< endl;
        option_cpu = true; // This option enables the CPU part computation.
    }

    if (option_mgpu) {
        cout<< prog<< ": Using multiple GPUs when ever possible."<< endl;
    }

    if (option_cpu) {
        cout<< prog<< ": Enable the CPU part computation."<< endl;
    }

    if (option_reg && option_fd) {
        cout<< prog<< ": Error: Roughness regularization and finite difference are exclusive."<< endl;
    }

    if (option_reg) {
        cout<< prog<< ": Enable roughness regularization."<< endl;
    } else if (option_fd) {
        cout<< prog<< ": Enable finite difference."<< endl;
    }

    if (option_tv) {
        cout<< prog<< ": Enable total variation."<< endl;
    }

    if (option_tv_update) {
        cout<< prog<< ": Update output for each TV iteration."<< endl;
    }

    if (option_fd || option_tv)
	{
       bool fdp_flag = false;
       for (int i=1;i<argc;i++)
       {
          if ( !strcmp("-fdp",argv[i]) )
			  fdp_flag = true;
       }
       if (!fdp_flag)
       {
	      cerr<<"\n Error: Regularization enabled, but no weight parameter specified. Hint: -fdp [lambda].\n";
          goto _ERROR_mri_main;
       }
	}

    cout<< endl;

    /*====================================================================*/
    /* Applying the MRI-Solver routines.                                  */
    /*====================================================================*/

    if (option_toeplitz_gridding||option_toeplitz_direct) {
        result = toeplitz(idir, odir, option_cg_num, !option_nogpu,
                          option_mgpu, option_cpu, option_reg,
                          option_fd, option_fdp,
                          option_tv, option_tv_num, option_tv_update,
                          option_toeplitz_direct, option_toeplitz_gridding,
						  option_gridOS_Q, option_gridOS_FHD,
						  option_ntime_segments, option_gpu_id,
						  option_reuseQ, option_reuse_Qlocation,
						  option_writeQ);
    } else {
        result = bruteForce(idir, odir, option_cg_num, !option_nogpu,
                           option_mgpu, option_cpu, option_reg,
                           option_fd, option_fdp,
                           option_tv, option_tv_num, option_tv_update,
						   option_gpu_id);
    }

    if (result == false) {
        cerr<< "Error: Failed to execute the "<< prog<< ".\n";
        goto _ERROR_mri_main;
    }

    /*====================================================================*/
    /* Ending.                                                            */
    /*====================================================================*/

    return 0;

    _ERROR_mri_main:
        return 1;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Displaying program usage.]                                  */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    void
programUsage(FILE *fp)
{
    fprintf(fp,
"Program usage: %s <Options>\n"
"Mandatory options:\n"
"  -idir <dir>         MRI input data directory.\n"
"\n"
"Image enhancement options:\n"
"  -toeplitz           Use the toeplitz reconstruction strategy. The def-\n"
"                      ault reconstruction strategy is brute force.\n"
"  -ntime_segs <num>   Specify how many time segments to be used by Toep-\n"
"                      litz reconstruction strategy.\n"
"  -cg_num <num>       Iteration number for Conjugate Gradient per slice.\n"
"                      Default is %d.\n"
"  -reg                Enable roughness regularization.\n"
"  -fd                 Enable finite difference.\n"
"  -fdp <num>          Specify finite difference penalizer value.\n"
"                      NO DEFAULT VALUE. HAS TO BE SET MANUALLY..\n"
"  -tv                 Enable total variation\n"
"  -tv_num <num>       Iteration number for total variation.\n"
"                      Default is %d.\n"
"  -tv_update          Update the outputs for each TV iteration.\n"
"\n"
"Miscellaneous options:\n"
"  -odir <dir>         MRI output data directory.\n"
"                      If the output directory is not provided,\n"
"                      \"<idir>/output\" is used.\n"
"  -cpu                Enable the CPU part computation as comparison.\n"
"                      Default is not enabled.\n"
"  -mgpu               Enable multiple GPU computatoin when possible.\n"
"                      When not specified, only one GPU is used.\n"
"  -nogpu              Disable the GPU part computation.\n"
"                      This enables the CPU computation automatically.\n"
"  -version            Version of this release and the license.\n"
"  -help               This message.\n"
"\n"
"Brute force examples:\n"
"  Run GPU computation on given data samples:\n"
"  $> ./mriSolver -idir mriData/128x128x1\n"
"\n"
"  Run both GPU and CPU computations with 8 CG iterations:\n"
"  $> ./mriSolver -idir mriData/128x128x1 -cg_num 8 -cpu\n"
"\n"
"  Enable roughness regularization with 10 CG iterations:\n"
"  $> ./mriSolver -idir mriData/128x128x1 -cg_num 10 -reg\n"
"\n"
"  Enable finite variation and total variation for 10 iterations:\n"
"  $> ./mriSolver -idir mriData/128x128x1 -fd -tv -tv_num 10\n"
"Toeplitz examples:\n"
"  $> ./mriSolver -idir mriData/64x64x1 -cg_num 21 -toeplitz  -tv -tv_num 1 -fdp 1e-6 -ntime_segs 12\n"
"\n"
, mriSolver_name, CG_NUM_DEFAULT, TV_NUM_DEFAULT
        );
}


