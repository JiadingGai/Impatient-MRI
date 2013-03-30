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

    File Name   [structure.cpp]

    Synopsis    [This file defines the common data structures used in the
        whole application.]

    Description []

    Revision    [0.1.1; Change all data structures to C++ forms, Code cleaning;
                 Xiao-Long Wu, ECE UIUC]
    Revision    [1.0a; Further optimization, Code cleaning, Adding comments;
                 Xiao-Long Wu, ECE UIUC, Jiading Gai, Beckman Institute]
    Date        [10/27/2010]

*****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>
#include <map>
#include <stdio.h>
#include <stdlib.h>

// CUDA library
#include <cutil.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>

// XCPPLIB libraries
#include <xcpplib_process.h>
#include <xcpplib_types.h>
#include <xcpplib_typesGpu.cuh>

// Project header files
#include <structures.h>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

//namespace uiuc_mri {

/*---------------------------------------------------------------------------*/
/*  Namespace used                                                           */
/*---------------------------------------------------------------------------*/

using namespace std;
using namespace xcpplib;

/*---------------------------------------------------------------------------*/
/*  Performance evaluation data structures                                   */
/*---------------------------------------------------------------------------*/

// Flop object for global accesses.
mriFlop mriFlop_g;

    mriFlop *
getMriFlop(void) { return &mriFlop_g; }

// Timer object for global accesses.
mriTimer mriTimer_g;

    mriTimer *
getMriTimer(void) { return &mriTimer_g; }

    void
initMriTimer(void) { mriTimer_g.initTimer(); }

    void
deleteMriTimer(void) { mriTimer_g.deleteTimer(); }

    void
startMriTimer(unsigned int &timer) { mriTimer_g.startTimer(timer); }

    void
stopMriTimer(unsigned int &timer) { mriTimer_g.stopTimer(timer); }

    void
printMriTimer(void) { mriTimer_g.printTimer(); }

/*---------------------------------------------------------------------------*/
/*  Memory usage tracing routines                                            */
/*---------------------------------------------------------------------------*/

#if DEBUG_MEMORY
    MemoryTrace mriMemoryTraceCpu("CPU", true);
    MemoryTrace mriMemoryTraceGpu("GPU", true);
#else
    MemoryTrace mriMemoryTraceCpu("CPU", false);
    MemoryTrace mriMemoryTraceGpu("GPU", false);
#endif

    void
insertMemoryUsageCpu(void *addr, const unsigned int usage)
{
    mriMemoryTraceCpu.insert(addr, usage);
}
    void
insertMemoryUsageGpu(void *addr, const unsigned int usage)
{
    mriMemoryTraceGpu.insert(addr, usage);
}

    void
eraseMemoryUsageCpu(void *addr)
{
    mriMemoryTraceCpu.erase(addr);
}
    void
eraseMemoryUsageGpu(void *addr)
{
    mriMemoryTraceGpu.erase(addr);
}

    unsigned int
getMemoryUsageCpu(void)
{
    return mriMemoryTraceCpu.getUsage();
}
    unsigned int
getMemoryUsageGpu(void)
{
    return mriMemoryTraceGpu.getUsage();
}

/*---------------------------------------------------------------------------*/
/*  Application-specific data structures                                     */
/*---------------------------------------------------------------------------*/

    template <typename T>
    T *
new_array(const size_t N, const memory_location loc)
{
    //dispatch on location
    if (loc == HOST_MEMORY) {
        return mriNewCpu<T>(N);
    } else {
        return mriNewGpu<T>(N);
    }
}

    template <typename T>
    void 
delete_array(T *p, const memory_location loc)
{
    //dispatch on location
    if (loc == HOST_MEMORY) {
        mriDeleteCpu(p);
    } else {
        mriDeleteGpu(p);
    };
}

    template<typename T>
    T *
new_host_array(const size_t N)
{
    return new_array<T>(N, HOST_MEMORY);
}

    template<typename T>
    T *
new_device_array(const size_t N)
{
    return new_array<T>(N, DEVICE_MEMORY);
}

    template<typename T>
    void 
delete_host_array(T *p)
{
    delete_array(p, HOST_MEMORY);
}

    template<typename T>
    void
delete_device_array(T *p)
{
    delete_array(p, DEVICE_MEMORY);
}

    void 
delete_CooMatrix(CooMatrix&coo, const memory_location loc)
{
    delete_array(coo.I, loc);
    delete_array(coo.J, loc);
    delete_array(coo.V, loc);
}

    void 
delete_host_matrix(CooMatrix&coo)
{
    delete_CooMatrix(coo, HOST_MEMORY);
}

/*---------------------------------------------------------------------------*/
/*  Sparse-matrix manipulation function definitions                          */
/*---------------------------------------------------------------------------*/

////////////////////////////////////////////////////////////////////////////////
//! Sum together the duplicate nonzeros in a CSR format
//! CSR format will be modified *in place*
//! @param num_rows       number of rows
//! @param num_cols       number of columns
//! @param Ap             CSR pointer array
//! @param Ai             CSR index array
//! @param Ax             CSR data array
////////////////////////////////////////////////////////////////////////////////
    void 
sum_csr_duplicates(
    const int num_rows, const int num_cols,
    int *Ap, int *Aj, FLOAT_T *Ax)
{
    int *next = new_host_array<int>(num_cols);
    FLOAT_T *sums = new_host_array<FLOAT_T>(num_cols);

    for (int i = 0; i < num_cols; i++) {
        next[i] = (int) -1;
        sums[i] = (FLOAT_T)   0;
    }

    int NNZ = 0;

    int row_start = 0;
    int row_end   = 0;

    for (int i = 0; i < num_rows; i++) {
        int head = (int)-2;

        row_start = row_end; //Ap[i] may have been changed
        row_end   = Ap[i + 1]; //Ap[i+1] is safe

        for (int jj = row_start; jj < row_end; jj++) {
            int j = Aj[jj];

            sums[j] += Ax[jj];
            if (next[j] == (int)-1) {
                next[j] = head;
                head    = j;
            }
        }

        while (head != (int)-2) {
            int curr = head; //current column
            head   = next[curr];

            if (sums[curr] != 0) {
                Aj[NNZ] = curr;
                Ax[NNZ] = sums[curr];
                NNZ++;
            }

            next[curr] = (int)-1;
            sums[curr] =  0;
        }
        Ap[i + 1] = NNZ;
    }

    delete_host_array(next);
    delete_host_array(sums);
}

////////////////////////////////////////////////////////////////////////////////
//! Convert COO format to CSR format
// Storage for output is assumed to have been allocated
//! @param rows           COO row array
//! @param cols           COO column array
//! @param data           COO data array
//! @param num_rows       number of rows
//! @param num_cols       number of columns
//! @param num_nonzeros   number of nonzeros
//! @param Ap             CSR pointer array
//! @param Ai             CSR index array
//! @param Ax             CSR data array
////////////////////////////////////////////////////////////////////////////////
    void 
coo2Csr(
    const int *rows, const int *cols, const FLOAT_T *data,
    const int num_rows, const int num_cols, const int num_nonzeros,
    int *Ap, int *Aj, FLOAT_T *Ax)
{
    for (int i = 0; i < num_rows; i++) Ap[i] = 0;
    for (int i = 0; i < num_nonzeros; i++) Ap[rows[i]]++;

    //cumsum the nnz per row to get Bp[]
    for (int i = 0, cumsum = 0; i < num_rows; i++) {
        int temp = Ap[i];
        Ap[i] = cumsum;
        cumsum += temp;
    }
    Ap[num_rows] = num_nonzeros;


//Jiading GAI
#if 0
	for (int i = 0; i < num_nonzeros; i++)
    {
	    Aj[i] = cols[i];
		Ax[i] = data[i];
	}
#else	
    //FIXME: what's Bj, Bx: write Aj,Ax into Bj,Bx
    for (int i = 0; i < num_nonzeros; i++) {
        int row  = rows[i];
        int dest = Ap[row];

        if (dest > num_nonzeros) {
            printf("*** error: dest value exceeds the maximum nonzeros.\n");
            printf("    dest: %d, num_nonzeros: %d\n", dest, num_nonzeros);
            fflush(stdout);
			_Exit(1);
        }
        int c = cols[i];
        Aj[dest] = c;
        Ax[dest] = data[i];

        Ap[row]++;
    }

    for (int i = 0, last = 0; i <= num_rows; i++) {
        int temp = Ap[i];
        Ap[i]  = last;
        last   = temp;
    }
#endif	
}

////////////////////////////////////////////////////////////////////////////////
//! Convert COOrdinate format (triplet) to CSR format
//! @param coo        CooMatrix
////////////////////////////////////////////////////////////////////////////////
    CsrMatrix
coo2Csr(const CooMatrix &coo, const bool compact)
{
    #if DEBUG_MODE // Error check
    coo.print("coo2Csr()");
    #endif

    CsrMatrix csr(coo);
    coo2Csr(coo.I, coo.J, coo.V,
            coo.num_rows, coo.num_cols, coo.num_nonzeros,
            csr.Ap, csr.Aj, csr.Ax);

    if (compact) {
        //sum duplicates together
        sum_csr_duplicates(csr.num_rows, csr.num_cols, csr.Ap, csr.Aj, csr.Ax);
        csr.num_nonzeros = csr.Ap[csr.num_rows];
    }

    return csr;
}

    CsrMatrix
mtx2Csr(
    const int *I, const int *J, const FLOAT_T *V,
    const int num_rows, const int num_cols, const int num_nonzeros)
{
    #if DEBUG_MODE // Error check
    printf("mtx2Csr()\n");
    printf("num_rows: %d, num_cols: %d, num_nonzeros: %d\n",
        num_rows, num_cols, num_nonzeros);
    for (int i = 0; i < num_nonzeros; i++) {
        if (I[i] >= num_rows) {
            printf("*** Errors on I %d at line %d of %s\n",
                i, __LINE__, __FILE__);
        }
        if (J[i] >= num_cols) {
            printf("*** Errors on J %d at line %d of %s\n",
                i, __LINE__, __FILE__);
        }
    }
    #endif

    CooMatrix coo(num_rows, num_cols, num_nonzeros, I, J, V);

    CsrMatrix csr = coo2Csr(coo, false);

    return csr;
}

    CooMatrix
read_CooMatrix(const char *mm_filename)
{
    CooMatrix coo;

    FILE *fid;
    MM_typecode matcode;

    fid = openFile(mm_filename, "r", !DEBUG_MODE);

    if (fid == NULL) {
        printf("Unable to open file %s\n", mm_filename);
        exit(1);
    }

    if (mm_read_banner(fid, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (!mm_is_valid(matcode)) {
        printf("Invalid Matrix Market file.\n");
        exit(1);
    }

    if (!((mm_is_real(matcode) || mm_is_integer(matcode) ||
        mm_is_pattern(matcode)) && mm_is_coordinate(matcode) &&
        mm_is_sparse(matcode))) {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        printf("Only sparse real-valued or pattern coordinate matrices are supported\n");
        exit(1);
    }

    int num_rows, num_cols, num_nonzeros;
    if (mm_read_mtx_crd_size(fid, &num_rows, &num_cols,
        &num_nonzeros) != 0) {
        exit(1);
    }

    coo.num_rows     = (int) num_rows;
    coo.num_cols     = (int) num_cols;
    coo.num_nonzeros = (int) num_nonzeros;

    coo.I = new_host_array<int>(coo.num_nonzeros);
    coo.J = new_host_array<int>(coo.num_nonzeros);
    coo.V = new_host_array<FLOAT_T>(coo.num_nonzeros);

    printf("Reading sparse matrix from file (%s):", mm_filename);
    fflush(stdout);

    if (mm_is_pattern(matcode)) {
        // pattern matrix defines sparsity pattern, but not values
        for (int i = 0; i < coo.num_nonzeros; i++ ) {
            if (fscanf(fid, "%d %d", &(coo.I[i]), &(coo.J[i])) == 2) {
                coo.I[i]--;      //adjust from 1-based to 0-based indexing
                coo.J[i]--;
                coo.V[i] = 1.0;  //use value 1.0 for all nonzero entries
            } else {
                fprintf(stderr, "*** Error at line %d of %s:\n", 
                    __LINE__, __FILE__);
                fprintf(stderr, "*** Error: Failed to read data from %s.\n",
                    mm_filename);
                exit(1);
            }
        }
    } else if (mm_is_real(matcode) || mm_is_integer(matcode)) {
        for ( int i = 0; i < coo.num_nonzeros; i++ ) {
            int I, J;
            double V;  // always read in a double and convert later if necessary

            if (fscanf(fid, "%d %d %lf", &I, &J, &V) == 3) {
                coo.I[i] = (int) I - 1;
                coo.J[i] = (int) J - 1;
                coo.V[i] = (FLOAT_T)  V;
            } else {
                fprintf(stderr, "*** Error at line %d of %s:\n", 
                    __LINE__, __FILE__);
                fprintf(stderr, "*** Error: Failed to read data from %s.\n",
                    mm_filename);
                exit(1);
            }
        }
    } else {
        printf("Unrecognized data type\n");
        exit(1);
    }

    fclose(fid); fid = NULL;
    printf(" done\n");

    if ( mm_is_symmetric(matcode)) { //duplicate off diagonal entries
        int off_diagonals = 0;
        for ( int i = 0; i < coo.num_nonzeros; i++ ) {
            if ( coo.I[i] != coo.J[i] )
                off_diagonals++;
        }

        int true_nonzeros = 2 * off_diagonals + (coo.num_nonzeros - off_diagonals);

        int *new_I = new_host_array<int>(true_nonzeros);
        int *new_J = new_host_array<int>(true_nonzeros);
        FLOAT_T *new_V = new_host_array<FLOAT_T>(true_nonzeros);

        int ptr = 0;
        for ( int i = 0; i < coo.num_nonzeros; i++ ) {
            if ( coo.I[i] != coo.J[i] ) {
                new_I[ptr] = coo.I[i];  new_J[ptr] = coo.J[i];  new_V[ptr] = coo.V[i];
                ptr++;
                new_J[ptr] = coo.I[i];  new_I[ptr] = coo.J[i];  new_V[ptr] = coo.V[i];
                ptr++;
            } else {
                new_I[ptr] = coo.I[i];  new_J[ptr] = coo.J[i];  new_V[ptr] = coo.V[i];
                ptr++;
            }
        }
        delete_host_array(coo.I); delete_host_array(coo.J); delete_host_array(coo.V);
        coo.I = new_I;  coo.J = new_J; coo.V = new_V;
        coo.num_nonzeros = true_nonzeros;
    } //end symmetric case

    return coo;
}

    CsrMatrix
read_CsrMatrix(const char *mm_filename, bool compact)
{
    CooMatrix coo = read_CooMatrix(mm_filename);
    CsrMatrix csr = coo2Csr(coo, compact);
    delete_host_matrix(coo);

    return csr;
}

/*---------------------------------------------------------------------------*/
/*  Other Function prototypes                                                */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Load data required for reconstruction from files (with      */
/*      specific format).]                                                   */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    void
loadInputData(
    const string &input_folder_path, FLOAT_T &version,
    int &num_slices, int &num_k, int &num_i, int &num_coil, 
    DataTrajArray &ktraj, DataTrajArray &itraj,
    TArray<FLOAT_T> &kdata_r, TArray<FLOAT_T> &kdata_i,
    TArray<FLOAT_T> &idata_r, TArray<FLOAT_T> &idata_i,
    TArray<FLOAT_T> &sensi_r, TArray<FLOAT_T> &sensi_i, 
    TArray<FLOAT_T> &fm, TArray<FLOAT_T> &t,
    CooMatrix &c, const bool enable_regularization,
	int &Nx, int &Ny, int &Nz)
{
    // Input data file names
    // ---------------------

    string kx_fn      = input_folder_path + "/kx.dat";
    string ky_fn      = input_folder_path + "/ky.dat";
    string kz_fn      = input_folder_path + "/kz.dat";
    string ix_fn      = input_folder_path + "/ix.dat";
    string iy_fn      = input_folder_path + "/iy.dat";
    string iz_fn      = input_folder_path + "/iz.dat";
    string kdata_r_fn = input_folder_path + "/kdata_r.dat";
    string kdata_i_fn = input_folder_path + "/kdata_i.dat";
    string idata_r_fn = input_folder_path + "/idata_r.dat";
    string idata_i_fn = input_folder_path + "/idata_i.dat";
    string sensi_r_fn = input_folder_path + "/sensi_r.dat";
    string sensi_i_fn = input_folder_path + "/sensi_i.dat";
    string fm_fn      = input_folder_path + "/fm.dat";
    string t_fn       = input_folder_path + "/t.dat";
    string c_fn       = input_folder_path + "/c.rmtx";

	// Test data format version (0.2 or 1.0 higher)
    FILE *fp0 = fopen(kx_fn.c_str(),"r");
	if(NULL==fp0) {
		printf("%s not found!\n",kx_fn.c_str());
		exit(1);
	}
	float the_version = -1.0f;
	//the_version should be 0.2 or 1.0 higher
	if(1!=fread(&the_version,sizeof(float),1,fp0)) {
		printf("Error: fread return value mismatch\n");
	    exit(1);
	}
	fclose(fp0);

    

    // Data dimensions
    // ---------------

    const int DIMENSION_SIZE = 2;
    int kx_dims[DIMENSION_SIZE] = {0};
    int ky_dims[DIMENSION_SIZE] = {0};
    int kz_dims[DIMENSION_SIZE] = {0};
    int ix_dims[DIMENSION_SIZE] = {0};
    int iy_dims[DIMENSION_SIZE] = {0};
    int iz_dims[DIMENSION_SIZE] = {0};
    int kdata_r_dims[DIMENSION_SIZE] = {0};
    int kdata_i_dims[DIMENSION_SIZE] = {0};
    int idata_r_dims[DIMENSION_SIZE] = {0};
    int idata_i_dims[DIMENSION_SIZE] = {0};
	int sensi_r_dims[DIMENSION_SIZE] = {0};
	int sensi_i_dims[DIMENSION_SIZE] = {0};
    int fm_dims[DIMENSION_SIZE] = {0};
    int t_dims[DIMENSION_SIZE] = {0};

    // Load data from files
    // --------------------

	int sensed_data_size = 0;
    TArray<FLOAT_T> kx, ky, kz, ix, iy, iz;

	if(the_version==0.2f){// backward compatiable with 0.2 format
       kx = readDataFile(kx_fn, version, kx_dims, num_coil,
           num_slices, sensed_data_size);
       ky = readDataFile(ky_fn, version, ky_dims, num_coil,
           num_slices, sensed_data_size);
       kz = readDataFile(kz_fn, version, kz_dims, num_coil,
           num_slices, sensed_data_size);
       num_k = kx_dims[0] * kx_dims[1];
       kdata_r = readDataFile(kdata_r_fn, version, kdata_r_dims, num_coil,
           num_slices, sensed_data_size);
       kdata_i = readDataFile(kdata_i_fn, version, kdata_i_dims, num_coil,
           num_slices, sensed_data_size);
       sensi_r = readDataFile(sensi_r_fn, version, sensi_r_dims, num_coil,
	       num_slices, sensed_data_size);
       sensi_i = readDataFile(sensi_i_fn, version, sensi_i_dims, num_coil,
	       num_slices, sensed_data_size);
       fm = readDataFile(fm_fn, version, fm_dims, num_coil,
           num_slices, sensed_data_size);

       //num_slices = (kdata_r_dims[0] * kdata_r_dims[1]) / (num_k*num_coil);
       ix = readDataFile(ix_fn, version, ix_dims, num_coil,
           num_slices, sensed_data_size);
       iy = readDataFile(iy_fn, version, iy_dims, num_coil,
           num_slices, sensed_data_size);
       iz = readDataFile(iz_fn, version, iz_dims, num_coil,
           num_slices, sensed_data_size);
       num_i = (ix_dims[0] * ix_dims[1]);
       idata_r = readDataFile(idata_r_fn, version, idata_r_dims, num_coil,
           num_slices, sensed_data_size);
       idata_i = readDataFile(idata_i_fn, version, idata_i_dims, num_coil,
           num_slices, sensed_data_size);
       t = readDataFile(t_fn, version, t_dims, num_coil, num_slices,
           sensed_data_size);
	}
	else{
       kx = readDataFile_10(kx_fn, version, kx_dims, num_coil,
           num_slices, sensed_data_size, Nx, Ny, Nz);
       ky = readDataFile_10(ky_fn, version, ky_dims, num_coil,
           num_slices, sensed_data_size, Nx, Ny, Nz);
       kz = readDataFile_10(kz_fn, version, kz_dims, num_coil,
           num_slices, sensed_data_size, Nx, Ny, Nz);
       num_k = kx_dims[0] * kx_dims[1];
       kdata_r = readDataFile_10(kdata_r_fn, version, kdata_r_dims, num_coil,
           num_slices, sensed_data_size, Nx, Ny, Nz);
       kdata_i = readDataFile_10(kdata_i_fn, version, kdata_i_dims, num_coil,
           num_slices, sensed_data_size, Nx, Ny, Nz);
       sensi_r = readDataFile_10(sensi_r_fn, version, sensi_r_dims, num_coil,
	       num_slices, sensed_data_size, Nx, Ny, Nz);
       sensi_i = readDataFile_10(sensi_i_fn, version, sensi_i_dims, num_coil,
	       num_slices, sensed_data_size, Nx, Ny, Nz);
       fm = readDataFile_10(fm_fn, version, fm_dims, num_coil,
           num_slices, sensed_data_size, Nx, Ny, Nz);

       //num_slices = (kdata_r_dims[0] * kdata_r_dims[1]) / (num_k*num_coil);
       ix = readDataFile_10(ix_fn, version, ix_dims, num_coil,
           num_slices, sensed_data_size, Nx, Ny, Nz);
       iy = readDataFile_10(iy_fn, version, iy_dims, num_coil,
           num_slices, sensed_data_size, Nx, Ny, Nz);
       iz = readDataFile_10(iz_fn, version, iz_dims, num_coil,
           num_slices, sensed_data_size, Nx, Ny, Nz);
       num_i = (ix_dims[0] * ix_dims[1]);
       idata_r = readDataFile_10(idata_r_fn, version, idata_r_dims, num_coil,
           num_slices, sensed_data_size, Nx, Ny, Nz);
       idata_i = readDataFile_10(idata_i_fn, version, idata_i_dims, num_coil,
           num_slices, sensed_data_size, Nx, Ny, Nz);
       t = readDataFile_10(t_fn, version, t_dims, num_coil, num_slices,
           sensed_data_size, Nx, Ny, Nz);
	}

	// One coil
	if (num_coil == 1) {
		for(int i=0; i<sensi_r_dims[0]; i++) {
			sensi_r.array[i] = 1.0;
			sensi_i.array[i] = 0.0;
		}
	}


    // Data for Sparse Matrix-Vector Multiplication
    // ============================================

    if (enable_regularization) {
        int num_cols = 0, num_rows = 0, num_nonzeros = 0;
        int *array_rows = NULL, *array_cols = NULL;
        FLOAT_T *array_values = NULL;
        readMtxFile(c_fn.c_str(), &num_cols, &num_rows, &array_rows,
            &array_cols, &array_values, &num_nonzeros, !DEBUG_MODE);

        c.num_rows = num_rows;
        c.num_cols = num_cols;
        c.num_nonzeros = num_nonzeros;
        c.I = array_rows;
        c.J = array_cols;
        c.V = array_values;
    }

    // Convert field map and trajectories into structures
    // --------------------------------------------------

    itraj.allocate(num_i);
    ktraj.allocate(num_k);
    for (int i = 0; i < num_i; i++) {
        itraj.array[i].x = ix.array[i];
        itraj.array[i].y = iy.array[i];
    }
    #if !DATATRAJ_NO_Z_DIM
    for (int s = 0; s < num_slices; s++) {
        for (int x = 0; x < num_i; x++) {
            // FIXME: Need to address the multislice problem
            // (*itraj)[s*(*num_i)+x].z=iz[s];
            itraj.array[x].z = iz.array[0];
        }
        for (int k = 0; k < num_k; k++) {
            // FIXME: 3D data
            //(*ktraj)[s*(*num_k)+k].z=kz[s];
            ktraj.array[k].z = kz.array[0];
        }
    }
    #endif
    for (int i = 0; i < num_k; i++) {
        ktraj.array[i].x = kx.array[i];
        ktraj.array[i].y = ky.array[i];
    }

    #if DEBUG_MODE // Error check
    if (enable_regularization) {
        c.print("loadInputData()");
    }
    #endif
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Load a single FLOAT_T array.]                               */
/*                                                                           */
/*  Description [The file format is starting with 1) file format version, 2) */
/*      the unit size per coil, per slice, 3) number of coils, 4) number of  */
/*      image slices, 5) the size of the whole data field within the file,   */
/*      and 6) the actual data values.]                                      */
/*                                                                           */
/*===========================================================================*/
    TArray<FLOAT_T>
readDataFile(const string &fn, FLOAT_T &version, int *data_dims, int &ncoils, int &nslices, int &sensed_data_size)
{
    FILE *fp = openFile(fn.c_str(), "rb", !DEBUG_MODE);

    float version_f;
    if (fread(&version_f, sizeof(float), 1, fp) != 1) {
        fprintf(stderr, "*** Error at line %d of %s:\n", __LINE__, __FILE__);
        fprintf(stderr, "*** Error: Missing version number in %s.\n",
            fn.c_str());
        exit(1);
    }
    version = version_f;

    // Read data dimensions
    int dim1 = 0;
    int dim2 = 1;
    if (fread(&dim1, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "*** Error at line %d of %s:\n", __LINE__, __FILE__);
        fprintf(stderr, "*** Error: Missing first dimension number in %s.\n",
            fn.c_str());
        exit(1);
    }

    data_dims[0] = dim1;
    data_dims[1] = dim2;

    if (fread(&ncoils, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "*** Error at line %d of %s:\n", __LINE__, __FILE__);
        fprintf(stderr, "*** Error: Missing ncoils number in %s.\n",
            fn.c_str());
        exit(1);
    }
    if (fread(&nslices, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "*** Error at line %d of %s:\n", __LINE__, __FILE__);
        fprintf(stderr, "*** Error: Missing nslices number in %s.\n",
            fn.c_str());
        exit(1);
    }
    if (fread(&sensed_data_size, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "*** Error at line %d of %s:\n", __LINE__, __FILE__);
        fprintf(stderr, "*** Error: Missing total data size inside the file number in %s.\n",
            fn.c_str());
        exit(1);
    }

    #if debug_io_msg
    printf("size: %d x %d.\n", data_dims[0], data_dims[1]);
    printf("sensed_data_size: %d\n", sensed_data_size);
    #endif

    // Allocate data
    TArray<FLOAT_T> data(sensed_data_size);

    // Load data: Data is stored in float type.
    #if ENABLE_DOUBLE_PRECISION // When FLOAT_T is double
    
    float *data_f = mriNewCpu<float>(sensed_data_size);
    int result = fread(data_f, sizeof(float), sensed_data_size, fp);
    if (result != sensed_data_size) {
        outputMsg("Data size is not correct.", false);
        outputMsg("Error while reading data from disk.", true);
    }
    fclose(fp); fp = NULL;

    for (int i = 0; i < sensed_data_size; i++) {
        data.array[i] = data_f[i];
    }
    mriDeleteCpu(data_f);
    
    #else // When FLOAT_T is float type.
    
    int result = fread(data.array, sizeof(FLOAT_T), sensed_data_size, fp);
    if (result != sensed_data_size) {
        outputMsg("Data size is not correct.", false);
        outputMsg("Error while reading data from disk.", true);
    }
    fclose(fp); fp = NULL;
    #endif

    return data;
}


    TArray<FLOAT_T>
readDataFile_10(const string &fn, FLOAT_T &version, int *data_dims, int &ncoils, 
		        int &nslices, int &sensed_data_size, int &Nx, int &Ny, int &Nz)
{
   const char impatient_keywords[9][200] = { "version", "xDimension", 
   "yDimension", "zDimension", "coil_number", "slice_number", "file_size", 
   "Binary_Size","Binary:"};

   int xDimension = -1, yDimension = -1, zDimension = -1;
   int coil_number = -1, slice_number = -1;
   int file_size = -1;
   int Binary_Size = -1;

   float *data_pointer = NULL;

   int data_size_per_coil = -1;

   //To count how many data points are read from each binary section.
   //combined_section_size has to equal file_size, otherwise you are
   //in trouble!
   int combined_section_size = 0;

   FILE *fp = fopen(fn.c_str(), "r");
   if(fp == NULL) {
	  printf("Error: Cannot open file %s for writing\n", fn.c_str());
	  exit(1);
   }
   else {
	  //printf("Processing %s \n",fn.c_str());
   }

   while (1) // loop thru the input file until "EOF" is detected.
   {
       int bufsize = 200;//Maximum number of characters per line
       char *buf = (char *) malloc(bufsize*sizeof(char));
       memset(buf, 0, bufsize*sizeof(char));

	   int cur_character;
       int begin_idx = 0;
       //Read the first line into buf
	   while ( '\n' != (cur_character=fgetc(fp)) )   
       {
         buf[begin_idx] = cur_character;
	     begin_idx++;
       }
       buf[begin_idx] = '\n';

       // Matching the sym op value pattern.
	   // each line could start with a number of white spaces
	   // as long as the whole line is less than bufsize.
       int symbol_start = 0, symbol_end = symbol_start;
       while( ' ' == buf[symbol_start] )//skip white spaces
       {
          symbol_start++;
          symbol_end++;
       }
	   if('\n'==buf[symbol_start])//if it's an empty line, continue to next line.
	   {
		   if(buf)
             free(buf);
		   continue;
	   }
       while( ' ' != buf[symbol_end] && '=' != buf[symbol_end] && '\n' != buf[symbol_end])
       {
          symbol_end++;
       }
       symbol_end--;//retract the end point to the previous location.

	   if ('/'==buf[symbol_start] && '/'==buf[symbol_start+1])
	   {	   
	       //ignore the comment line if it starts with "//"
	       if(buf)
			   free(buf);
		   continue;
	   }
	   else if ('E'==buf[symbol_start] && 'O'==buf[symbol_start+1] && 'F'==buf[symbol_start+2])
	   {
		  //reaching the end of input.
	      //printf("End of Input.\n");
		  //fflush(stdout);
		  if(buf)
			  free(buf);
		  break;
	   }

	   // Matching for the equal-sign operator
       int op_start = symbol_end + 1, op_end = op_start;
       while ( ' ' == buf[op_start] )//skip white spaces
       {
          op_start++;
	      op_end++;
       }
       if('=' != buf[op_end])
       {
          printf("Data format error: symbol should be followed by an equal sign at %d of %s!\n",__LINE__,__FILE__);
		  fflush(stdout);
	      _Exit(0);
       }

       int val_start = op_end + 1, val_end = val_start;
       while ( ' ' == buf[val_start] ) //skip white spaces
       {
          val_start++;
	      val_end++;
       }
       while( ' ' != buf[val_end] && '\n' != buf[val_end] )
       {
          val_end++;
       }
       val_end--;//retract the end point to the previous location.


       int count;
       char *symbol_temp = (char *) malloc(bufsize*sizeof(char));//copy symbol from buf to here.
       memset(symbol_temp, 0, bufsize*sizeof(char));
       count = 0;
       while ( symbol_start <= symbol_end )
       {
          //copy symbol from buf to here.
          symbol_temp[count++] = buf[symbol_start++];
       }

       char *value_temp = (char *) malloc(bufsize*sizeof(char));//copy value from buf to here.
       memset(value_temp , 0, bufsize*sizeof(char));
       count = 0;
       while ( val_start <= val_end )
       {
         //copy value from buf to here.
         value_temp[count++] = buf[val_start++];
       }
   
       if(0==strcmp(symbol_temp,impatient_keywords[0]))
       {
          version = atof( value_temp );
		  //printf("version = %f\n",version);
       }
	   else if(0==strcmp(symbol_temp,impatient_keywords[1]))
       {
          xDimension = atoi( value_temp );
		  Nx = xDimension;
		  //printf("xDimension = %d\n", xDimension);
       }
	   else if(0==strcmp(symbol_temp,impatient_keywords[2]))
       {
          yDimension = atoi( value_temp );
		  Ny = yDimension;
		  //printf("yDimension = %d\n", yDimension);
       }
	   else if(0==strcmp(symbol_temp,impatient_keywords[3]))
       {
          zDimension = atoi( value_temp );
		  Nz = zDimension;
		  //printf("zDimension = %d\n", zDimension);
       }
	   else if(0==strcmp(symbol_temp,impatient_keywords[4]))
       {
          coil_number = atoi( value_temp );
		  ncoils = coil_number;
		  //printf("coil_number = %d\n", coil_number);
       } 
	   else if(0==strcmp(symbol_temp,impatient_keywords[5]))
       {
          slice_number = atoi( value_temp );
		  nslices = slice_number;
		  //printf("slice_number = %d\n", slice_number);
       } 
	   else if(0==strcmp(symbol_temp,impatient_keywords[6]))
	   {
	      file_size = atoi( value_temp );
		  //printf("file_size = %d\n", file_size);
		  sensed_data_size = file_size;
		  if(-1==coil_number)
		  {
             coil_number = 1;
             #if DEBUG_MODE
		     printf("coil_number unspecified in this file %s.\n",fn.c_str());
			 //exit(1);
             #endif
		  }

		  data_size_per_coil = file_size / coil_number;
		  int dim1 = data_size_per_coil;
          int dim2 = 1;
          data_dims[0] = dim1;
          data_dims[1] = dim2;
		  
	      data_pointer = (float *) malloc(file_size * sizeof(float));
	   }
	   else if(0==strcmp(symbol_temp,impatient_keywords[7]))
       {
          if(file_size==-1) {
		     printf("Error: invalid file_size in this file\n");
			 exit(1);
		  }

          Binary_Size = atoi( value_temp );//printf("Binary_Size = %d\n", Binary_Size);
		  // Up to any abitrary number of lines of comments in between, 
		  // <Binary_Size> has to be followed by <Binary:> and then 
		  // binary data ... so the following code looks for <Binary:>
		  while (1) 
		  {
            int cur_character;
            memset(buf, 0, bufsize*sizeof(char));
            int begin_idx = 0;
            while ( '\n' != (cur_character=fgetc(fp)) )   
            {
               buf[begin_idx] = cur_character;
	           begin_idx++;
            }
            buf[begin_idx] = '\n';

            int symbol_start = 0, symbol_end = symbol_start;
            while( ' ' == buf[symbol_start] )
            {
              symbol_start++;
              symbol_end++;
            }
	        if('\n'==buf[symbol_start])//if it's an empty line, continue to next line.
			{
				continue;
			}
            while( '\n' != buf[symbol_end] )
            {
               symbol_end++;
            }
            symbol_end--;//retract the end point to the previous location.

	        //ignore any comments
	        if ('/'==buf[symbol_start] && '/'==buf[symbol_start+1])
			{
				continue;
            }

            int count;
            char *Binary_symbol = (char *) malloc(bufsize*sizeof(char));
            memset(Binary_symbol, 0, bufsize*sizeof(char));
            count = 0;
            while ( symbol_start <= symbol_end )
            {
              Binary_symbol[count++] = buf[symbol_start++];
            }
	        
			if(0==strcmp(Binary_symbol,impatient_keywords[8]))
			{
			   //printf("Binary: time to read binary data\n");
			   if(Binary_symbol)
				   free(Binary_symbol);
			   break;// if found a pair, no need to loop no more.
			}
			else
			{
			   printf("Data format error: Binary_Size has to pair with Binary:!\n");
			   fflush(stdout);
			   _Exit(0);
			}
			if(Binary_symbol)
				free(Binary_symbol);
		  }

          int fread_returned = (int) fread(data_pointer+combined_section_size, sizeof(float), Binary_Size, fp);
		  if(Binary_Size!=fread_returned)
		  {
		    printf("Error:fread did not read enough data!\n");
			exit(1);
		  }
		  combined_section_size += Binary_Size;

       }
	   if(symbol_temp)
         free(symbol_temp);
	   if(value_temp)
         free(value_temp);
	   if(buf)
         free(buf);
   } // end of the most outer while(1)
   fclose(fp);

   if(combined_section_size!=file_size) {
     printf("The combined number of elements from multiple binary sections do \
			 not match file_size.\n ");
	 exit(1);
   }

    #if debug_io_msg
    printf("size: %d x %d.\n", data_dims[0], data_dims[1]);
    printf("sensed_data_size: %d\n", sensed_data_size);
    #endif

    // Allocate data
    TArray<FLOAT_T> data(sensed_data_size);
    for (int i = 0; i < sensed_data_size; i++) {
        data.array[i] = data_pointer[i];
    }

	free(data_pointer);
    return data;
}

// ==================== exportDataCpu ====================
    void 
exportDataCpu(
    const string &fn, const FLOAT_T *array, int num_elements
    )
{
    FILE *fid = openFile(fn.c_str(), "wt", !DEBUG_MODE);
    for (int i = 0; i < num_elements; i++) {
        fprintf(fid, "%f\n", array[i]); // FIXME: Can be faster.
    }
    fclose(fid);
}
    void 
exportDataCpu(
    const string &fn, const TArray<FLOAT_T> &fa, int num_elements
    )
{
    exportDataCpu(fn, fa.array, num_elements);
}

// ==================== exportDataGpu ====================
    void 
exportDataGpu(const string &fn, FLOAT_T *array, int num_elements)
{
    FLOAT_T *tmp = mriNewCpu<FLOAT_T>(num_elements);
    cudaMemcpy(tmp, array, num_elements * sizeof(FLOAT_T), cudaMemcpyDeviceToHost);
    FILE *fid = openFile(fn.c_str(), "wt", !DEBUG_MODE);
    for (int i = 0; i < num_elements; i++) {
        fprintf(fid, "%f\n", tmp[i]);
    }
    fclose(fid);
    mriDeleteCpu(tmp);
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Pad a given vector to the size of power of two.]            */
/*                                                                           */
/*  Description [The padded items will be filled with zeros. The returned    */
/*      vector must be freed by the caller function.]                        */
/*                                                                           */
/*===========================================================================*/
    FLOAT_T *
padVectorPowerOfTwo(const FLOAT_T *array, const int element_num)
{
    FLOAT_T *a = NULL;

    if (!isPowerOfTwo(element_num)) {
        int size_v = getLeastPowerOfTwo(element_num);

        // For example, we must pad 3770 to 4096 for easy manipulation in GPU.
        a = mriNewCpu<FLOAT_T>(size_v);
        for (int i = 0; i < element_num; i++) {
            a[i] = array[i];
        }
        for (int i = element_num; i < size_v; i++) {
            a[i] = 0.0;
        }
        // Not free original array
        //delete [] array;

    // Input size is multiples of vectorProductGpu_BLOCK_SIZE.
    } else {
        a = mriNewCpu<FLOAT_T>(element_num);
        memcpy(a, array, sizeof(array));
    }

    return a;
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

//}

