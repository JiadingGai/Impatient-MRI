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

    File Name   [utils.h]

    Synopsis    [Toeplitz's utility library header.]

    Description []

    Revision    [1.0; Jiading Gai, Beckman Institute UIUC]
    Date        [03/23/2011]

*****************************************************************************/


#ifndef UTILS_STONE
#define UTILS_STONE

#include <cufft.h>

//#define GEFORCE_8800
#define CUDA11
//#define CUDA10
             
#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#ifdef CUDA11
#define REAL .x
#define IMAG .y
#else
#define REAL [0]
#define IMAG [1]
#endif

struct matrix_t {
    int numDims;
    int* dims;
    int elems;
    int isComplex;
    cufftComplex* host;
    cufftComplex* device;

    matrix_t();
    ~matrix_t();

    void init(int srcNumDims, int* srcDims, int srcElems);
    void init2D(int dim1, int dim2);
    void init3D(int dim1, int dim2, int dim3);

    void writeB(char fName[]);
    void write(char fName[]);

    void read(char fName[]);
    void read_pack(char fName[], int zeroComplex, int offset=0);

    void reshapeVecToMat3D(int dim2, int dim1, int dim0);

    void clear_host();
    void alloc_device();
    void copy_to_device();
    void copy_to_host();
    void free_device();
    void assign_device(matrix_t* m);

    void debug_print(char* name, int first, int last);
};

struct sparse_matrix_t {
    int numRows;
    int numCols;
    int maxNumNonZeroElems;
    int numNonZeroElems;

    int *jc;
    int *ir;
    float *rVals;
    float *iVals;

    int *devJc;
    int *devIr;
    float *devRVals;

    int isComplex;

	sparse_matrix_t();
	~sparse_matrix_t();

	void alloc_device();
	void free_device();
	void copy_to_device();

	void init_JGAI(int nRows, int nCols, int maxNonZeroElems, int nNonZeroElems);

    // length(ir) = numNonZeroElems
    // length(jc) = numCols+1
    // length(rVals) = numNonZeroElems
	void write(char fName[]);
	void read(char fName[]);

    // Returns 1 if result is complex. Returns 0 otherwise.
    // lhs = this * m
	int mult(matrix_t* lhs, matrix_t *m);

    // Returns 1 if result is complex. Returns 0 otherwise.
    // lhs = this' * m
	int mult_trans(matrix_t* lhs, matrix_t *m);
};

   float* 
readDataFile_JGAI(const char* file_name, float& version, 
                  int& data_size_per_coil, int& ncoils, 
                  int& nslices, int& sensed_data_size);


   float* 
readDataFile_JGAI_10(const char* file_name, float& version, 
		             int& data_size_per_coil, int& ncoils, 
					 int& nslices, int& sensed_data_size);


   void 
writeDataFile_JGAI(const char *file_name, float version, 
		           int data_size_per_coil, int ncoils, 
				   int nslices, int sensed_data_size, float *data_pointer);

  void 
writeDataFile_JGAI_10(const char *file_name, float version, int xDimension,
		              int yDimension, int zDimension, int coil_number, 
					  int slice_number, int file_size, float *data_pointer);

  void 
print_device_mem(float *p_d, int len, int start, int end, int interval);

  void 
setupMemoryGPU(int num, int size, float *&dev_ptr, float *&host_ptr);

  void
measureMemoryUsageGPU(void);


#endif
