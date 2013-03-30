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

    File Name   [utils.cpp]

    Synopsis    [Toeplitz's utility library code.]

    Description []

    Revision    [1.0; Jiading Gai, Beckman Institute UIUC]
    Date        [03/23/2011]

*****************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cutil.h>
#include "utils.h"


matrix_t::matrix_t() : numDims(0), dims(NULL), elems(0), isComplex(0), host(NULL), device(NULL) 
{ }

matrix_t::~matrix_t()
{
    if (dims) free(dims);
    if (host) free(host);
    if (device) cudaFree(device);
}

void matrix_t::init(int srcNumDims, int* srcDims, int srcElems)
{
    int x = 0;

    numDims = srcNumDims;
    dims = (int*)malloc(numDims * sizeof(int));
    for (x = 0; x < numDims; x++)
        dims[x] = srcDims[x];
    elems = srcElems;
    host = (cufftComplex*)calloc(elems, sizeof(cufftComplex));
}

void matrix_t::init2D(int dim1, int dim2)
{
    numDims = 2;
    dims = (int*)malloc(2 * sizeof(int));
    dims[0] = dim1;
    dims[1] = dim2;
    elems = dim1*dim2;
    host = (cufftComplex*)calloc(elems, sizeof(cufftComplex));
    alloc_device();
	copy_to_device();
}

void matrix_t::init3D(int dim1, int dim2, int dim3)
{
    numDims = 3;
    dims = (int*)malloc(3 * sizeof(int));
    dims[0] = dim1;
    dims[1] = dim2;
    dims[2] = dim3;
    elems = dim1*dim2*dim3;
    host = (cufftComplex*)calloc(elems, sizeof(cufftComplex));
}

void matrix_t::writeB(char fName[])
{  // Write to disk in binary format.
   float *real, *imag;
   real = (float*)calloc(elems, sizeof(float));
   imag = (float*)calloc(elems, sizeof(float));
   cudaMemcpy(host, device, elems * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

   for (int i = 0; i < elems; i++) {
       real[i] = host[i]REAL;
	   imag[i] = host[i]IMAG;
    }

	FILE* fid = fopen(fName, "w");
	fwrite(real, sizeof(float), elems, fid);
	fwrite(imag, sizeof(float), elems, fid);
	free(real);
	free(imag);
	fclose(fid);	  
}

void matrix_t::write(char fName[]) {
    FILE* fid = fopen(fName, "w");
    fprintf(fid, "%d ", numDims);
    for (int d = 0; d < numDims; d++)
       fprintf(fid, "%d ", dims[d]);
    fprintf(fid, "%d %d\n", elems, isComplex);
    fwrite(host, sizeof(cufftComplex), elems, fid);
    fclose(fid);
}

void matrix_t::read(char fName[]) {
    FILE* fid = fopen(fName, "r");
    if ( EOF==fscanf(fid, "%d ", &numDims) )
    {
       printf("Error:matrix_t::read\n");
       exit(1);
	}
    int a;
    dims = (int*)malloc(numDims * sizeof(int));
    for (int d = 0; d < numDims; d++) {
		if ( EOF==fscanf(fid, "%d ", &a) )
        {
           printf("Error:matrix_t::read\n");
           exit(1);
		}
       dims[d] = a;
    }

    if ( EOF==fscanf(fid, "%d %d\n", &elems, &isComplex) )
    {
        printf("Error: matrix_t::read\n");
        exit(1);
	}
    host = (cufftComplex*)calloc(elems, sizeof(cufftComplex));

    if ( elems!=fread(host, sizeof(cufftComplex), elems, fid) )
    {
        printf("Error: matrix_t::read\n");
        exit(1);
	}

    fclose(fid);
}

void matrix_t::read_pack(char fName[], int zeroComplex, int offset) {
    // offset is equivalent to l; it's used to facilitate reading
    // multiple Q_l (corresponding to different time segments) from 
    // the disk. The default is 0.
    FILE* fid = fopen(fName, "r");

    isComplex = 1;

#ifdef DPINPUTS
    double* real = (double*)calloc(elems, sizeof(double));
    double* imag = (double*)calloc(elems, sizeof(double));
    fseek(fid, offset*2*elems*sizeof(double), SEEK_SET);
    fread(real, sizeof(double), elems, fid);
    fread(imag, sizeof(double), elems, fid);
#else
    float* real = (float*)calloc(elems, sizeof(float));
    float* imag = (float*)calloc(elems, sizeof(float));
    fseek(fid, offset*2*elems*sizeof(float), SEEK_SET);
    if ( elems!=fread(real, sizeof(float), elems, fid) )
	{
        printf("Error: matrix_t::read_pack\n");
        exit(1);
    }
    if ( elems!=fread(imag, sizeof(float), elems, fid) )
	{
        printf("Error: matrix_t::read_pack\n");
        exit(1);
    }
#endif

    if (zeroComplex) { //i.e., symTraj == 1
      for (int i = 0; i < elems; i++) {
        host[i]REAL = real[i];
        // host[i]IMAG = 0.0f;
      }
    }
    else {
      for (int i = 0; i < elems; i++) {
        host[i]REAL = real[i];
        host[i]IMAG = imag[i];
      }
    }

    fclose(fid);
    free(real);
    free(imag);
}

void matrix_t::reshapeVecToMat3D(int dim2, int dim1, int dim0)
{
    assert(dims);
    free(dims);
    numDims = 3;
    dims = (int*)malloc(numDims * sizeof(int));
    dims[2] = dim2;
    dims[1] = dim1;
    dims[0] = dim0;
}

void matrix_t::clear_host()
{
    memset(host, 0, elems*sizeof(cufftComplex));
}

void matrix_t::alloc_device() 
{
    CUDA_SAFE_CALL( 
	  cudaMalloc((void**)&device, elems * sizeof(cufftComplex))
    );
}

void matrix_t::copy_to_device() 
{
    CUDA_SAFE_CALL(
    cudaMemcpy(device, host, elems * sizeof(cufftComplex), cudaMemcpyHostToDevice)
	);
}

void matrix_t::copy_to_host()
{
    CUDA_SAFE_CALL(
    cudaMemcpy(host, device, elems * sizeof(cufftComplex), cudaMemcpyDeviceToHost)
	);
}

void matrix_t::free_device()
{
    if (device) {
      cudaFree(device);
      device = NULL;
    }
}

void matrix_t::assign_device(matrix_t* m)
{
    isComplex = m->isComplex;
    cudaMemcpy(device, m->device, elems * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
}

void matrix_t::debug_print(char* name, int first, int last)
{
    for (int x = first; x < last; x++)
      printf("%s[%d] = %g + %gi\n", name, x, host[x]REAL, host[x]IMAG);
}

sparse_matrix_t::sparse_matrix_t() : numRows(0), numCols(0), maxNumNonZeroElems(0), numNonZeroElems(0), 
                      jc(NULL), ir(NULL), rVals(NULL), iVals(NULL), 
                      devJc(NULL), devIr(NULL), devRVals(NULL), 
                      isComplex(0)
{}


sparse_matrix_t::~sparse_matrix_t()
{ 
    if (jc) free(jc);
    if (ir) free(ir);
    if (rVals) free(rVals);
    if (iVals) free(iVals);
    if (devJc) cudaFree(devJc);
    if (devIr) cudaFree(devIr);
    if (devRVals) cudaFree(devRVals);
}

void sparse_matrix_t::alloc_device() 
{
    cudaMalloc((void**)&devJc, (numCols+1) * sizeof(int));
    cudaMalloc((void**)&devIr, numNonZeroElems * sizeof(int));
    cudaMalloc((void**)&devRVals, numNonZeroElems * sizeof(float));
}

void sparse_matrix_t::free_device()
{
    if (devJc) { cudaFree(devJc); devJc = NULL; }
    if (devIr) { cudaFree(devIr); devIr = NULL; }
    if (devRVals) { cudaFree(devRVals); devRVals = NULL; }
}

void sparse_matrix_t::copy_to_device() 
{
    cudaMemcpy(devJc, jc, (numCols+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devIr, ir, numNonZeroElems * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devRVals, rVals, numNonZeroElems * sizeof(float), cudaMemcpyHostToDevice);
}

void sparse_matrix_t::init_JGAI(int nRows, int nCols, int maxNonZeroElems, int nNonZeroElems)
{
    numRows = nRows;
	numCols = nCols;
	maxNumNonZeroElems = maxNonZeroElems;
	numNonZeroElems = nNonZeroElems;
	jc = (int *) malloc((numCols+1)*sizeof(int));
	ir = (int *) malloc((numNonZeroElems*sizeof(int)));
	rVals = (float *) malloc(numNonZeroElems*sizeof(float));
	iVals = (float *) malloc(numNonZeroElems*sizeof(float));
}

// length(ir) = numNonZeroElems
// length(jc) = numCols+1
// length(rVals) = numNonZeroElems
void sparse_matrix_t::write(char fName[]) {
    FILE* fid = fopen(fName, "w");
    fprintf(fid, "%d %d %d %d\n", numRows, numCols, numNonZeroElems, isComplex);
    fwrite(ir, sizeof(int), numNonZeroElems, fid);
    fwrite(jc, sizeof(int), numCols+1, fid);
    fwrite(rVals, sizeof(float), numNonZeroElems, fid);
    if (isComplex)
      fwrite(iVals, sizeof(float), numNonZeroElems, fid);
    fclose(fid);
}

void sparse_matrix_t::read(char fName[]) {
    FILE* fid = fopen(fName, "r");

    int a,b,c,d;
    if ( EOF==fscanf(fid, "%d %d %d %d\n", &a, &b, &c, &d) )
    {
        printf("Error: sparse_matrix_t::read\n");
        exit(1);
    }
    numRows = a;
    numCols = b;
    numNonZeroElems = c;
    isComplex = d;

    printf("rows %d cols %d nz %d cmplx %d\n", numRows, numCols, numNonZeroElems, isComplex);

    ir = (int*)calloc(numNonZeroElems, sizeof(int));
    jc = (int*)calloc(numCols + 1, sizeof(int));
    rVals = (float*)calloc(numNonZeroElems, sizeof(float));
    if (isComplex)
      iVals = (float*)calloc(numNonZeroElems, sizeof(float));

    if ( numNonZeroElems!=fread(ir, sizeof(int), numNonZeroElems, fid) )
    {
        printf("Error: sparse_matrix_t::read\n");
        exit(1);
    }
    if ( (numCols+1)!=fread(jc, sizeof(int), numCols+1, fid) )
	{
        printf("Error: sparse_matrix_t::read\n");
        exit(1);
    }
    if ( numNonZeroElems!=fread(rVals, sizeof(float), numNonZeroElems, fid) )
	{
        printf("Error: sparse_matrix_t::read\n");
        exit(1);
    }
    if (isComplex)
    {
        if ( numNonZeroElems!=fread(iVals, sizeof(float), numNonZeroElems, fid) )
        {
            printf("Error: sparse_matrix_t::read\n");
            exit(1);
        }
	}
    fclose(fid);

    assert(numNonZeroElems == jc[numCols]);
    maxNumNonZeroElems = 0;
}

// Returns 1 if result is complex. Returns 0 otherwise.
// lhs = this * m
int sparse_matrix_t::mult(matrix_t* lhs, matrix_t *m)
{
    int col = 0;
    int total = 0;
    int resultComplex = 1;

    if (isComplex) {
      if (m->isComplex) {
        for (col = 0; col < numCols; col++) {
          int starting_row_index = jc[col];
          int stopping_row_index = jc[col+1];
          if (starting_row_index != stopping_row_index) {
            for (int current_row_index = starting_row_index; current_row_index < stopping_row_index; current_row_index++) {
              lhs->host[ir[current_row_index]]REAL += rVals[total] * m->host[col]REAL - iVals[total] * m->host[col]IMAG;
              lhs->host[ir[current_row_index]]IMAG += iVals[total] * m->host[col]REAL + rVals[total] * m->host[col]IMAG;
              total++;
            }
          }
        }
      }
      else {
        for (col = 0; col < numCols; col++) {
          int starting_row_index = jc[col];
          int stopping_row_index = jc[col+1];
          if (starting_row_index != stopping_row_index) {
            for (int current_row_index = starting_row_index; current_row_index < stopping_row_index; current_row_index++) {
              lhs->host[ir[current_row_index]]REAL += rVals[total] * m->host[col]REAL;
              lhs->host[ir[current_row_index]]IMAG += iVals[total] * m->host[col]REAL;
              total++;
            }
          }
        }
      }
    }
    else {
      if (m->isComplex) {
        for (col = 0; col < numCols; col++) {
          int starting_row_index = jc[col];
          int stopping_row_index = jc[col+1];
          if (starting_row_index != stopping_row_index) {
            for (int current_row_index = starting_row_index; current_row_index < stopping_row_index; current_row_index++) {
              lhs->host[ir[current_row_index]]REAL += rVals[total] * m->host[col]REAL;
              lhs->host[ir[current_row_index]]IMAG += rVals[total] * m->host[col]IMAG;
              total++;
            }
          }
        }
      }
      else {
        resultComplex = 0;
        for (col = 0; col < numCols; col++) {
          int starting_row_index = jc[col];
          int stopping_row_index = jc[col+1];
          if (starting_row_index != stopping_row_index) {
            for (int current_row_index = starting_row_index; current_row_index < stopping_row_index; current_row_index++) {
              lhs->host[ir[current_row_index]]REAL += rVals[total] * m->host[col]REAL;
              total++;
            }
          }
        }
      }
    }
    return resultComplex;
}

// Returns 1 if result is complex. Returns 0 otherwise.
// lhs = this' * m
int sparse_matrix_t::mult_trans(matrix_t* lhs, matrix_t *m)
{
    int col = 0;
    int resultComplex = 1;

    // Element of sparse matrix is in row = ir[current_row_index], column = col.
    // Thus, transpose position of element is in row = col, column = ir[current_row_index].
    if (isComplex) {
      if (m->isComplex) {
        for (col = 0; col < numCols; col++) {
          int starting_row_index = jc[col];
          int stopping_row_index = jc[col+1];
          if (starting_row_index != stopping_row_index) {
            for (int current_row_index = starting_row_index; current_row_index < stopping_row_index; current_row_index++) {
              lhs->host[col]REAL += rVals[current_row_index] * m->host[ir[current_row_index]]REAL - iVals[current_row_index] * m->host[ir[current_row_index]]IMAG;
              lhs->host[col]IMAG += iVals[current_row_index] * m->host[ir[current_row_index]]REAL + rVals[current_row_index] * m->host[ir[current_row_index]]IMAG;
            }
          }
        }
      }
      else {
        for (col = 0; col < numCols; col++) {
          int starting_row_index = jc[col];
          int stopping_row_index = jc[col+1];
          if (starting_row_index != stopping_row_index) {
            for (int current_row_index = starting_row_index; current_row_index < stopping_row_index; current_row_index++) {
              lhs->host[col]REAL += rVals[current_row_index] * m->host[ir[current_row_index]]REAL;
              lhs->host[col]IMAG += iVals[current_row_index] * m->host[ir[current_row_index]]REAL;
            }
          }
        }
      }
    }
    else {
      if (m->isComplex) {
        for (col = 0; col < numCols; col++) {
          int starting_row_index = jc[col];
          int stopping_row_index = jc[col+1];
          if (starting_row_index != stopping_row_index) {
            for (int current_row_index = starting_row_index; current_row_index < stopping_row_index; current_row_index++) {
              lhs->host[col]REAL += rVals[current_row_index] * m->host[ir[current_row_index]]REAL;
              lhs->host[col]IMAG += rVals[current_row_index] * m->host[ir[current_row_index]]IMAG;
            }
          }
        }
      }
      else {
        resultComplex = 0;
        for (col = 0; col < numCols; col++) {
          int starting_row_index = jc[col];
          int stopping_row_index = jc[col+1];
          if (starting_row_index != stopping_row_index) {
            for (int current_row_index = starting_row_index; current_row_index < stopping_row_index; current_row_index++) {
              lhs->host[col]REAL += rVals[current_row_index] * m->host[ir[current_row_index]]REAL;
            }
          }
        }
      }
    }
    return resultComplex;
}

float* readDataFile_JGAI(const char* file_name, float& version, int& data_size_per_coil, 
                       int& ncoils, int& nslices, int& sensed_data_size)
{
	FILE* fp = fopen(file_name,"r");
	if (fp == NULL) {
		printf("Error: Cannot open file %s\n", file_name);
		exit(1);
	}

	if ( 1!=fread(&version,sizeof(float),1,fp) )
    {
       printf("Error reading header variable No.1.\n");
       exit(1);
    }

	if ( 1!=fread(&data_size_per_coil,sizeof(int),1,fp) )
    {
       printf("Error reading header variable No.2.\n");
       exit(1);
    }

    if ( 1!=fread(&ncoils,sizeof(int),1,fp) )
    {
       printf("Error reading header variable No.3.\n");
       exit(1);
    }

	if ( 1!=fread(&nslices, sizeof(int), 1, fp) )
    {
       printf("Error reading header variable No.4.\n");
       exit(1);
    }

	if ( 1!=fread(&sensed_data_size, sizeof(int), 1, fp) )
    {
       printf("Error reading header variable No.5.\n");
       exit(1);
    }

	float* data_pointer = (float *) malloc(sensed_data_size * sizeof(float));
    if ( sensed_data_size!=fread(data_pointer, sizeof(float), sensed_data_size, fp) )
    {
       printf("Error reading data:%s at line %d in %s\n",file_name,__LINE__,__FILE__);
       exit(1);
    }
	fclose(fp);
	
	return data_pointer;
}

float* readDataFile_JGAI_10(const char* file_name, float& version, int& data_size_per_coil, 
                            int& ncoils, int& nslices, int& sensed_data_size)
{
   const char impatient_keywords[9][200] = { "version", "xDimension", 
   "yDimension", "zDimension", "coil_number", "slice_number", "file_size", 
   "Binary_Size","Binary:"};

   int xDimension = -1, yDimension = -1, zDimension = -1;
   int coil_number = -1, slice_number = -1;
   int file_size = -1;
   int Binary_Size = -1;

   float *data_pointer = NULL;

   //To count how many data points are read from each binary section.
   //combined_section_size has to equal file_size, otherwise you are
   //in trouble!
   int combined_section_size = 0;

   FILE *fp = fopen(file_name, "r");
   if(fp == NULL) {
	  printf("Error: Cannot open file %s for writing\n", file_name);
	  exit(1);
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
          printf("Data format error: symbol should be followed by an equal sign!\n");
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
		  //printf("xDimension = %d\n", xDimension);
       }
	   else if(0==strcmp(symbol_temp,impatient_keywords[2]))
       {
          yDimension = atoi( value_temp );
		  //printf("yDimension = %d\n", yDimension);
       }
	   else if(0==strcmp(symbol_temp,impatient_keywords[3]))
       {
          zDimension = atoi( value_temp );
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
		     printf("coil_number unspecified in file %s.\n",file_name);
			 //exit(1);
             #endif
		  }
		  data_size_per_coil = file_size / coil_number;
		  
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
		       continue;
            while( '\n' != buf[symbol_end] )
            {
               symbol_end++;
            }
            symbol_end--;//retract the end point to the previous location.

	        //ignore any comments
	        if ('/'==buf[symbol_start] && '/'==buf[symbol_start+1])
		       continue;


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
			   //printf("Binary: time to read binary data from %s.\n",file_name);
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

   return data_pointer;
}

void writeDataFile_JGAI( const char *file_name, float version, int data_size_per_coil, int ncoils, int nslices, int sensed_data_size, float *data_pointer )
{
    FILE *fp = fopen(file_name, "wb");
	if (fp == NULL) {
	        printf("Error: Cannot open file %s for writing\n", file_name);
	        exit(1);
    }
    fwrite(&version, sizeof(float), 1, fp);
    fwrite(&data_size_per_coil, sizeof(int), 1, fp);
    fwrite(&ncoils, sizeof(int), 1, fp);
    fwrite(&nslices, sizeof(int), 1, fp);
    fwrite(&sensed_data_size, sizeof(int), 1, fp);
    fwrite(data_pointer, sizeof(float), sensed_data_size, fp);
    fclose(fp);
}

  void 
writeDataFile_JGAI_10(const char *file_name, float version, int xDimension,
		              int yDimension, int zDimension, int coil_number, int slice_number,
					  int file_size, float *data_pointer)
{

	FILE *fp = fopen(file_name, "w");
	if(fp==NULL) {
       printf("Error: Cannot open file %s for writing\n", file_name);
	   exit(1);
	}
	fprintf(fp,"version = %f\n", version);
	fprintf(fp,"xDimension = %d\n", xDimension);
	fprintf(fp,"yDimension = %d\n", yDimension);
	fprintf(fp,"zDimension = %d\n", zDimension);

	if(coil_number!=-1) {
        // coil_number is relevant to this file
		fprintf(fp,"coil_number = %d\n", coil_number);
	}
	else {
		// if coil_number is irrelevant to this file, we can safely
		// pretend that coil_number = 1.
        coil_number = 1;
        #if DEBUG_MODE
		printf("coil_number unspecified in this file.\n");
		#endif
	}
	fprintf(fp, "slice_number = %d\n", slice_number);
	fprintf(fp, "file_size = %d\n", file_size);

	int file_size_per_coil = file_size / coil_number;
	int fwrite_returned = -1;//check fwrite's return value.
	for(int i=0;i<coil_number;i++)
	{
	   fprintf(fp,"//Coil No. %d\n",i);
	   fprintf(fp,"Binary_Size = %d\n",file_size_per_coil);
	   fprintf(fp,"Binary:\n");

	   fwrite_returned = fwrite(data_pointer+i*file_size_per_coil,
			                    sizeof(float),file_size_per_coil,fp);
	   if(file_size_per_coil!=fwrite_returned)
	   {
	      printf("Error: The number of elements successfully written is wrong!");
		  exit(1);
	   }
	}

	fprintf(fp,"EOF\n");
	fclose(fp);
}

void print_device_mem(float *p_d, int len, int start, int end, int interval)
{
    float *p = (float *) calloc(len, sizeof(float));
    CUDA_SAFE_CALL(cudaMemcpy(p, p_d, len * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = start; i <= end; i += interval)
        printf("[%d]=%f, ", i, p[i]);
    printf("\n");

    free(p);
}

void setupMemoryGPU(int num, int size, float *&dev_ptr, float *&host_ptr) {
    CUDA_SAFE_CALL(cudaMalloc((void **) &dev_ptr, num * size));
    CUT_CHECK_ERROR("cudaMalloc failed");
    CUDA_SAFE_CALL(cudaMemcpy(dev_ptr, host_ptr, num * size, cudaMemcpyHostToDevice));
    CUT_CHECK_ERROR("cudaMemcpy failed");
}

  void
measureMemoryUsageGPU(void)
{
    // show memory usage of GPU
    size_t free_byte ;
    size_t total_byte ;
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
    if ( cudaSuccess != cuda_status ){
         printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
         exit(1);
    }

    float free_db = (float)free_byte ;
    float total_db = (float)total_byte ;
    float used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", 
			used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}
