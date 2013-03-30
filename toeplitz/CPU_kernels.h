#include "stdio.h"
#include "UDTypes.h"
#include <cufft.h>

   void 
calculateLUT(float beta, float width, float*& LUT, unsigned int& sizeLUT);
   int 
gridding_Gold_2D(unsigned int n, parameters params, ReconstructionSample* sample, 
		      float* LUT, unsigned int sizeLUT, 
			  float *t, float l, float tau, 
			  cufftComplex* gridData, float* sampleDensity);

   int 
gridding_CPU_2D(unsigned int n, parameters params, ReconstructionSample* sample, 
		     int* CPUbin, int CPUbin_size, float* LUT, int sizeLUT, 
			 float *t, float l, float tau, 
			 cufftComplex* gridData[], float* sampleDensity[], int* indeces[]);


   int 
gridding_Gold_3D(unsigned int n, parameters params, ReconstructionSample* sample, 
		      float* LUT, unsigned int sizeLUT, 
			  float *t, float l, float tau, 
			  cufftComplex* gridData, float* sampleDensity);

   int 
gridding_CPU_3D(unsigned int n, parameters params, ReconstructionSample* sample, 
		     int* CPUbin, int CPUbin_size, float* LUT, int sizeLUT, 
			 float *t, float l, float tau, 
			 cufftComplex* gridData[], float* sampleDensity[], int* indeces[]);

#if 0
void gridding_GPU_CPU (parameters params, samplePtArray sampleArray, int* start, cmplx* gridData, float* sampleDensity, float beta);

int gridding_GPU(unsigned int n, parameters params, samplePtArray sample, float* LUT, int sizeLUT, cmplx* gridData, float* sampleDensity);
#endif
