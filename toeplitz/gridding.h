#ifndef TOEPLITZ_GRIDDING_CUH
#define TOEPLITZ_GRIDDING_CUH
#include "UDTypes.h"
void binning_kernel1_CPU(unsigned int n, ReconstructionSample *Sample_h,
                         unsigned int *numPts_h, parameters params);

void scanLargeArray_CPU(int len, unsigned int *input);

void binning_kernel2_CPU(unsigned int n, ReconstructionSample *Sample, 
		                 unsigned int *binStartAddr, unsigned int *numPts, 
						 parameters params, samplePtArray SampleArray,
                         int& CPUbinSize, int *CPUbin);

#endif
