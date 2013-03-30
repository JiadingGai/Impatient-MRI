#ifndef CUDA_INTERFACE_H
#define CUDA_INTERFACE_H
#include <UDTypes.h>
   void 
CUDA_interface (unsigned int n, parameters params, ReconstructionSample* sample, 
		        float* LUT, int sizeLUT, 
                float *t, float *t_d, float l, float tau, float beta,
				cufftComplex* gridData, float* sampleDensity);
#endif
