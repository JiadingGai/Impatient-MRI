#include <gridding.h>
#include <math.h>
#include <utils.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// t0 is there b/c t.dat does not start with 0.0f.
static    __host__ __device__ 
float hanning_d(float tm, float tau, float l, float t0)
{
    float taul = tau * l;
    float result;
    if ( fabs(tm - taul - t0) < tau ) {
        result = 0.5f + 0.5f * cosf(PI * (tm - taul - t0) / tau);
    } else {
        result = 0.0f;
    }
    //FIXME:
    //result = 1.0f;
    return result;
}

///* Jiading GAI - GRIDDING - BEGIN
/*From Numerical Recipes in C, 2nd Edition*/
static float bessi0(float x)
{
    float ax,ans;
    float y;
    
    if ((ax=fabs(x)) < 3.75)
    {
        y=x/3.75;
        y=y*y;
        ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492+y*(0.2659732+
            y*(0.360768e-1+y*0.45813e-2)))));
    }
    else
    {
        y=3.75/ax;
        ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1+y*(0.225319e-2+
             y*(-0.157565e-2+y*(0.916281e-2+y*(-0.2057706e-1+y*(0.2635537e-1+
             y*(-0.1647633e-1+y*0.392377e-2))))))));
    }
    return ans;
}

	void 
binning_kernel1_CPU(unsigned int n, ReconstructionSample *Sample_h,
                    unsigned int *numPts_h, parameters params)
{
  unsigned int gridNumElems = params.gridSize[0] * 
                              params.gridSize[1] * 
                              params.gridSize[2] ;
  unsigned int binsize = params.binsize;
  float gridOS = params.gridOS;
  unsigned int Nxy_grid = params.gridSize[0] * 
                          params.gridSize[1] ;

  for(unsigned int i=0;i<n;i++)
  {

     ReconstructionSample pt;
     unsigned int binIdx;

     pt = Sample_h[i];
     pt.kX = (gridOS)*(pt.kX+((float)params.imageSize[0])/2.0f);
     pt.kY = (gridOS)*(pt.kY+((float)params.imageSize[1])/2.0f);
	 if(1==params.imageSize[2])
     {
	   pt.kZ = 0.0f;
	 }
	 else
     {
       pt.kZ = (gridOS)*(pt.kZ+((float)params.imageSize[2])/2.0f);
	 }

	 /*
	 // Clamp k trajectories between [0,gridSize]
	 // B/c not all data given to me are normalized 
	 // between [-imageSize/2,imageSize/2].
	 if( pt.kZ < 0.0f )
		 pt.kZ = 0.0f;
	 if( pt.kZ > (float)params.gridSize[2] )
		 pt.kZ = (float)params.gridSize[2];

	 if( pt.kX < 0.0f )
		 pt.kX = 0.0f;
	 if( pt.kX > (float)params.gridSize[0] )
		 pt.kX = (float)params.gridSize[0];

	 if( pt.kY < 0.0f )
		 pt.kY = 0.0f;
	 if( pt.kY > (float)params.gridSize[1] )
		 pt.kY = (float)params.gridSize[1];
     // */

     binIdx = (unsigned int)(pt.kZ)*Nxy_grid + 
              (unsigned int)(pt.kY)*params.gridSize[0] + 
              (unsigned int)(pt.kX);

     if(numPts_h[binIdx]<binsize)
     {
        numPts_h[binIdx] += 1;
     }
  }  
}

  void
scanLargeArray_CPU(int len, unsigned int *input)
{
  unsigned int *output = (unsigned int*) malloc(len*sizeof(unsigned int));
  output[0] = 0;

  for(int i=1;i<len;i++)
    output[i] = output[i-1] + input[i-1];

  memcpy(input, output, len*sizeof(unsigned int));
  free(output);
}

   void
binning_kernel2_CPU(unsigned int n, ReconstructionSample *Sample, unsigned int *binStartAddr,
                    unsigned int *numPts, parameters params, samplePtArray SampleArray,
                    int& CPUbinSize, int *CPUbin)
{
   unsigned int gridNumElems = params.gridSize[0] * 
                               params.gridSize[1] * 
                               params.gridSize[2] ;
   unsigned int binsize = params.binsize;
   float gridOS = params.gridOS;
   unsigned int Nxy_grid = params.gridSize[0] * 
                           params.gridSize[1] ;

   for(unsigned int i=0;i<n;i++)
   {
      ReconstructionSample pt;
      unsigned int binIdx;
      unsigned int cap;

      pt = Sample[i];
      pt.kX = (gridOS)*(pt.kX+((float)params.imageSize[0])/2.0f);
      pt.kY = (gridOS)*(pt.kY+((float)params.imageSize[1])/2.0f);
	  if(1==params.imageSize[2])
	  {
        pt.kZ = 0.0f;
	  }
	  else
	  {
        pt.kZ = (gridOS)*(pt.kZ+((float)params.imageSize[2])/2.0f);
	  }

	  /*
	  // Clamp k trajectories between [0,gridSize].
	  // B/c not all input data given to me are normalized 
	  // between [-imageSize/2,imageSize/2].
      if( pt.kZ < 0.0f )
	      pt.kZ = 0.0f;
      if( pt.kZ > (float)params.gridSize[2] )
	      pt.kZ = (float)params.gridSize[2];

      if( pt.kX < 0.0f )
	      pt.kX = 0.0f;
      if( pt.kX > (float)params.gridSize[0] )
	      pt.kX = (float)params.gridSize[0];

	 if( pt.kY < 0.0f )
		 pt.kY = 0.0f;
	 if( pt.kY > (float)params.gridSize[1] )
		 pt.kY = (float)params.gridSize[1];
	 // */

      binIdx = (unsigned int)(pt.kZ)*Nxy_grid + 
               (unsigned int)(pt.kY)*params.gridSize[0] + 
               (unsigned int)(pt.kX);

      if(numPts[binIdx]<binsize)
      {
         cap = numPts[binIdx];
         numPts[binIdx] += 1;
         
         float2 data;
         data.x = pt.real;
         data.y = pt.imag;
 
         float2 loc1;
         loc1.x = pt.kX;
         loc1.y = pt.kY;
 
         float2 loc2;
         loc2.x = pt.kZ;
         loc2.y = pt.sdc;

         float2 loc3;
		 loc3.x = pt.t;
		 //loc3.y = pt.dummy;

         SampleArray.data[binStartAddr[binIdx]+cap] = data;
         SampleArray.loc1[binStartAddr[binIdx]+cap] = loc1;
         SampleArray.loc2[binStartAddr[binIdx]+cap] = loc2;
         SampleArray.loc3[binStartAddr[binIdx]+cap] = loc3;
      }
      else
      {
         cap = CPUbinSize;
         CPUbinSize += 1;
         CPUbin[cap] = i;
      }
   }  
}

// Jiading GAI - GRIDDING - END */
