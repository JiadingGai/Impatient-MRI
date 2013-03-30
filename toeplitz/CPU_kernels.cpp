#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cstring>

#include "CPU_kernels.h"
#include "UDTypes.h"

#define max(x,y) ((x<y)?y:x)
#define min(x,y) ((x>y)?y:x)

#define PI 3.14159265359

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

///*From Numerical Recipes in C, 2nd Edition
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
// */

void calculateLUT(float beta, float width, float*& LUT, unsigned int& sizeLUT){
  float v;
  float _width2_4 = (width*width)/4.0;

  if(width > 0){
    // compute size of LUT based on kernel width
    sizeLUT = (unsigned int)(10000*width);

    // allocate memory
    LUT = (float*) malloc (sizeLUT*sizeof(float));

    for(unsigned int k=0; k<sizeLUT; ++k){
      // compute value to evaluate kernel at
      // v in the range 0:(_width/2)^2
      v = (float(k)/float(sizeLUT))*_width2_4;

      // compute kernel value and store
      LUT[k] = bessi0(beta*sqrt(1.0-(v/_width2_4)));
    }
  }
}

float kernel_value_LUT(float v, float* LUT, int sizeLUT, float _1_width2_4)
{
  unsigned int k0;
  float v0;

  v *= (float)sizeLUT;
  k0=(unsigned int)(v*_1_width2_4);
  v0 = ((float)k0)/_1_width2_4;
  return  LUT[k0] + ((v-v0)*(LUT[k0+1]-LUT[k0])/_1_width2_4);
}

// 2D gold gridding on CPU
   int 
gridding_Gold_2D(unsigned int n, parameters params, ReconstructionSample* sample, 
	      float* LUT, unsigned int sizeLUT, 
              float *t, float l, float tau,
	      cufftComplex* gridData, float* sampleDensity)
{
  unsigned int NxL, NxH;
  unsigned int NyL, NyH;
  //unsigned int NzL, NzH;

  unsigned int nx;
  unsigned int ny;
  //unsigned int nz;

  int idx;

  float w;

  float shiftedKx, shiftedKy/*, shiftedKz*/;
  float distX, kbX, distY, kbY/*, distZ, kbZ*/;

  float kernelWidth = params.kernelWidth;
  float beta = 18.5547;
  float gridOS = params.gridOS;

  unsigned int Nx = params.imageSize[0];
  unsigned int Ny = params.imageSize[1];
  //unsigned int Nz = params.imageSize[2];

  //Jiading GAI
  float t0 = t[0];

  for (unsigned int i=0; i < n; i++)
  {
    ReconstructionSample pt = sample[i];

	//Jiading GAI
    float atm = hanning_d(t[i], tau, l, t0);//a_l(t_m)

    shiftedKx = (gridOS)*(pt.kX+((float)Nx)/2.0f);
    shiftedKy = (gridOS)*(pt.kY+((float)Ny)/2.0f);
    //shiftedKz = ((float)gridOS)*(pt.kZ+((float)Nz)/2);

	//if(shiftedKx < 0.0f)
	//   shiftedKx = 0.0f;
	//if(shiftedKx > ((float)gridOS)*((float)Nx));
	//   shiftedKx = ((float)gridOS)*((float)Nx);
	//if(shiftedKy < 0.0f)
	//   shiftedKy = 0.0f;
	//if(shiftedKy > ((float)gridOS)*((float)Ny));
	//   shiftedKy = ((float)gridOS)*((float)Ny);


    NxL = (int)(fmax(0.0f,ceil(shiftedKx - kernelWidth*(gridOS)/2.0f)));
    NxH = (int)(fmin((gridOS*(float)Nx-1.0f),floor(shiftedKx + kernelWidth*(gridOS)/2.0f)));

    NyL = (int)(fmax(0.0f,ceil(shiftedKy - kernelWidth*(gridOS)/2.0f)));
    NyH = (int)(fmin((gridOS*(float)Ny-1.0f),floor(shiftedKy + kernelWidth*(gridOS)/2.0f)));

    //NzL = (int)(fmax(0.0f,ceil(shiftedKz - kernelWidth*((float)gridOS)/2)));
    //NzH = (int)(fmin((float)(gridOS*Nz-1),floor(shiftedKz + kernelWidth*((float)gridOS)/2)));

    for(ny=NyL; ny<=NyH; ++ny)
    {
       distY = fabs(shiftedKy - ((float)ny))/(gridOS);
       kbY = bessi0(beta*sqrt(1.0-(2.0*distY/kernelWidth)*(2.0*distY/kernelWidth)))/kernelWidth;
       if (kbY!=kbY)//if kbY = NaN
           kbY=0;
 
       for(nx=NxL; nx<=NxH; ++nx)
       {
          distX = fabs(shiftedKx - ((float)nx))/(gridOS);
          kbX = bessi0(beta*sqrt(1.0-(2.0*distX/kernelWidth)*(2.0*distX/kernelWidth)))/kernelWidth;
          if (kbX!=kbX)//if kbX = NaN
              kbX=0;
 
           /* kernel weighting value */
           if (params.useLUT){
              w = kbX * kbY;
           } else {
              w = kbX * kbY;
           }
           /* grid data */
           idx = nx + (ny)*params.gridSize[0]/* + (nz)*gridOS*Nx*gridOS*Ny*/;
           gridData[idx].x += (w*pt.real*atm);
           gridData[idx].y += (w*pt.imag*atm);

           /* estimate sample density */
           sampleDensity[idx] += w;
       }
    }   
  }

  // re-arrange dimensions and output
  // Nady uses: x->y->z
  // IMPATIENT uses: z->x->y
  // So we need to convert from (x->y->z)-order to (z->x->y)-order
  int gridNumElems = params.gridSize[0] * params.gridSize[1];
  cufftComplex *gridData_reorder = (cufftComplex*) calloc(gridNumElems, sizeof(cufftComplex));

  for(int x=0;x<params.gridSize[0];x++)
  for(int y=0;y<params.gridSize[1];y++)
  {
    int lindex_nady      = x + y*params.gridSize[0];
    int lindex_impatient = y + x*params.gridSize[0];
   
    gridData_reorder[lindex_impatient] = gridData[lindex_nady];
  }
  memcpy((void*)gridData,(void*)gridData_reorder,gridNumElems*sizeof(cufftComplex));

  free(gridData_reorder);

  return 1;
}


// 3D gold gridding on CPU
   int 
gridding_Gold_3D(unsigned int n, parameters params, ReconstructionSample* sample, 
		      float* LUT, unsigned int sizeLUT, 
			  float *t, float l, float tau,
			  cufftComplex* gridData, float* sampleDensity)
{
  unsigned int NxL, NxH;
  unsigned int NyL, NyH;
  unsigned int NzL, NzH;

  unsigned int nx;
  unsigned int ny;
  unsigned int nz;

  int idx;

  float w;

  float shiftedKx, shiftedKy, shiftedKz;
  float distX, kbX, distY, kbY, distZ, kbZ;

  float kernelWidth = params.kernelWidth;
  float beta = 18.5547;
  float gridOS = params.gridOS;

  unsigned int Nx = params.imageSize[0];
  unsigned int Ny = params.imageSize[1];
  unsigned int Nz = params.imageSize[2];

  //Jiading GAI
  float t0 = t[0];

  for (unsigned int i=0; i < n; i++)
  {
    ReconstructionSample pt = sample[i];

	//Jiading GAI
    float atm = hanning_d(t[i], tau, l, t0);//a_l(t_m)

	shiftedKx = (gridOS)*(pt.kX+((float)Nx)/2.0f);
    shiftedKy = (gridOS)*(pt.kY+((float)Ny)/2.0f);
    shiftedKz = (gridOS)*(pt.kZ+((float)Nz)/2.0f);

//	if(shiftedKx < 0.0f)
//	   shiftedKx = 0.0f;
//	if(shiftedKx > ((float)gridOS)*((float)Nx));
//	   shiftedKx = ((float)gridOS)*((float)Nx);
//	if(shiftedKy < 0.0f)
//	   shiftedKy = 0.0f;
//	if(shiftedKy > ((float)gridOS)*((float)Ny));
//	   shiftedKy = ((float)gridOS)*((float)Ny);
//	if(shiftedKz < 0.0f)
//	   shiftedKz = 0.0f;
//	if(shiftedKz > ((float)gridOS)*((float)Nz));
//	   shiftedKz = ((float)gridOS)*((float)Nz);


    NxL = (int)(fmax(0.0f,ceil(shiftedKx - kernelWidth*(gridOS)/2.0f)));
    NxH = (int)(fmin((gridOS*(float)Nx-1.0f),floor(shiftedKx + kernelWidth*((float)gridOS)/2.0f)));

	NyL = (int)(fmax(0.0f,ceil(shiftedKy - kernelWidth*(gridOS)/2.0f)));
    NyH = (int)(fmin((gridOS*(float)Ny-1.0f),floor(shiftedKy + kernelWidth*((float)gridOS)/2.0f)));

	NzL = (int)(fmax(0.0f,ceil(shiftedKz - kernelWidth*(gridOS)/2.0f)));
    NzH = (int)(fmin((gridOS*(float)Nz-1.0f),floor(shiftedKz + kernelWidth*((float)gridOS)/2.0f)));

    for(nz=NzL; nz<=NzH; ++nz)
	{
       distZ = fabs(shiftedKz - ((float)nz))/(gridOS);
       kbZ = bessi0(beta*sqrt(1.0-(2.0*distZ/kernelWidth)*(2.0*distZ/kernelWidth)))/kernelWidth;
       if (kbZ!=kbZ)//if kbZ = NaN
           kbZ=0;
 
	   for(ny=NyL; ny<=NyH; ++ny)
       {
          distY = fabs(shiftedKy - ((float)ny))/(gridOS);
          kbY = bessi0(beta*sqrt(1.0-(2.0*distY/kernelWidth)*(2.0*distY/kernelWidth)))/kernelWidth;
          if (kbY!=kbY)//if kbY = NaN
                kbY=0;
 
		  for(nx=NxL; nx<=NxH; ++nx)
          {
             distX = fabs(shiftedKx - ((float)nx))/(gridOS);
             kbX = bessi0(beta*sqrt(1.0-(2.0*distX/kernelWidth)*(2.0*distX/kernelWidth)))/kernelWidth;
             if (kbX!=kbX)//if kbX = NaN
                kbX=0;
 
             /* kernel weighting value */
             if (params.useLUT){
               w = kbX * kbY * kbZ;
             } else {
               w = kbX * kbY * kbZ;
             }
             /* grid data */
			 idx = nx + (ny)*params.gridSize[0] + (nz)*params.gridSize[0]*params.gridSize[1];
             gridData[idx].x += (w*pt.real*atm);
             gridData[idx].y += (w*pt.imag*atm);

             /* estimate sample density */
             sampleDensity[idx] += w;
           }
        }
     }
  }

  // re-arrange dimensions and output
  // Nady uses: x->y->z
  // IMPATIENT uses: z->x->y
  // So we need to convert from (x->y->z)-order to (z->x->y)-order
  int gridNumElems = params.gridSize[0] * params.gridSize[1] * params.gridSize[2];
  cufftComplex *gridData_reorder = (cufftComplex*) calloc(gridNumElems, sizeof(cufftComplex));

  for(int x=0;x<params.gridSize[0];x++)
  for(int y=0;y<params.gridSize[1];y++)
  for(int z=0;z<params.gridSize[2];z++)
  {
    int lindex_nady = x + y*params.gridSize[0] + z*params.gridSize[0]*params.gridSize[1];
    int lindex_impatient = z + x*params.gridSize[2] + y*params.gridSize[0]*params.gridSize[2];
   
    gridData_reorder[lindex_impatient] = gridData[lindex_nady];
  }
  memcpy((void*)gridData,(void*)gridData_reorder,gridNumElems*sizeof(cufftComplex));

  free(gridData_reorder);

  return 1;
}

// 2D gridding on CPU
   int 
gridding_CPU_2D(unsigned int n, parameters params, ReconstructionSample* sample, 
		     int* CPUbin, int CPUbin_size, float* LUT, int sizeLUT, 
			 float *t, float l, float tau,
			 cufftComplex* gridData[], float* sampleDensity[], int* indeces[])
{
  unsigned int NxL, NxH;
  unsigned int NyL, NyH;
  //unsigned int NzL, NzH;

  unsigned int nx;
  unsigned int ny;
  //unsigned int nz;

  int idx;

  float w;

  float shiftedKx, shiftedKy/*, shiftedKz*/;
  float distX, kbX, distY, kbY/*, distZ, kbZ*/;

  float kernelWidth = params.kernelWidth;
  float beta = 18.5547;
  float gridOS = params.gridOS;

  unsigned int Nx = params.imageSize[0];
  unsigned int Ny = params.imageSize[1];
  //unsigned int Nz = params.imageSize[2];

  int gridNumElems = params.gridSize[0]*params.gridSize[1]/**params.gridSize[2]*/;

  //Jiading GAI
  float t0 = t[0];
  
  int pos = 0;
  int* binAlloc = (int*) malloc (gridNumElems*sizeof(int));
  memset(binAlloc, 0xFF, gridNumElems*sizeof(int));
  (*indeces) = (int*) malloc (gridNumElems*sizeof(int));
  (*gridData) = (cufftComplex*) calloc (gridNumElems,sizeof(cufftComplex));
  (*sampleDensity) = (float*) calloc (gridNumElems,sizeof(float));

  if (*gridData == NULL || *sampleDensity == NULL || *indeces == NULL){
    printf("unable to allocate temporary CPU space\n");
    exit(1);
  }

  for (int i=0; i < CPUbin_size; i++)
  {
    ReconstructionSample pt = sample[CPUbin[i]];

	//Jiading GAI
    float atm = hanning_d(t[i], tau, l, t0);//a_l(t_m)
	
    shiftedKx = (gridOS)*(pt.kX+((float)Nx)/2.0f);
    shiftedKy = (gridOS)*(pt.kY+((float)Ny)/2.0f);
    //shiftedKz = ((float)gridOS)*(pt.kZ+((float)Nz)/2);

//	if(shiftedKx < 0.0f)
//	   shiftedKx = 0.0f;
//	if(shiftedKx > ((float)gridOS)*((float)Nx));
//	   shiftedKx = ((float)gridOS)*((float)Nx);
//	if(shiftedKy < 0.0f)
//	   shiftedKy = 0.0f;
//	if(shiftedKy > ((float)gridOS)*((float)Ny));
//	   shiftedKy = ((float)gridOS)*((float)Ny);


    NxL = (int)(fmax(0.0f,ceil(shiftedKx - kernelWidth*(gridOS)/2.0f)));
    NxH = (int)(fmin((gridOS*(float)Nx-1.0f),floor(shiftedKx + kernelWidth*(gridOS)/2.0f)));

    NyL = (int)(fmax(0.0f,ceil(shiftedKy - kernelWidth*(gridOS)/2.0f)));
    NyH = (int)(fmin((gridOS*(float)Ny-1.0f),floor(shiftedKy + kernelWidth*(gridOS)/2.0f)));

    //NzL = (int)(fmax(0.0f,ceil(shiftedKz - kernelWidth*((float)gridOS)/2)));
    //NzH = (int)(fmin((float)(gridOS*Nz-1),floor(shiftedKz + kernelWidth*((float)gridOS)/2)));
    
    for(ny=NyL; ny<=NyH; ++ny)
    {
        distY = fabs(shiftedKy - ((float)ny))/(gridOS);
        kbY = bessi0(beta*sqrt(1.0-(2.0*distY/kernelWidth)*(2.0*distY/kernelWidth)))/kernelWidth;
        if(kbY!=kbY)//if kbY = NaN
           kbY=0;

        for(nx=NxL; nx<=NxH; ++nx)
        {
          distX = fabs(shiftedKx - ((float)nx))/(gridOS);
          kbX = bessi0(beta*sqrt(1.0-(2.0*distX/kernelWidth)*(2.0*distX/kernelWidth)))/kernelWidth;
          if(kbX!=kbX)//if kbX = NaN
             kbX=0;

          /* kernel weighting value */
          if (params.useLUT){
              w = kbX * kbY;
          } else {
              w = kbX * kbY;
          }

          /* grid data */		  
	      idx = nx + (ny)*params.gridSize[0];
          if(binAlloc[idx] == -1){
             binAlloc[idx] = pos;
             (*indeces)[pos] = idx;
             pos++;
          }

          (*gridData)[binAlloc[idx]].x += (w*pt.real*atm);
          (*gridData)[binAlloc[idx]].y += (w*pt.imag*atm);

          /* estimate sample density */
          (*sampleDensity)[binAlloc[idx]] += w;
             
        }
    }
  }

  free(binAlloc);
  return pos;
}


// 3D gridding on CPU
   int 
gridding_CPU_3D(unsigned int n, parameters params, ReconstructionSample* sample, 
		     int* CPUbin, int CPUbin_size, float* LUT, int sizeLUT, 
			 float *t, float l, float tau,
			 cufftComplex* gridData[], float* sampleDensity[], int* indeces[])
{
  unsigned int NxL, NxH;
  unsigned int NyL, NyH;
  unsigned int NzL, NzH;

  unsigned int nx;
  unsigned int ny;
  unsigned int nz;

  int idx;

  float w;

  float shiftedKx, shiftedKy, shiftedKz;
  float distX, kbX, distY, kbY, distZ, kbZ;

  float kernelWidth = params.kernelWidth;
  float beta = 18.5547;
  float gridOS = params.gridOS;

  unsigned int Nx = params.imageSize[0];
  unsigned int Ny = params.imageSize[1];
  unsigned int Nz = params.imageSize[2];

  int gridNumElems = params.gridSize[0]*params.gridSize[1]*params.gridSize[2];

  //Jiading GAI
  float t0 = t[0];
  
  int pos = 0;
  int* binAlloc = (int*) malloc (gridNumElems*sizeof(int));
  memset(binAlloc, 0xFF, gridNumElems*sizeof(int));
  (*indeces) = (int*) malloc (gridNumElems*sizeof(int));
  (*gridData) = (cufftComplex*) calloc (gridNumElems,sizeof(cufftComplex));
  (*sampleDensity) = (float*) calloc (gridNumElems,sizeof(float));

  if (*gridData == NULL || *sampleDensity == NULL || *indeces == NULL){
    printf("unable to allocate temporary CPU space\n");
    exit(1);
  }

  for (int i=0; i < CPUbin_size; i++)
  {
    ReconstructionSample pt = sample[CPUbin[i]];

	//Jiading GAI
    float atm = hanning_d(t[i], tau, l, t0);//a_l(t_m)
	
	shiftedKx = (gridOS)*(pt.kX+((float)Nx)/2.0f);
    shiftedKy = (gridOS)*(pt.kY+((float)Ny)/2.0f);
    shiftedKz = (gridOS)*(pt.kZ+((float)Nz)/2.0f);

//	if(shiftedKx < 0.0f)
//	   shiftedKx = 0.0f;
//	if(shiftedKx > ((float)gridOS)*((float)Nx));
//	   shiftedKx = ((float)gridOS)*((float)Nx);
//	if(shiftedKy < 0.0f)
//	   shiftedKy = 0.0f;
//	if(shiftedKy > ((float)gridOS)*((float)Ny));
//	   shiftedKy = ((float)gridOS)*((float)Ny);
//	if(shiftedKz < 0.0f)
//	   shiftedKz = 0.0f;
//	if(shiftedKz > ((float)gridOS)*((float)Nz));
//	   shiftedKz = ((float)gridOS)*((float)Nz);


    NxL = (int)(fmax(0.0f,ceil(shiftedKx - kernelWidth*(gridOS)/2.0f)));
    NxH = (int)(fmin((gridOS*(float)Nx-1.0f),floor(shiftedKx + kernelWidth*(gridOS)/2.0f)));

	NyL = (int)(fmax(0.0f,ceil(shiftedKy - kernelWidth*(gridOS)/2.0f)));
    NyH = (int)(fmin((gridOS*(float)Ny-1.0f),floor(shiftedKy + kernelWidth*(gridOS)/2.0f)));

	NzL = (int)(fmax(0.0f,ceil(shiftedKz - kernelWidth*(gridOS)/2.0f)));
    NzH = (int)(fmin((gridOS*(float)Nz-1.0f),floor(shiftedKz + kernelWidth*(gridOS)/2.0f)));
    
    for(nz=NzL; nz<=NzH; ++nz)
    {
       distZ = fabs(shiftedKz - ((float)nz))/(gridOS);
       kbZ = bessi0(beta*sqrt(1.0-(2.0*distZ/kernelWidth)*(2.0*distZ/kernelWidth)))/kernelWidth;
       if(kbZ!=kbZ)//if kbZ = NaN
          kbZ=0;

	   for(ny=NyL; ny<=NyH; ++ny)
       {
          distY = fabs(shiftedKy - ((float)ny))/(gridOS);
          kbY = bessi0(beta*sqrt(1.0-(2.0*distY/kernelWidth)*(2.0*distY/kernelWidth)))/kernelWidth;
          if(kbY!=kbY)//if kbY = NaN
             kbY=0;

		  for(nx=NxL; nx<=NxH; ++nx)
          {
            distX = fabs(shiftedKx - ((float)nx))/(gridOS);
            kbX = bessi0(beta*sqrt(1.0-(2.0*distX/kernelWidth)*(2.0*distX/kernelWidth)))/kernelWidth;
            if(kbX!=kbX)//if kbX = NaN
                kbX=0;

             /* kernel weighting value */
             if (params.useLUT){
               w = kbX * kbY * kbZ;
             } else {
               w = kbX * kbY * kbZ;
             }

             /* grid data */
			 idx = nx + (ny)*params.gridSize[0] + (nz)*params.gridSize[0]*params.gridSize[1];
             if(binAlloc[idx] == -1){
                binAlloc[idx] = pos;
                (*indeces)[pos] = idx;
                pos++;
             }

             (*gridData)[binAlloc[idx]].x += (w*pt.real*atm);
             (*gridData)[binAlloc[idx]].y += (w*pt.imag*atm);

             /* estimate sample density */
             (*sampleDensity)[binAlloc[idx]] += w;
             
          }
       }
    }
}

  free(binAlloc);
  return pos;
}
