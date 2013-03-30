#ifndef _UDTYPES_H__
#define _UDTYPES_H__
#include <vector_types.h>// required by float2

typedef struct{
  int numSamples;
  int imageSize[3];
  int gridSize[3];
  float gridOS;
  float kernelWidth;
  int binsize;
  int useLUT;
  int sync;
}parameters;

typedef struct{
  float real;
  float imag;
  float kX;
  float kY;
  float kZ;
  float sdc;
  float t;
  float dummy;
} ReconstructionSample;

typedef struct{
  float2* data;
  float2* loc1;
  float2* loc2;
  float2* loc3;
} samplePtArray;
#endif
