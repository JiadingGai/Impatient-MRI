/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

//
//  Utility funcs to wrap up saving a surface or the back buffer as a PPM file
//	In addition, wraps up a threshold comparision of two PPMs.
//
//	These functions are designed to be used to implement an automated QA testing for SDK samples.
//
//	Author: Bryan Dudash
//  Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include <cutil.h>
#include <rendercheck_d3d10.h>

HRESULT CheckRenderD3D10::ActiveRenderTargetToPPM(ID3D10Device *pDevice, const char *zFileName)
{
	ID3D10RenderTargetView *pRTV = NULL;
	pDevice->OMGetRenderTargets(1,&pRTV,NULL);

	ID3D10Resource *pSourceResource = NULL;
	pRTV->GetResource(&pSourceResource);

	return ResourceToPPM(pDevice,pSourceResource,zFileName);
}

HRESULT CheckRenderD3D10::ResourceToPPM(ID3D10Device*pDevice, ID3D10Resource *pResource, const char *zFileName)
{
	D3D10_RESOURCE_DIMENSION rType;
	pResource->GetType(&rType);

	if(rType != D3D10_RESOURCE_DIMENSION_TEXTURE2D)
	{
		printf("SurfaceToPPM: pResource is not a 2D texture! Aborting...\n");
		return E_FAIL;
	}

	ID3D10Texture2D * pSourceTexture = (ID3D10Texture2D *)pResource;
	ID3D10Texture2D * pTargetTexture = NULL;

	D3D10_TEXTURE2D_DESC desc;
	pSourceTexture->GetDesc(&desc);
	desc.BindFlags = 0;
	desc.CPUAccessFlags = D3D10_CPU_ACCESS_READ;
	desc.Usage = D3D10_USAGE_STAGING;
		
	if(FAILED(pDevice->CreateTexture2D(&desc,NULL,&pTargetTexture)))
	{
		printf("SurfaceToPPM: Unable to create target Texture resoruce! Aborting... \n");
		return E_FAIL;
	}

	pDevice->CopyResource(pTargetTexture,pSourceTexture);

	D3D10_MAPPED_TEXTURE2D mappedTex2D;
	pTargetTexture->Map(0,D3D10_MAP_READ,0,&mappedTex2D);
	
	// Need to convert from dx pitch to pitch=width
	unsigned char *pPPMData = new unsigned char[desc.Width*desc.Height*4];
	for(unsigned int iHeight = 0;iHeight<desc.Height;iHeight++)
	{
		memcpy(&(pPPMData[iHeight*desc.Width*4]),(unsigned char*)(mappedTex2D.pData)+iHeight*mappedTex2D.RowPitch,desc.Width*4);
	}

	pTargetTexture->Unmap(0);

	// Prepends the PPM header info and bumps byte data afterwards
	cutSavePPM4ub(zFileName, pPPMData, desc.Width, desc.Height);

	delete [] pPPMData;
	pTargetTexture->Release();

	return S_OK;
}

bool CheckRenderD3D10::PPMvsPPM( const char *src_file, const char *ref_file, const char *exec_path, 
                                 const float epsilon, const float threshold )
{
    char *ref_file_path = cutFindFilePath(ref_file, exec_path);
    if (ref_file_path == NULL) {
        printf("CheckRenderD3D10::PPMvsPPM unable to find <%s> in <%s> Aborting comparison!\n", ref_file, exec_path);
        printf(">>> Check info.xml and [project//data] folder <%s> <<<\n", ref_file);
        printf("Aborting comparison!\n");
        printf("  FAILED!\n");
        return false;
    }

    return cutComparePPM(src_file,ref_file_path,epsilon,threshold,true) == CUTTrue;
}