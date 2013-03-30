/*
(C) Copyright 2010 The Board of Trustees of the University of Illinois.
All rights reserved.

Developed by:

                         IMPACT Research Groups
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

Neither the names of the IMPACT Research Group, the University of Illinois,
nor the names of its contributors may be used to endorse or promote products
derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
THE SOFTWARE.
*/

/*****************************************************************************

    File Name   [xcpplib_typesGpu.cuh]

    Synopsis    [Helper functions on types used in CUDA files.]

    Description []

    Revision    [0.1; Initial build; Xiao-Long Wu, ECE UIUC]
    Date        [04/04/2010]

 *****************************************************************************/

#ifndef XCPPLIB_TYPESGPU_CUH
#define XCPPLIB_TYPESGPU_CUH

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>

// CUDA libraries
#include <cuda.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>

// XCPPLIB libraries
#include <xcpplib_global.h>
#include <xcpplib_process.h>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

namespace xcpplib {

/*---------------------------------------------------------------------------*/
/*  Macro implementations                                                    */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*  Data structure implementations and template function implementations     */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Allocate/Deallocate a GPU array in a good-looking way.]     */
/*                                                                           */
/*  Description [The returned array should be freed by the caller using the  */
/*      corresponding delete function.]                                      */
/*                                                                           */
/*===========================================================================*/

    template <class T> T *
newCuda(const int dim_x, const bool if_clean = true) // Size of X dimension
{
    ensure(dim_x > 0, XLIB_USR_2_1("X"));

    T * var_1d = NULL;
    #if 1
    cudaError e = cudaMalloc((void **)&var_1d, dim_x * sizeof(T));
    makeSure(e == cudaSuccess, "CUDA runtime failure on cudaMalloc.");
    #else
    cutilSafeCall(cudaMalloc((void **)&var_1d, dim_x * sizeof(T)));
    #endif

    ensure(var_1d, XLIB_DEV_2_2("an 1-D array"));
    if (if_clean) {
        cudaError e = cudaMemset(var_1d, 0, dim_x * sizeof(T));
        makeSure(e == cudaSuccess, "CUDA runtime failure on cudaMemset.");
    }

    return var_1d;
}

    template <class T>
    inline void
clearCuda(T * var_1d, const int dim_x) // Size of X dimension
{
    ensure(dim_x > 0, XLIB_USR_2_1("X"));
    ensure(var_1d, XLIB_DEV_2_2("an 1-D array"));
    cudaError e = cudaMemset(var_1d, 0, dim_x * sizeof(T));
    makeSure(e == cudaSuccess, "CUDA runtime failure on cudaMemset.");
}

#if 0
    #define deleteCuda(var_1d)                     \
        ensure(var_1d, XLIB_USR_1_3("var_1d"));    \
        cutilSafeCall(cudaFree(var_1d));

#else
    template <class T>
    inline void
deleteCuda(T * var_1d)
{
    ensure(var_1d, XLIB_USR_1_3("var_1d"));
    #if 1
    cudaError e = cudaFree(var_1d);
    makeSure(e == cudaSuccess, "CUDA runtime failure on cudaFree.");
    #else
    cutilSafeCall(cudaFree(var_1d));
    #endif
}
#endif

    template <class T> void
copyCudaHostToDevice(T * dst, const T * src, const int dim_x) {
    ensure(dst, XLIB_USR_1_3("dst"));
    ensure(src, XLIB_USR_1_3("src"));
    ensure(dim_x > 0, XLIB_USR_2_1("X"));
    #if 1
    cudaError e = cudaMemcpy(dst, src, dim_x * sizeof(T),
                             cudaMemcpyHostToDevice);
    makeSure(e == cudaSuccess,
        "CUDA runtime failure on cudaMemcpyHostToDevice.");
    #else
    cutilSafeCall(cudaMemcpy(dst, src, dim_x * sizeof(T),
                             cudaMemcpyHostToDevice));
    #endif
}

    template <class T> void
copyCudaDeviceToHost(T * dst, const T * src, const int dim_x) {
    ensure(dst, XLIB_USR_1_3("dst"));
    ensure(src, XLIB_USR_1_3("src"));
    ensure(dim_x > 0, XLIB_USR_2_1("X"));

    cudaThreadSynchronize();
    #if 1
    cudaError e = cudaMemcpy(dst, src, dim_x * sizeof(T),
                             cudaMemcpyDeviceToHost);
    makeSure(e == cudaSuccess,
        "CUDA runtime failure on cudaMemcpyDeviceToHost.");
    #else
    cutilSafeCall(cudaMemcpy(dst, src, dim_x * sizeof(T),
                             cudaMemcpyDeviceToHost));
    #endif
}

    template <class T> void
copyCudaDeviceToDevice(T * dst, const T * src, const int dim_x) {
    ensure(dst, XLIB_USR_1_3("dst"));
    ensure(src, XLIB_USR_1_3("src"));
    ensure(dim_x > 0, XLIB_USR_2_1("X"));
    cudaThreadSynchronize();

    #if 1
    cudaError e = cudaMemcpy(dst, src, dim_x * sizeof(T),
                             cudaMemcpyDeviceToDevice);
    makeSure(e == cudaSuccess,
        "CUDA runtime failure on cudaMemcpyDeviceToDevice.");
    #else
    cutilSafeCall(cudaMemcpy(dst, src, dim_x * sizeof(T),
                             cudaMemcpyDeviceToDevice));
    #endif
}

/*---------------------------------------------------------------------------*/
/*  Class GpuDataType Begin                                                  */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [A wrapper data structure for manipulating CUDA device data  */
/*      types.]                                                              */
/*                                                                           */
/*  Description [By this wrapper, programmers don't need to explicitly define*/
/*      the data copy commands for manipulating CUDA device data.]           */
/*                                                                           */
/*===========================================================================*/

    template <class T>
class GpuDataType
{
public:
    GpuDataType(void);

    GpuDataType(unsigned int s, bool if_clean = true);

    GpuDataType(const T *a, unsigned int s, bool if_clean = true);

    ~GpuDataType();

        T *
    getData(unsigned int s);

        T *
    getData(void);

        T *
    getDataPtr(void);

        void
    putData(const T *a, unsigned int s, bool if_clean = true);

        void
    freeData(void);

        unsigned int
    getSize(void);

        unsigned int
    getSizeMem(void);

private:
    T *array;               // data pointer to the device memory
    unsigned int size;      // number of data elements
    unsigned int size_mem;  // size of data elements in bytes
};

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Class constructor]                                          */
/*                                                                           */
/*  Description [Allocate an empty object.]                                  */
/*                                                                           */
/*===========================================================================*/

    template <class T>
    inline
GpuDataType<T>::GpuDataType(void)
{
    array = NULL;
    size = size_mem = 0;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Class constructor]                                          */
/*                                                                           */
/*  Description [Allocate a kernel object with given data size and type.]    */
/*                                                                           */
/*===========================================================================*/

    template <class T>
    inline
GpuDataType<T>::GpuDataType(
    unsigned int s,     // number of data elements
    bool if_clean)      // if clean the allocated memory
{
    makeSure(s > 0, "Data size must be greater than zero.", __FILE__, __LINE__);
    array = newCuda<T>(s, if_clean);
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Class constructor]                                          */
/*                                                                           */
/*  Description [Allocate a kernel object with given data, size, and data    */
/*      type. Data is copied to the device side.]                            */
/*                                                                           */
/*===========================================================================*/

    template <class T>
    inline
GpuDataType<T>::GpuDataType(
    const T * a,        // data pointer to the host memory
    unsigned int s,     // number of data elements
    bool if_clean)      // if clean the allocated memory
                        // Default is true.
{
    makeSure(s > 0, "Data size must be greater than zero.", __FILE__, __LINE__);
    array = newCuda<T>(s, if_clean);
    copyCudaHostToDevice(array, a, s);
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Class destructor]                                           */
/*                                                                           */
/*  Description [Deallocate a kernel object if it's not empty.]              */
/*                                                                           */
/*===========================================================================*/

    template <class T>
    inline
GpuDataType<T>::~GpuDataType()
{
    if (size > 0) {
        cutilSafeCall(cudaFree(array));
        array = NULL;
        size = size_mem = 0;
    }
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Get data.]                                                  */
/*                                                                           */
/*  Description [Get back the kernel object.]                                */
/*                                                                           */
/*  Note        [Host data must be freed by the caller.]                     */
/*                                                                           */
/*===========================================================================*/

    template <class T>
    inline T *
GpuDataType<T>::getData(
    unsigned int s)     // number of data elements
{
    makeSure(s > 0, "Data size must be greater than zero.", __FILE__, __LINE__);
    unsigned int s_mem = s * sizeof(T);
    T *a = new T[s_mem];
    cutilSafeCall(cudaMemcpy(a, array, s_mem, cudaMemcpyDeviceToHost));
    return a;
}

    template <class T>
    inline T *
GpuDataType<T>::getData(void)
{
    return getData(size);
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Get the pointer of data.]                                   */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    template <class T>
    inline T *
GpuDataType<T>::getDataPtr(void)
{
    return array;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Put data]                                                   */
/*                                                                           */
/*  Description [Copy the given data to the kernel.]                         */
/*                                                                           */
/*===========================================================================*/

    template <class T>
    inline void
GpuDataType<T>::putData(
    const T * a,        // data pointer to the host memory
    unsigned int s,     // number of data elements
    bool if_clean)      // if clean the allocated memory before copying
                        // Default is true.
{
    makeSure(s > 0, "Data size must be greater than zero.", __FILE__, __LINE__);
    array = newCuda<T>(s, if_clean);
    copyCudaHostToDevice(array, a, s);
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Free data.]                                                 */
/*                                                                           */
/*  Description [Release the device memory.]                                 */
/*                                                                           */
/*===========================================================================*/

    template <class T>
    inline void
GpuDataType<T>::freeData(void)
{
    if (size > 0) {
        cutilSafeCall(cudaFree(array));
        array = NULL;
        size = size_mem = 0;
    }
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Get number of data elements.]                               */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    template <class T>
    inline unsigned int
GpuDataType<T>::getSize(void)
{
    return size;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Get size of data elements in bytes.]                        */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    template <class T>
    inline unsigned int
GpuDataType<T>::getSizeMem(void)
{
    return size_mem;
}

/*---------------------------------------------------------------------------*/
/*  Class GpuDataType End                                                    */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*  Function prototypes                                                      */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

}

#endif // XCPPLIB_TYPESGPU_CUH

