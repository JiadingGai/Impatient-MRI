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

    File Name   [xcpplib_types.h]

    Synopsis    [Arrays, stacks, maps, and other types and containers.]

    Description [Helper functions for manipulating arrays, stacks, maps, and
        other types and containers.]

    Revision    [0.1; Initial build; Xiao-Long Wu, ECE UIUC]
    Date        [04/04/2010]

 *****************************************************************************/

#ifndef XCPPLIB_TYPES_H
#define XCPPLIB_TYPES_H

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>
#include <deque>
#include <exception>
#include <string>
#include <sstream>      // for stringstream class
#include <map>
#include <cstdlib>      // for rand()
#include <cstring>

// XCPPLIB libraries
#include <xcpplib_global.h>
#include <xcpplib_process.h>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

namespace xcpplib {

/*---------------------------------------------------------------------------*/
/*  Namespace used                                                           */
/*---------------------------------------------------------------------------*/

using namespace std;

/*---------------------------------------------------------------------------*/
/*  Macro implementations                                                    */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Convert any values of any basic types into string type.]    */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

// toString() is the function to call for converting.
#if 0
#define xcpplibToStringPrototype(TYPE)                                       \
        string                                                               \
    toString(const TYPE &a);

xcpplibApplyBuiltInType(xcpplibToStringPrototype);

    string
toString(const bool &a);

#else

#define xcpplibToString(TYPE)                                                \
        inline string                                                        \
    toString(const TYPE &a)                                                  \
    {                                                                        \
        stringstream s;                                                      \
        s<< a;                                                               \
        return s.str();                                                      \
    }

xcpplibApplyBuiltInType(xcpplibToString);

    inline string
toString(const bool &a)
{
    string s;
    if (a) s = "T";
    else s = "F";
    return s;
}
#endif

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Change a macro into a string.]                              */
/*                                                                           */
/*  Description [STRINGIFY() is what we actually use.]                       */
/*                                                                           */
/*  Note        [It applies to macros only. Not variables.]                  */
/*                                                                           */
/*===========================================================================*/

#define TOSTRING(X)  #X
#define STRINGIFY(X) TOSTRING(X)

    string
stringIResize(const string &str, const size_t n, char c);

    bool
checkEdgeOverlap(
    unsigned int &found_digits,   // Overlapping digits found
    const unsigned int &start_digits,   // Starting digits
    const string &read1, const string &read2,
    // Compare digits with a certain stride to reduce time.
    // This may produce some false positive results.
    const unsigned int check_stride = 1);

    bool
checkEdgeOverlapFix(
    const unsigned int &overlap_digits,   // How many overlapping digits
    const string &read1, const string &read2,
    // Compare digits with a certain stride to reduce time.
    // This may produce some false positive results.
    const unsigned int check_stride = 1);

// ============================================================================
// The following string converting functions seem obsolete.
// ============================================================================

// Convert int type to string type.
    inline string
int2String(const int &i)
{
    string s;
    stringstream ss(s);
    ss<< i;
    return ss.str();
}

// Convert float type to string type.
    inline string
float2String(const float &f)
{
    string s;
    stringstream ss(s);
    ss<< f;
    return ss.str();
}

// Convert double type to string type.
    inline string
double2String(const double &d)
{
    string s;
    stringstream ss(s);
    ss<< d;
    return ss.str();
}

// Stringify types. For example, stringifyType(int) returns "int" string. 
#define stringifyType(T) stringifyType_((T) 1)
#define stringifyTypeMacro(TYPE) \
        inline string \
    stringifyType_(const TYPE &one) { if (one) return TOSTRING(TYPE); \
                                      else return TOSTRING(TYPE); }
xcpplibApplyBuiltInType(stringifyTypeMacro);

/*---------------------------------------------------------------------------*/
/*  Data structure implementations                                           */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [A memory trace class for logging the memory allocation/     */
/*      deallocation process.]                                               */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*  Note        [For multiple dimension aggregate variables, the memory      */
/*      records to be freed can be different variables pointing to the same  */
/*      memory address.]                                                     */
/*                                                                           */
/*===========================================================================*/

// FIXME: Should provide a method to query the memory size from a given memory
//        address.

class MemoryTrace
{
// This has to be multimap because multiple threads will allocate the same
// memory addresses.
typedef multimap<void *, unsigned int> MemoryTraceMap;

private:
    string name;                // name of this memory trace task
    unsigned int memory_usage;  // memory usage in bytes
    unsigned int max_memory_usage; // the max memory usage so far.
    MemoryTraceMap trace;       // memory trace database

    bool if_verbose;            // If show messages during the process

public:
    // Constructor/Destructors ===============================================

    MemoryTrace(const string &name_, const bool if_verbose_ = false)
    : name(name_), memory_usage(0), max_memory_usage(0),
      if_verbose(if_verbose_)
    { }

    ~MemoryTrace(void) {
        if (if_verbose) {
            if (max_memory_usage < 1000000) {
                cout<< "MemoryTrace: "<< name<< ": Peak memory usage is "
                    << max_memory_usage<< " bytes"<< endl;
            } else {
                cout<< "MemoryTrace: "<< name<< ": Peak memory usage is "<<
                    (float) max_memory_usage/1000000.0<<" Mega bytes"<< endl;
            }
        }
        if (if_verbose && memory_usage != 0) {
            outputMsg("MemoryTrace: Memory is not freed totally", false);
            MemoryTraceMap::iterator itr = trace.begin();
            makeSure(itr != trace.end(), "Should have items left.");
            cout<< "MemoryTrace: "<< name<< ": Last item address: "
                << (*itr).first<< endl;
            cout<< "MemoryTrace: "<< name<< ": Last item size: "
                << (*itr).second<< " bytes"<< endl;
        } else if (if_verbose && memory_usage == 0) {
            cout<< "MemoryTrace: "<< name<< ": All memory is released."<< endl;
        }
    }

    // Member functions ======================================================

        void
    insert(void *addr, const unsigned int usage)
    {
        memory_usage += usage;
        if (memory_usage > max_memory_usage) {
            max_memory_usage = memory_usage;
        }

        trace.insert(make_pair(addr, usage));

        if (if_verbose) {
            cout<< "MemoryTrace: "<< name<< ": insert: "<< addr
                << " ("<< trace.size()<< " in database)"<< endl;
        }
    }

        void
    erase(void *addr)
    {
        MemoryTraceMap::iterator itr = trace.find(addr);
        if (itr == trace.end()) {
            outputWarn("Can't find the memory address.");
            message("Make sure you use the right function to allocate memory.");
        }
        memory_usage -= (*itr).second;
        trace.erase(itr);
        if (if_verbose) {
            cout<< "MemoryTrace: "<< name<< ": erase: "<< addr
                << " ("<< trace.size()<< " in database)"<< endl;
        }
    }

        inline unsigned int
    getUsage(void)
    {
        return memory_usage;
    }
}; // End of class MemoryTrace

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [A specialized fast stack.]                                  */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    template <class T>
class Stack
{
protected:
    #if 0
    std::deque<T> c; // Container for the elements
    #else // Faster than using the deque container.
    std::vector<T> c; // Container for the elements
    #endif

public:
    Stack(void) {
        //c.reserve(100); // This is not working for vector
    }

    // Reset the stack
        inline void
    clear(void) {
        c.clear();
    }
    
    // Number of elements
        inline typename std::vector<T>::size_type
    size(void) const {
        return c.size();
    }

    // Is stack empty?
        inline bool
    empty(void) const {
        return c.empty();
    }

    // Push element into the stack
        inline void
    push(const T &ele) {
        c.push_back(ele);
    }

    // Pop element out of the stack and return its value
        inline T
    pop(void) {
        if (c.empty()) { // throw exception
            throw readEmptyStack();
        }
        T ele(c.back());
        c.pop_back();
        return ele;
    }

    // Return value of the top element to be popped.
        inline const T &
    top(void) const {
        if (c.empty()) { // throw exception
            throw readEmptyStack();
        }
        return c.back();
    }

    // Exception class for pop() and top() with empty stack
    class readEmptyStack : public std::exception {
    public:
            virtual const char * 
        what() const throw() {
            return XLIB_USR_1_4;
        }
    };
}; // End of class Stack

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [A specialized fast stack with fixed size.]                  */
/*                                                                           */
/*  Description [For some reason, this one is slower than Stack by 0.02 sec.]*/
/*                                                                           */
/*===========================================================================*/
const unsigned int StackFix_c_allocate_max = 100;

    template <class T>
class StackFix
{
protected:
    T *c;
    unsigned int c_size;
    unsigned int c_allocate;

public:
    StackFix(void) {
        c_allocate = StackFix_c_allocate_max;
        c = new T[c_allocate];
        c_size = 0;
    }

    StackFix(const unsigned int &s) {
        c_allocate = s;
        c = new T[c_allocate];
        c_size = 0;
    }

    ~StackFix(void) {
        if (!this->empty()) {
            delete [] c;
            c_size = 0;
        }
    }

    // Reset the stack
        inline void
    clear(void) {
        if (!this->empty()) {
            delete [] c;
            c = new T[c_allocate];
            c_size = 0;
        }
    }
    
    // Number of elements
        inline unsigned int
    size(void) const {
        return c_size;
    }

    // Is stack empty?
        inline bool
    empty(void) const {
        return (c_size == 0);
    }

    // Is stack full?
        inline bool
    full(void) const {
        return (c_size >= c_allocate);
    }

    // Push element into the stack
        inline void
    push(const T &ele) {
        if (this->full()) { // throw exception
            throw pushFullStack();
        }
        c[c_size] = ele;
        c_size++;
    }

    // Pop element out of the stack and return its value
        inline T
    pop(void) {
        if (this->empty()) { // throw exception
            throw readEmptyStack();
        }
        T ele(this->back());
        c_size--;
        return ele;
    }

        inline const T &
    back(void) const {
        if (this->empty()) { // throw exception
            throw readEmptyStack();
        }
        return c[c_size-1];
    }

    // Return value of the top element to be popped.
        inline const T &
    top(void) const {
        if (this->empty()) { // throw exception
            throw readEmptyStack();
        }
        return this->back();
    }

    // Exception class for pop() and top() with empty stack
    class readEmptyStack : public std::exception {
    public:
            virtual const char * 
        what() const throw() {
            return XLIB_USR_1_4;
        }
    };

    // Exception class for stack overflow
    class pushFullStack : public std::exception {
    public:
            virtual const char * 
        what() const throw() {
            return XLIB_USR_1_5;
        }
    };
}; // End of class StackFix

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [A class of T type array.]                                   */
/*                                                                           */
/*  Description [We use this class to facilitate the coding without explicit */
/*      allocation and deallocation of the memory.]                          */
/*                                                                           */
/*  Todo list   [Provide an interface such that we don't need to worry about */
/*      the tedious data copy between cpu and gpu.]                          */
/*                                                                           */
/*===========================================================================*/

    template <class T>
class TArray
{
public:                     // The data members are intended to be public.
    T *array;
    unsigned int size;      // number of data elements
    unsigned int size_mem;  // size of data elements in bytes

public:
    // Constructors/Destructors
    // ========================

    TArray(void) { array = NULL; size = 0; size_mem = 0; };
    TArray(const unsigned int s) {
        allocate(s); size_mem = s * sizeof(T);
    };

    ~TArray() { freeData(); };

    // Copy constructor and copy assignment operator
    // =============================================

    TArray(const TArray &rhs) {
        *this = rhs; // call the assignment operator
    }

        TArray &
    operator = (const TArray &rhs) {
        if (this != &rhs) {
            array = new T[rhs.size];
            size = rhs.size;
            size_mem = rhs.size_mem;
            memcpy(array, rhs.array, rhs.size_mem);
        }
        return *this;
    }

    // Private members/data
    // ====================

        inline void
    allocate(const unsigned int s) {
        size = s;
        size_mem = s * sizeof(T);
        array = new T[size];
    }

        T *
    getDataPtr(void) const;

        void
    putData(const T *a, unsigned int s, bool if_clean = true);

        void
    freeData(void);

        unsigned int
    getSize(void) const;

        unsigned int
    getSizeMem(void) const;
};

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Free data.]                                                 */
/*                                                                           */
/*  Description [Release the memory.]                                        */
/*                                                                           */
/*===========================================================================*/

    template <class T>
    inline void
TArray<T>::freeData(void)
{
    if (size > 0) {
        delete [] array; array = NULL;
        size = size_mem = 0;
    }
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Get the pointer of data.]                                   */
/*                                                                           */
/*  Description [The returned pointer should not be modified. If it's        */
/*      modified, the data member "size" and "size_mem" should be updated.]  */
/*                                                                           */
/*===========================================================================*/

    template <class T>
    inline T *
TArray<T>::getDataPtr(void) const
{
    return array;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Put data]                                                   */
/*                                                                           */
/*  Description [Copy the given data to the allocated memory.]               */
/*                                                                           */
/*===========================================================================*/

    template <class T>
    inline void
TArray<T>::putData(
    const T * a,        // data pointer to the source data
    unsigned int s,     // number of data elements
    bool if_clean)      // if clean the allocated memory before copying
                        // Default is true.
{
    makeSure(s > 0, "Data size must be greater than zero.", __FILE__, __LINE__);
    allocate(s);
    if (if_clean) memset(array, 0, size_mem);
    memcpy(array, a, size_mem);
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
TArray<T>::getSize(void) const
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
TArray<T>::getSizeMem(void) const
{
    return size_mem;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Allocate/Deallocate a multi-dimension array in efficient    */
/*      ways. When the other array pointer is provided, the contents of the  */
/*      given array are copied to the new one.]                              */
/*                                                                           */
/*  Description [The returned array should be freed by the caller using the  */
/*      corresponding delete function.]                                      */
/*                                                                           */
/*===========================================================================*/

    template <class T> T *
newArray1D(const int dim_x, // Size of X dimension
    const bool if_manual_reset = true, // If manually reset all contents.
                               // This is useful when type T has its own reset.
    T * var_1d2 = NULL) // If given, a copy of var_1d2 is applied.
{
    ensure(dim_x > 0, XLIB_USR_2_1("X"));

    T * var_1d;
    var_1d = new T[dim_x];
    ensure(var_1d, XLIB_DEV_2_2("an 1-D array"));
    if (if_manual_reset) { // All array elements are set to 0.
        memset(var_1d, 0, dim_x * sizeof(T));
    }

    if (var_1d2 != NULL) {
        memcpy(var_1d, var_1d2, dim_x * sizeof(T));
    }
    return var_1d;
}

#if 0
#define deleteArray1D(var_1d)               \
    ensure(var_1d, XLIB_USR_1_3("var_1d")); \
    delete [] var_1d;
#else
    template <class T> void
deleteArray1D(T * var_1d)
{
    ensure(var_1d, XLIB_USR_1_3("var_1d"));
    delete [] var_1d;
}
#endif

    template <class T>
    inline void
copyArray1D(T * dst, const T * src, const int dim_x) {
    ensure(dst, XLIB_USR_1_3("dst"));
    ensure(src, XLIB_USR_1_3("src"));
    ensure(dim_x > 0, XLIB_USR_2_1("X"));
    memcpy(dst, src, dim_x * sizeof(T));
}

    template <class T> T **
newArray2D(
    const int dim_x, const int dim_y, // Sizes of X and Y dimensions
    T ** var_2d2 = NULL) // If given, a copy of var_2d2 is applied.
{
    ensure(dim_x > 0, XLIB_USR_2_1("X"));
    ensure(dim_y > 0, XLIB_USR_2_1("Y"));

    T ** var_2d;
    var_2d = new T* [dim_y];
    ensure(var_2d, XLIB_DEV_2_2("a 2-D array"));

    var_2d[0] = new T [dim_y * dim_x];
    ensure(var_2d[0], XLIB_DEV_2_2("a 2-D array"));
    // All array elements are set to 0.
    memset(var_2d[0], 0, dim_y * dim_x * sizeof(T));
    for (int i = 1; i < dim_y; i++) { var_2d[i] = var_2d[i-1] + dim_x; }

    if (var_2d2 != NULL) {
        memcpy(var_2d[0], var_2d2[0], dim_y * dim_x * sizeof(T));
    }
    return var_2d;
}

    template <class T>
    inline void
deleteArray2D(T ** var_2d)
{
    ensure(var_2d, XLIB_USR_1_3("var_2d"));
    ensure(var_2d[0], XLIB_USR_1_3("var_2d"));
    delete [] var_2d[0];
    delete [] var_2d;
}

// FIXME: For some reason, 2-D pointer (**) array isn't always compiled.
    template <class T>
    inline void
copyArray2D(T * dst, const T * src, const int dim_x, const int dim_y) {
    ensure(dst, XLIB_USR_1_3("dst"));
    ensure(src, XLIB_USR_1_3("src"));
    //ensure(dst[0], XLIB_USR_1_3("dst[0]"));
    //ensure(src[0], XLIB_USR_1_3("src[0]"));
    ensure(dim_x > 0, XLIB_USR_2_1("X"));
    ensure(dim_y > 0, XLIB_USR_2_1("Y"));
    //memcpy(dst[0], src[0], dim_y * dim_x * sizeof(T));
    memcpy(dst, src, dim_y * dim_x * sizeof(T));
}

    template <class T> T ***
newArray3D( // Sizes of X, Y, and Z dimensions
    const int dim_x, const int dim_y, const int dim_z,
    T *** var_3d2 = NULL) // If given, a copy of var_3d2 is applied.
{
    ensure(dim_x > 0, XLIB_USR_2_1("X"));
    ensure(dim_y > 0, XLIB_USR_2_1("Y"));
    ensure(dim_z > 0, XLIB_USR_2_1("Z"));

    T *** var_3d;
    var_3d = new T** [dim_z];
    ensure(var_3d, XLIB_DEV_2_2("a 3-D array"));

    var_3d[0] = new T* [dim_z * dim_y];
    ensure(var_3d[0], XLIB_DEV_2_2("a 3-D array"));
    for (int i = 1; i < dim_z; i++) { var_3d[i] = var_3d[i-1] + dim_y; }

    var_3d[0][0] = new T [dim_z * dim_y * dim_x];
    ensure(var_3d[0][0], XLIB_DEV_2_2("a 3-D array"));
    // All array elements are set to 0.
    memset(var_3d[0][0], 0, dim_z * dim_y * dim_x * sizeof(T));
    for (int i = 0; i < dim_z; i++) {
        for (int j = 1; j < dim_y; j++) {
            var_3d[i][j] = var_3d[i][j-1] + dim_x;
        }
        if (i+1 < dim_z) var_3d[i+1][0] = var_3d[i][0] + dim_y * dim_x;
    }

    if (var_3d2 != NULL) {
        memcpy(var_3d[0][0], var_3d2[0][0], dim_z * dim_y * dim_x * sizeof(T));
    }
    return var_3d;
}

    template <class T>
    inline void
deleteArray3D(T *** var_3d)
{
    ensure(var_3d, XLIB_USR_1_3("var_3d"));
    ensure(var_3d[0], XLIB_USR_1_3("var_3d[0]"));
    ensure(var_3d[0][0], XLIB_USR_1_3("var_3d[0][0]"));
    delete [] var_3d[0][0];
    delete [] var_3d[0];
    delete [] var_3d;
}

// FIXME: For some reason, 3-D pointer (***) array isn't always compilable.
    template <class T>
    inline void
copyArray3D(T * dst, const T * src,
    const int dim_x, const int dim_y, const int dim_z) {
    ensure(dst, XLIB_USR_1_3("dst"));
    ensure(src, XLIB_USR_1_3("src"));
    //ensure(dst[0], XLIB_USR_1_3("dst[0]"));
    //ensure(src[0], XLIB_USR_1_3("src[0]"));
    //ensure(dst[0][0], XLIB_USR_1_3("dst[0][0]"));
    //ensure(src[0][0], XLIB_USR_1_3("src[0][0]"));
    ensure(dim_x > 0, XLIB_USR_2_1("X"));
    ensure(dim_y > 0, XLIB_USR_2_1("Y"));
    ensure(dim_z > 0, XLIB_USR_2_1("Z"));
    //memcpy(dst[0][0], src[0][0], dim_z * dim_y * dim_x * sizeof(T));
    memcpy(dst, src, dim_z * dim_y * dim_x * sizeof(T));
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Allocates an array with given values of a specified type.]  */
/*                                                                           */
/*  Description [The values can be the following according to a given type.] */
/*                                                                           */
/*===========================================================================*/

// Data distribution types

namespace InitDataKind {
    enum T {
        RANDOM,         // Random generated values.
        ASCENDING,      // Increasing by one from 0.
        DESCENDING      // Decreasing by one from size.
    };
}

template <class T>
    void
initArray(
    T *data,
    const unsigned int size,
    const InitDataKind::T type = InitDataKind::RANDOM,
    const T value_max = 127)
{
    T begin = 0;
    switch(type) {
    case InitDataKind::RANDOM:
        for (unsigned int i = 0; i < size; ++i) {
            data[i] = rand() / (T) value_max;
        }
        break;
    case InitDataKind::ASCENDING:
        begin = 0;
        for (unsigned int i = 0; i < size; ++i) {
            data[i] = begin;
            begin++;
        }
        break;
    case InitDataKind::DESCENDING:
        begin = size;
        for (unsigned int i = 0; i < size; ++i) {
            data[i] = begin;
            begin--;
        }
        break;
    default:
        outputMsg("Undefined type value.", __FILE__, __LINE__);
    }
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Supplementary print functions for template <class T> multi- */
/*      dimension pointer arrays.]                                           */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*  Note        [Boundry checked is not applied for the multi-dimension      */
/*      template <class T> arrays.]                                          */
/*  Note        [For some reason, GCC-4.3.4 doesn't support template function*/
/*      with multiple-dimension array pointers as function parameters, we    */
/*      have to use macros to enumerate all used built-in-types.]            */
/*                                                                           */
/*===========================================================================*/

    template <class T> string
contents(const T *rhs, /* 1D array */
    const unsigned int dim_x) { /* Size of the 1D array */
    ensure(rhs, XLIB_USR_1_3("array *rhs"));
    string s = "{";
    for (unsigned int i = 0; i < dim_x; i++) {
        if (i+1 < dim_x) s += toString(rhs[i]) + ",";
        else s += toString(rhs[i]);
    }
    s = s + "}";
    return s;
}

#define xcpplibContentsT2DPtrPrototype(TYPE)                                  \
    string                                                                    \
contents(TYPE **rhs, /* 2D array */                                           \
    /* Sizes of the 2D array */                                               \
    const unsigned int dim_y, const unsigned int dim_x);

xcpplibApplyBuiltInType(xcpplibContentsT2DPtrPrototype);

#define xcpplibContentsT3DPtrPrototype(TYPE)                                  \
    string                                                                    \
contents(TYPE ***rhs, /* 3D array */                                          \
    /* Sizes of the 3D array */                                               \
    const unsigned int dim_z, const unsigned int dim_y,                       \
    const unsigned int dim_x);

xcpplibApplyBuiltInType(xcpplibContentsT3DPtrPrototype);

/*---------------------------------------------------------------------------*/
/*  Testing functions                                                        */
/*---------------------------------------------------------------------------*/

    void
xcpplibTypesTest(void);

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

}

#endif // XCPPLIB_TYPES_H

