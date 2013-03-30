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

    File Name   [xcpplib_types.cpp]

    Synopsis    [Helper functions on types.]

    Description [See the corresponding header file for details.]

    Revision    [0.1; Initial build; Xiao-Long Wu, ECE UIUC]
    Date        [04/04/2010]

 *****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>
#include <cstdio>
#include <cstring>

// XCPPLIB libraries
#include <xcpplib_global.h>
#include <xcpplib_process.h>
#include <xcpplib_types.h>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

namespace xcpplib {

/*---------------------------------------------------------------------------*/
/*  Namespace used                                                           */
/*---------------------------------------------------------------------------*/

using namespace std;

/*---------------------------------------------------------------------------*/
/*  Function implementations                                                 */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Convert any values of any basic types into string type.]    */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

#if 0 // These are listed in the header file as inline functions
#define xcpplibToString(TYPE)                                                \
        string                                                               \
    toString(const TYPE &a)                                                  \
    {                                                                        \
        stringstream s;                                                      \
        s<< a;                                                               \
        return s.str();                                                      \
    }

xcpplibApplyBuiltInType(xcpplibToString);

    string
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
/*  Synopsis    [Inversely Resizes the string content to n characters.]      */
/*                                                                           */
/*  Description [If n is smaller than the current length of the string, the  */
/*      content is reduced to its first n characters, the rest being dropped.*/
/*      If n is greater than the current length of the string, the content   */
/*      is expanded by inserting in the front of the string as many          */
/*      instances of the c character as needed to reach a size of n          */
/*      characters.                                                          */
/*                                                                           */
/*      For example, "102" can be resized to "0102" when n is 4 and c is '0'.*/
/*      And "102" can also be resized to "10" when n is 2.]                  */
/*                                                                           */
/*===========================================================================*/
    string
stringIResize(const string &str, const size_t n, char c)
{
    if (str.length() >= n) {
        string new_str(str);
        new_str.resize(n, c);
        return new_str;
    } else {
        string new_str;
        new_str.append(n - str.length(), c);
        return new_str + str;
    }
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Check if two reads have overlapping in between on their     */
/*      two edges (ends) according to the given overlap digit values. The    */
/*      found overlap digit value is returned if there is one.]              */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    bool
checkEdgeOverlap(
    unsigned int &found_digits,   // Overlapping digits found
    const unsigned int &start_digits,   // Starting digits
    const string &read1, const string &read2,
    // Compare digits with a certain stride to reduce time.
    // This may produce some false positive results.
    // Note: The odd check_stride can have wrong comparison for even number of
    // read digits.
    const unsigned int check_stride)
{
    #if XCPPLIB_DEBUG_MODE
    const bool if_debug = true;
    #endif
    #define DEBUG_checkEdgeOverlap 1
    xcpplibMsgDebugHead(if_debug);

    bool if_overlapping = false;

    try { // Begin of try *****************************************************

    if (start_digits < 1) {
        msg(MSG_ERROR, "Starting digits must be greater than 0.");
        throw MSG_INTERNAL_ERROR;
    }
    if (check_stride < 1) {
        msg(MSG_ERROR, "Checking stride digits must be greater than 0.");
        throw MSG_INTERNAL_ERROR;
    }

    xcpplibMsgDebug("read1: %s", read1.c_str());
    xcpplibMsgDebug("read2: %s", read2.c_str());

    found_digits = start_digits;

    // Check until the shorter one's length is reached.
    // read2.length > read1.length
    //   read2 ATCGATCGATCG ->
    //   read1  <- ATCGATCG
    // read2.length < read1.length
    //   read2 ATCGATCG ->
    //   read1  <- ATCGATCGATCG
    // read2.length == read1.length
    //   read2 ATCGATCGATCG ->
    //   read1  <- ATCGATCGATCG

    #if 1 // Check from the given start_digits and break when find matches.
    xcpplibMsgDebug("read2 -> <- read1");
    for (long i = start_digits-1;
        i < (long) read1.length() && (long) read2.length()-i > 0;
        i+=check_stride) {
        #if DEBUG_checkEdgeOverlap
        xcpplibMsgDebug("read2 %s -> <- %s read1",
            read2.substr(read2.length()-i-1, i+1).c_str(),
            read1.substr(0, i+1).c_str());
        #endif
        const int result = read1.compare(0, i+1,
                           read2, read2.length()-i-1, i+1);
        if (result == 0) {
            if_overlapping = true;
            found_digits = i+1;
            xcpplibMsgDebug("***Found overlapping digits %u.", i+1);
            break;
        }
    }
    #else
    // Check every digit until the given start_digits number is reached.
    // start_digits is output variable.
    xcpplibMsgDebug("read2 -> <- read1");
    for (long i = 0;
        i < (long) read1.length() && (long) read2.length()-i > 0; i++) {
        #if DEBUG_checkEdgeOverlap
        xcpplibMsgDebug("read2 %s -> <- %s read1",
            read2.substr(read2.length()-i-1, i+1).c_str(),
            read1.substr(0, i+1).c_str());
        #endif
        const int result = read1.compare(0, i+1,
                           read2, read2.length()-i-1, i+1);
        if (result == 0) {
            if_overlapping = true;
            start_digits = i+1;
            xcpplibMsgDebug("***Found overlapping digits %u.", start_digits);
        }
    }
    #endif

    if (if_overlapping) throw EXIT_NORMAL;

    xcpplibMsgDebug("read1 -> <- read2");
    //   read1 ATCGATCGATCG ->
    //   read2  <- ATCGATCGATCG
    #if 1 // Check from the given start_digits and break when find matches.
    for (long i = start_digits-1;
        i < (long) read2.length() && (long) read1.length()-i > 0;
        i+=check_stride) {
        #if DEBUG_checkEdgeOverlap
        xcpplibMsgDebug("read1 %s -> <- %s read2",
            read1.substr(read1.length()-i-1, i+1).c_str(),
            read2.substr(0, i+1).c_str());
        #endif
        const int result = read2.compare(0, i+1,
                           read1, read1.length()-i-1, i+1);
        if (result == 0) {
            if_overlapping = true;
            found_digits = i+1;
            xcpplibMsgDebug("***Found overlapping digits %u.", i+1);
            break;
        }
    }
    #else
    // Check every digit until the given start_digits number is reached.
    // start_digits is output variable.
    for (long i = 0;
        i < (long) read2.length() && (long) read1.length()-i > 0; i++) {
        #if DEBUG_checkEdgeOverlap
        xcpplibMsgDebug("read1 %s -> <- %s read2",
            read1.substr(read1.length()-i-1, i+1).c_str(),
            read2.substr(0, i+1).c_str());
        #endif
        const int result = read2.compare(0, i+1,
                           read1, read1.length()-i-1, i+1);
        if (result == 0) {
            if_overlapping = true;
            start_digits = i+1;
            xcpplibMsgDebug("***Found overlapping digits %u.", start_digits);
        }
    }
    #endif
    } // End of try ***********************************************************

    catch (EXIT_TYPE error_code) {
        switch (error_code) {
        case EXIT_NORMAL: break; // Do nothing
        default:
            xcpplibMsgDebugTail(if_debug);
            exit(1);
        }
    }

    xcpplibMsgDebugTail(if_debug);
    return if_overlapping;
} // End of checkEdgeOverlap()

    void
checkEdgeOverlapTest(void)
{
    msg(2, "Testing checkEdgeOverlap().");

    const unsigned int start_digits = 4;
    unsigned int found_digits = 0;
    string read1, read2;

    read1 = "12345678";
    read2 = "12345678";
    // Note: The odd check_stride can have wrong comparison for even number of
    // read digits.
    if (!checkEdgeOverlap(found_digits, start_digits, read1, read2, 2)) {
        msg(MSG_INTERNAL_ERROR, "*** The implementation has bugs.");
    }

    read1 = "1234";
    read2 = "12345678";
    if (!checkEdgeOverlap(found_digits, start_digits, read1, read2)) {
        msg(MSG_INTERNAL_ERROR, "*** The implementation has bugs.");
    }

    read1 = "12345678";
    read2 = "1234";
    if (!checkEdgeOverlap(found_digits, start_digits, read1, read2)) {
        msg(MSG_INTERNAL_ERROR, "*** The implementation has bugs.");
    }

    read1 = "12345678";
    read2 = "12341234";
    if (!checkEdgeOverlap(found_digits, start_digits, read1, read2)) {
        msg(MSG_INTERNAL_ERROR, "*** The implementation has bugs.");
    }

    read1 = "12345678";
    read2 = "12344321";
    if (checkEdgeOverlap(found_digits, start_digits, read1, read2)) {
        msg(MSG_INTERNAL_ERROR, "*** The implementation has bugs.");
    }
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Check if two reads have overlapping in between on their     */
/*      two edges (ends) according to the given overlap digit values. The    */
/*      overlapping will be exactly the given overlap digit values.]         */
/*                                                                           */
/*  Description []                                                           */
/*  Note        [Using inline function is faster but the debugging messages  */
/*      can not be printed according the macro definitions in other files.]  */
/*                                                                           */
/*===========================================================================*/

    bool
checkEdgeOverlapFix(
    const unsigned int &overlap_digits,   // How many overlapping digits
    const string &read1, const string &read2,
    // Compare digits with a certain stride to reduce time.
    // This may produce some false positive results.
    const unsigned int check_stride)
{
    #if XCPPLIB_DEBUG_MODE
    const bool if_debug = true;
    #endif
    xcpplibMsgDebugHead(if_debug);
    #define DEBUG_checkEdgeOverlapFix 1

    if (check_stride < 1) {
        msg(MSG_ERROR, "Checking stride digits must be greater than 0.");
        throw MSG_INTERNAL_ERROR;
    }

    xcpplibMsgDebug("read1: %s", read1.c_str());
    xcpplibMsgDebug("read2: %s", read2.c_str());

    // Check until the shorter one's length is reached.
    // read2.length > read1.length
    //   read2 ATCGATCGATCG ->
    //   read1  <- ATCGATCG
    // read2.length < read1.length
    //   read2 ATCGATCG ->
    //   read1  <- ATCGATCGATCG
    // read2.length == read1.length
    //   read2 ATCGATCGATCG ->
    //   read1  <- ATCGATCGATCG

    // Check from the given overlap_digits and break when find matches.
    #if DEBUG_checkEdgeOverlapFix
    xcpplibMsgDebug("read2 -> <- read1");
    #endif
    if (overlap_digits <= read1.length() &&
        (long) (read2.length()-overlap_digits) >= 0) {
        #ifdef DEBUG_checkEdgeOverlapFix
        xcpplibMsgDebug("read2 %s -> <- %s read1",
            read2.substr(read2.length()-overlap_digits, overlap_digits).c_str(),
            read1.substr(0, overlap_digits).c_str());
        #endif
        #if 1 // Compare digits with a certain stride to reduce time.
              // This may produce some false positive results.
        int result = 0;
        for (long j = 0; j < (long) overlap_digits; j+=check_stride) {
            xcpplibMsgDebug("read1[%u](%c) ?= read2[%u](%c)", j, read1[j],
                read2.length()-overlap_digits+j,
                read2[read2.length()-overlap_digits+j]);
            if (read1[j] != read2[read2.length()-overlap_digits+j]) {
                result = 1; break;
            }
        }
        #elif 0 // Functional equivalent to the original version
        int result = 0;
        for (long j = 0; j < (long) overlap_digits; j++) {
            xcpplibMsgDebug("read1[%u](%c) ?= read2[%u](%c)", j, read1[j],
                read2.length()-overlap_digits+j,
                read2[read2.length()-overlap_digits+j]);
            if (read1[j] != read2[read2.length()-overlap_digits+j]) {
                result = 1; break;
            }
        }
        #else // Original version
        xcpplibMsgDebug("read1[%u, %u] ?= read2[%u, %u]", 0, overlap_digits,
            read2.length()-overlap_digits, overlap_digits);
        const int result = read1.compare(0, overlap_digits,
                  read2, read2.length()-overlap_digits, overlap_digits);
        #endif
        if (result == 0) {
            xcpplibMsgDebug("***Found overlapping digits %u.", overlap_digits);
            return true;
        }
    }

    #ifdef DEBUG_checkEdgeOverlapFix
    xcpplibMsgDebug("read1 -> <- read2");
    #endif
    //   read1 ATCGATCGATCG ->
    //   read2  <- ATCGATCGATCG
    // Check from the given overlap_digits and break when find matches.
    if (overlap_digits <= read2.length() &&
        (long) (read1.length()-overlap_digits) >= 0) {
        #ifdef DEBUG_checkEdgeOverlapFix
        xcpplibMsgDebug("read1 %s -> <- %s read2",
            read1.substr(read1.length()-overlap_digits, overlap_digits).c_str(),
            read2.substr(0, overlap_digits).c_str());
        #endif
        #if 1 // Compare digits with a certain stride to reduce time.
              // This may produce some false positive results.
        int result = 0;
        for (long j = 0; j < (long) overlap_digits; j+=check_stride) {
            xcpplibMsgDebug("read2[%u](%c) ?= read1[%u](%c)", j, read2[j],
                read1.length()-overlap_digits+j,
                read1[read1.length()-overlap_digits+j]);
            if (read2[j] != read1[read1.length()-overlap_digits+j]) {
                result = 1; break;
            }
        }
        #elif 0 // Functional equivalent to the original version
        int result = 0;
        for (long j = 0; j < (long) overlap_digits; j++) {
            xcpplibMsgDebug("read2[%u](%c) ?= read1[%u](%c)", j, read2[j],
                read1.length()-overlap_digits+j,
                read1[read1.length()-overlap_digits+j]);
            if (read2[j] != read1[read1.length()-overlap_digits+j]) {
                result = 1; break;
            }
        }
        #else // Original version
        xcpplibMsgDebug("read2[%u, %u] ?= read1[%u, %u]", 0, overlap_digits,
            read1.length()-overlap_digits, overlap_digits);
        const int result = read2.compare(0, overlap_digits,
                  read1, read1.length()-overlap_digits, overlap_digits);
        #endif
        if (result == 0) {
            xcpplibMsgDebug("***Found overlapping digits %u.", overlap_digits);
            return true;
        }
    }

    xcpplibMsgDebugTail(if_debug);
    return false;
} // End of checkEdgeOverlapFix()

    void
checkEdgeOverlapFixTest(void)
{
    msg(2, "Testing checkEdgeOverlapFix().");

    unsigned int overlap_digits = 10;
    string read1, read2;

    read1 = "1234567890";
    read2 = "1234567890";
    if (!checkEdgeOverlapFix(overlap_digits, read1, read2)) {
        msg(MSG_INTERNAL_ERROR, "*** The implementation has bugs.");
    }

    read1 = "123400000000";
    read2 = "123456780000";
    if (checkEdgeOverlapFix(overlap_digits, read1, read2)) {
        msg(MSG_INTERNAL_ERROR, "*** The implementation has bugs.");
    }

    read1 = "123456780000";
    read2 = "123400000000";
    if (checkEdgeOverlapFix(overlap_digits, read1, read2)) {
        msg(MSG_INTERNAL_ERROR, "*** The implementation has bugs.");
    }

    read1 = "12345678";
    read2 = "12341234";
    if (checkEdgeOverlapFix(overlap_digits, read1, read2)) {
        msg(MSG_INTERNAL_ERROR, "*** The implementation has bugs.");
    }

    read1 = "123456780000";
    read2 = "123443211234";
    if (checkEdgeOverlapFix(overlap_digits, read1, read2)) {
        msg(MSG_INTERNAL_ERROR, "*** The implementation has bugs.");
    }
    overlap_digits = 4;
    if (!checkEdgeOverlapFix(overlap_digits, read1, read2)) {
        msg(MSG_INTERNAL_ERROR, "*** The implementation has bugs.");
    }
}

/*---------------------------------------------------------------------------*/
/*  Class TArray                                                             */
/*---------------------------------------------------------------------------*/

// Template class method implementations must be at the header file.

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Supplementary print functions for array class.]       */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*  Note        [Boundry checked is not applied for the multi-dimension      */
/*      array arrays.]                                                 */
/*  Note        [For some reason, GCC-4.3.4 doesn't support template function*/
/*      with multiple-dimension array pointers as function parameters, we    */
/*      have to use macros to enumerate all used built-in-types.]            */
/*                                                                           */
/*===========================================================================*/

#define xcpplibContentsT2DPtr(TYPE)                                           \
    string                                                                    \
contents(TYPE **rhs, /* 2D array */                                           \
    /* Sizes of the 2D array */                                               \
    const unsigned int dim_y, const unsigned int dim_x) {                     \
    ensure(rhs, XLIB_USR_1_3("array **rhs"));                                 \
    string s = "{";                                                           \
    for (unsigned int j = 0; j < dim_y; ++j) {                                \
        s += "{";                                                             \
        for (unsigned int i = 0; i < dim_x; ++i) {                            \
            if (i+1 < dim_x) s += toString(rhs[j][i]) + ",";                  \
            else s += toString(rhs[j][i]);                                    \
        }                                                                     \
        if (j+1 < dim_y) s += "},";                                           \
        else s += "}";                                                        \
    }                                                                         \
    s += "}";                                                                 \
    return s;                                                                 \
}

xcpplibApplyBuiltInType(xcpplibContentsT2DPtr);

#define xcpplibContentsT3DPtr(TYPE)                                           \
    string                                                                    \
contents(TYPE ***rhs, /* 3D array */                                          \
    /* Sizes of the 3D array */                                               \
    const unsigned int dim_z, const unsigned int dim_y,                       \
    const unsigned int dim_x) {                                               \
    ensure(rhs, XLIB_USR_1_3("array ***rhs"));                                \
    string s = "{";                                                           \
    for (unsigned int k = 0; k < dim_z; ++k) {                                \
        s += "{";                                                             \
        for (unsigned int j = 0; j < dim_y; ++j) {                            \
            s += "{";                                                         \
            for (unsigned int i = 0; i < dim_x; ++i) {                        \
                if (i+1 < dim_x) s += toString(rhs[k][j][i]) + ",";           \
                else s += toString(rhs[k][j][i]);                             \
            }                                                                 \
            if (j+1 < dim_y) s += "},";                                       \
            else s += "}";                                                    \
        }                                                                     \
        if (k+1 < dim_z) s += "},";                                           \
        else s += "}";                                                        \
    }                                                                         \
    s += "}";                                                                 \
    return s;                                                                 \
}

xcpplibApplyBuiltInType(xcpplibContentsT3DPtr);

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [This function is used for testing purpose only.]            */
/*                                                                           */
/*  Description [It can be called by other functions to facilitate the       */
/*      testing process.]                                                    */
/*                                                                           */
/*===========================================================================*/

    void
xcpplibTypesTest(void)
{
    checkEdgeOverlapTest();
    checkEdgeOverlapFixTest();
}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

}

