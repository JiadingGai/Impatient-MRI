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

    File Name   [xcpplib_global.h]

    Synopsis    [This file defines the global structures, variables, and so
        on, used in xcpplib library. This header file shall be used among other
        xcpplib source files and listed at the very beginning place.]

    Revision    [0.1; Initial build; Xiao-Long Wu, ECE UIUC]
    Date        [04/19/2010]

 *****************************************************************************/

#ifndef XCPPLIB_GLOBAL_H
#define XCPPLIB_GLOBAL_H

/*---------------------------------------------------------------------------*/
/*  Included libraries                                                       */
/*---------------------------------------------------------------------------*/

#include <iostream>
#include <cstdio>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

namespace xcpplib {

/*---------------------------------------------------------------------------*/
/*  Debug macro definitions used in XCPP libraries                           */
/*---------------------------------------------------------------------------*/

// This enables all XCPPLIB debugging messages if enabled in the functions.
#ifdef XCPPLIB_DEBUG
    #warning "XCPPLIB_DEBUG is enabled."
    #define XCPPLIB_DEBUG_MODE              true
    #define XCPPLIB_ENABLE_DEBUG_MSG        true
    // If enable gdb to trace back to the source of errors instead of outputting
    // error/warning messages only.
    #define XCPPLIB_ENABLE_GDB_TRACE        true
#else
    #define XCPPLIB_DEBUG_MODE              false
    #define XCPPLIB_ENABLE_DEBUG_MSG        false
    #define XCPPLIB_ENABLE_GDB_TRACE        false
#endif

#if XCPPLIB_ENABLE_DEBUG_MSG
    // This will slow down the performance a lot.
    #warning "XCPPLIB_ENABLE_DEBUG_MSG may slow down the performance a lot."
    #define xcpplibMsgDebug(args...)       msg(MSG_DEBUG, ##args)
#else
    #define xcpplibMsgDebug(args...)
#endif

#if XCPPLIB_ENABLE_DEBUG_MSG
    #define xcpplibMsgDebugHead(DEBUG_VARIABLE) \
        bool pre_msg_debug_mode; \
        if (DEBUG_VARIABLE) { \
            pre_msg_debug_mode = msgGetDebug(); \
            msgSetDebug(true); \
        } else { \
            pre_msg_debug_mode = msgGetDebug(); \
            msgSetDebug(false); \
        }

    #define xcpplibMsgDebugTail(DEBUG_VARIABLE) \
        msgSetDebug(pre_msg_debug_mode);
#else
    #define xcpplibMsgDebugHead(DEBUG_VARIABLE)
    #define xcpplibMsgDebugTail(DEBUG_VARIABLE)
#endif

/*---------------------------------------------------------------------------*/
/*  Macro definitions used in XCPP libraries or others                       */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Merge two strings into one.]                                */
/*                                                                           */
/*  Description [This is mainly used in the macro definition body to declare */
/*      variables using macro as part of the variable names.]                */
/*                                                                           */
/*===========================================================================*/

#define MERGE(X1, X2)       X1 ## X2
#define MERGE3(X1, X2, X3)  X1 ## X2 ## X3

// This macro is used to reduce coding efforts on built-in types, except bool.
#define xcpplibApplyBuiltInType(FUNC_NAME, args...) \
    FUNC_NAME(int                    , ##args);     \
    FUNC_NAME(unsigned int           , ##args);     \
    FUNC_NAME(short                  , ##args);     \
    FUNC_NAME(unsigned short         , ##args);     \
    FUNC_NAME(long int               , ##args);     \
    FUNC_NAME(unsigned long int      , ##args);     \
    FUNC_NAME(long long int          , ##args);     \
    FUNC_NAME(unsigned long long int , ##args);     \
    FUNC_NAME(float                  , ##args);     \
    FUNC_NAME(double                 , ##args);     \
    FUNC_NAME(long double            , ##args);     \
    FUNC_NAME(char                   , ##args);     \
    FUNC_NAME(unsigned char          , ##args);

// Obsolete
#if 0
#ifdef XCPPLIB_DEBUG
    #define XCPPLIB_NORMAL    0       // Printing processing stages
    #define XCPPLIB_DETAIL    1       // Printing detailed variable values
#else
    #define XCPPLIB_NORMAL    99      // Printing processing stages
    #define XCPPLIB_DETAIL    99      // Printing detailed variable values
#endif
#endif

/*---------------------------------------------------------------------------*/
/*  Constant declarations                                                    */
/*---------------------------------------------------------------------------*/

static const char XCPPLIB_VERSION[10] = "v0.1";

static const char * const ALPHABET_LOWER = "abcdefghijklmnopqrstuvwxyz";
static const char * const ALPHABET_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
static const char * const DIGITS = "1234567890";
static const char * const IDENTIFIER =
       "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_";

// The following values are different from machine to machine.
// We must use "define" instead of "static const" to avoid type issues.
// FIXME: Should have a better way to decide these numbers automatically.
//#if defined (Linux)
    #define MAX_INT_VALUE           2147483647
    #define MAX_INT_VALUE_DIGIT     10
    #define MAX_CHAR_VALUE          127
    #define MAX_CHAR_VALUE_DIGIT    3
//#else
//    #error "Undefined platform!\n"
//    #error "You must define the max integer value at this platform here.\n"
//#endif

/*---------------------------------------------------------------------------*/
/*  Output message definitions                                               */
/*---------------------------------------------------------------------------*/

// "XLIB_DEV": Errors caused by careless developers (me usually).
// Hence functions accessed internally will use these messages.
// First number: Category, second number: serial number
// ===========================================================================

// Trivial or unknown errors -------------------------------------------------

#define XLIB_DEV_0_0 \
    "XLIB-DEV-0-0: Unknown error caused by the developer."
#define XLIB_DEV_0_1 \
    "XLIB-DEV-0-1: Current maximum supported array dimension is 3."

// Violations on accessing arrays or pointers --------------------------------

#define XLIB_DEV_1_0 \
    "XLIB-DEV-1-0: Unknown error on accessing arrays or pointers."

// Violations on allocating memory -------------------------------------------

#define XLIB_DEV_2_0 \
    "XLIB-DEV-2-0: Unknown error on allocating memory."
#define XLIB_DEV_2_1 \
    "XLIB-DEV-2-1: Failed to allocate more memory."
#define XLIB_DEV_2_2(MEM_NAME) \
    "XLIB-DEV-2-2: Failed to allocate more memory for " #MEM_NAME "."

// "XLIB_USR": Errors caused by careless users.
// Hence functions accessed externally by users will use these messages.
// First number: Category, second number: serial number
// ===========================================================================

// Trivial or unknown errors -------------------------------------------------

#define XLIB_USR_0_0 \
    "XLIB-USR-0-0: Unknown error caused by the user."

// Violations on accessing memory --------------------------------------------

#define XLIB_USR_1_0 \
    "XLIB-USR-1-0: Unknown error on accessing arrays or pointers."
#define XLIB_USR_1_1(DIM) \
    "XLIB-USR-1-1: Accessing " #DIM " dimension with negative subscript."
#define XLIB_USR_1_2(MEM_NAME, DIM) \
    "XLIB-USR-1-2: Accessing " #MEM_NAME " at " #DIM " dimension with negative subscript."
#define XLIB_USR_1_3(PTR_NAME) \
    "XLIB-USR-1-3: Null input pointer parameter " #PTR_NAME "."
#define XLIB_USR_1_4 \
    "XLIB-USR-1-4: Read empty stack."
#define XLIB_USR_1_5 \
    "XLIB-USR-1-5: Stack overflow."

// Violations on allocating memory -------------------------------------------

#define XLIB_USR_2_0 \
    "XLIB-USR-2-0: Unknown error on allocating memory."
#define XLIB_USR_2_1(DIM) \
    "XLIB-USR-2-1: Allocating memory at " #DIM " dimension with illegal value."
#define XLIB_USR_2_2(MEM_NAME, DIM) \
    "XLIB-USR-2-2: Allocating " #MEM_NAME " at " #DIM " dimension with illegal value."

/*---------------------------------------------------------------------------*/
/*  Variable declarations                                                    */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*  External function prototypes                                             */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*  Static function prototypes                                               */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*  General function prototypes                                              */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*  Testing functions                                                        */
/*---------------------------------------------------------------------------*/

    void
xcpplibGlobalTest(void);

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

}

#endif // XCPPLIB_GLOBAL_H


