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

    File Name   [xcpplib_others.h]

    Synopsis    [Miscellaneous functions.]

    Description [Miscellaneous helper routines which can not be categorized.]

    Revision    [0.1; Initial build; Xiao-Long Wu, ECE UIUC]
    Date        [04/04/2010]

 *****************************************************************************/

#ifndef XCPPLIB_OTHERS_H
#define XCPPLIB_OTHERS_H

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>
#include <time.h>      // for timer
#include <sys/time.h>  // for timer

// XCPPLIB library
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
/*  Data structure implementations                                           */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*  Function prototypes or implementations                                   */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Timer.]                                                     */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

#define timerStart(JOB_NAME, SPACES)                                         \
        printf("%sTiming '%s' started\n", SPACES, #JOB_NAME);                \
        struct timeval MERGE(JOB_NAME, tv);                                  \
        struct timezone MERGE(JOB_NAME, tz);                                 \
        const clock_t MERGE(JOB_NAME, startTime) = clock();                  \
        gettimeofday(&MERGE(JOB_NAME, tv), &MERGE(JOB_NAME, tz));            \
        long MERGE(JOB_NAME, GtdStartTime) =                                 \
             MERGE(JOB_NAME, tv).tv_sec * 1000 +                             \
             MERGE(JOB_NAME, tv).tv_usec / 1000;                             \
        fflush(stdout);

#define timerStop(JOB_NAME, SPACES)                                          \
        gettimeofday(&MERGE(JOB_NAME, tv), &MERGE(JOB_NAME, tz));            \
        long MERGE(JOB_NAME, GTODEndTime) =                                  \
            MERGE(JOB_NAME, tv).tv_sec * 1000 +                              \
            MERGE(JOB_NAME, tv).tv_usec / 1000;                              \
        const clock_t MERGE(JOB_NAME, endTime) = clock();                    \
        const clock_t MERGE(JOB_NAME, elapsedTime) =                         \
            MERGE(JOB_NAME, endTime) - MERGE(JOB_NAME, startTime);           \
        const double MERGE(JOB_NAME, GtdTimeInSeconds) =                     \
            (double)(MERGE(JOB_NAME, GTODEndTime) -                          \
            MERGE(JOB_NAME, GtdStartTime)) / 1000.;                          \
        printf("%s  GetTimeOfDay Time = %g sec\n", SPACES,                   \
            MERGE(JOB_NAME, GtdTimeInSeconds));                              \
        const double MERGE(JOB_NAME, clockTimeInSeconds) =                   \
            (MERGE(JOB_NAME, elapsedTime) / (double)CLOCKS_PER_SEC);         \
        printf("%s  Clock Time        = %g sec\n", SPACES,                   \
            MERGE(JOB_NAME, clockTimeInSeconds));                            \
        printf("%sTiming '%s' ended\n", SPACES, #JOB_NAME);                  \
        fflush(stdout);

#define TIME_IT(ROUTINE_NAME__, LOOPS__, ACTION__, SPACES) \
    { \
        int loops = 0; \
        printf("%sTiming '%s' started\n", SPACES, ROUTINE_NAME__); \
        struct timeval tv; \
        struct timezone tz; \
        const clock_t startTime = clock(); \
        gettimeofday(&tv, &tz); long GtdStartTime =  tv.tv_sec * 1000 + tv.tv_usec / 1000; \
        for (loops = 0; loops < (LOOPS__); ++loops) { \
            ACTION__; \
        } \
        gettimeofday(&tv, &tz); long GTODEndTime =  tv.tv_sec * 1000 + tv.tv_usec / 1000; \
        const clock_t endTime = clock(); \
        const clock_t elapsedTime = endTime - startTime; \
        const double clockTimeInSeconds = \
            (elapsedTime / (double)CLOCKS_PER_SEC); \
        printf("%s  GetTimeOfDay Time (for %d iterations) = %g sec\n", \
            SPACES, LOOPS__, (double)(GTODEndTime - GtdStartTime) / 1000.); \
        printf("%s  Clock Time        (for %d iterations) = %g sec\n", \
            SPACES, LOOPS__, clockTimeInSeconds); \
        printf("%sTiming '%s' ended\n", SPACES, ROUTINE_NAME__); \
        fflush(stdout); \
    }

// You must call timerInit() before other timer functions.
    void
timerInit(unsigned int t = 1); // Default is Timer 1

    double
timerElapsedDay(void); // In days

    double
timerElapsedHour(void); // In hours

    double
timerElapsedMin(void); // In minutes

    double
timerElapsedSec(void); // In seconds

    double
timerElapsedMs(void); // In milliseconds

    string
timerElapsedTimeString(void);

    void
timerTest(void);

    string
getCurrentTime(void);

    string
getCurrentDate(void);

    string
getCurrentDateTime(void);

/*===========================================================================*/
/*  File/Directory management routines                                       */
/*===========================================================================*/

    EXIT_TYPE
getFileName(string &file_name, const string &file_path);

    string
getFileName(const string &file_path);

    EXIT_TYPE
getFileNames(
    string **file_names,
    const unsigned int num_files,
    const string *file_paths);

    EXIT_TYPE
getDirName(string &file_dir, const string &file_path);

    string
getDirName(const string &file_path);

    EXIT_TYPE
getDirNames(
    string **dir_names,
    const unsigned int num_dirs,
    const string *dir_paths);

    EXIT_TYPE
checkDirPath(const char * path,
    const bool if_create_dir = false); // If create the directory
    EXIT_TYPE
checkDirPath(const string &path,
    const bool if_create_dir = false); // If create the directory

    EXIT_TYPE
checkFilePath(const char * path);
    EXIT_TYPE
checkFilePath(const string &path);

    FILE *
openFile(
    const char * const fn_p,
    const char * const open_mode_p,
    const int if_silent = true); // If not show messages

    EXIT_TYPE
mergeFiles(
    const string &merged_file_path,
    const unsigned int &file_num, const string *file_paths,
    const bool if_interrupt = false, // If enable interrupt during computation
    const bool if_verbose = false);

    EXIT_TYPE
mergeFiles(
    const string &merged_file_path,
    const string &file_path1, const string &file_path2,
    const bool if_interrupt = false, // If enable interrupt during computation
    const bool if_verbose = false);

    EXIT_TYPE
copyFile(
    const string &dst, const string &src, const bool if_verbose = false);

    EXIT_TYPE
appendFile(
    const string &dst, const string &src, const bool if_verbose = false);

    EXIT_TYPE
moveFile(
    const string &dst, const string &src, const bool if_verbose = false);

    EXIT_TYPE
cleanDir(const string &dir_path, const bool if_verbose = false);

    EXIT_TYPE
removeDir(const string &dir_path, const bool if_verbose = false);

    EXIT_TYPE
removeFile(const string &file_path, const bool if_verbose = false);

    EXIT_TYPE
grepString(
    const string &log_file, // File to store the grepped string.
                            // Grepped string is appended.
    const string &str, const string &file_path, const bool if_verbose = false);

    void
checkPermission(
    const char * const fn_p,
    const char * const open_mode_p,
    const int if_silent             // If not show messages
    );

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Compare two arrays of floating-point numbers.]              */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    inline string
reverse(const string &src)
{
    string dst;
    for (int i = src.length(); i > -1; i--) {
        dst.push_back(src[i]);
    }
    return dst;
}

    EXIT_TYPE
compareArray(
    const float *data1, const float *data2, const int ele_num,
    const float precision);

    void
printDiff(
    const float *data1, const float *data2, const int ele_num,
    const float precision);

/*---------------------------------------------------------------------------*/
/*  Testing functions                                                        */
/*---------------------------------------------------------------------------*/

    void
xcpplibOthersTest(void);

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

}

#endif // XCPPLIB_OTHERS_H

