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

    File Name   [xcpplib_others.cpp]

    Synopsis    [Miscellaneous functions.]

    Description [See the corresponding header file for details.]

    Revision    [0.1; Initial build; Xiao-Long Wu, ECE UIUC]
    Date        [04/04/2010]

 *****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/types.h> // for stat & mkdir
#include <sys/stat.h>  // for stat & mkdir
#include <unistd.h>    // for stat & mkdir

// XCPPLIB libraries
#include <xcpplib_process.h>
#include <xcpplib_types.h>
#include <xcpplib_others.h>

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
/*  Synopsis    [Get the elapsed time in different units.]                   */
/*                                                                           */
/*  Description [It uses clock() function to get the timing information.]    */
/*                                                                           */
/*  Note        [You must call timerInit() to start the counting.]           */
/*  Note        [This supports only one timer so multiple timing measurement */
/*      are not enabled.]                                                    */
/*  Note        [Timer 0 doesn't reflect the time spent on system calls.     */
/*      Which means when you call another program through system call, it    */
/*      doesn't count. Timer 0 therefore counts the time taken by the CPU on */
/*      this program. Timer 1 counts the time starting from launching the    */
/*      program till the program is finished, no matter if during this time  */
/*      period the program is fully executed or not.]                        */
/*                                                                           */
/*===========================================================================*/

static unsigned int timer_id = 1;

// Timer 0
static clock_t timer_clock_beg; // Unit is in clock ticks

// Timer 1
static long timer_gettimeofday_beg; // Unit is in milliseconds
static struct timeval timer_time_value;
static struct timezone timer_time_zone;

    void
timerInit(unsigned int t) // Default is Timer 1
{
    timer_id = t;
    if (timer_id == 0) {
        timer_clock_beg = clock();
        //msg(MSG_DEBUG, "timer_clock_beg: %f", (double) timer_clock_beg);
    } else if (timer_id == 1) {
        gettimeofday(&timer_time_value, &timer_time_zone);
        timer_gettimeofday_beg = timer_time_value.tv_sec * 1000 +
                                 timer_time_value.tv_usec / 1000;
    } else {
        message("Undefined timer id.");
    }
}

    double
timerElapsedDay(void) // In days
{
    double elapsed_day = 0;
    if (timer_id == 0) { // Timer 0
        const clock_t now = clock();
        const clock_t elapsed = now - timer_clock_beg;
        elapsed_day = ((double) elapsed / (double) CLOCKS_PER_SEC) / 
                       (3600.0 * 24.0);

    } else { // Timer 1
        gettimeofday(&timer_time_value, &timer_time_zone);
        long timer_gettimeofday_end = timer_time_value.tv_sec * 1000 +
                                      timer_time_value.tv_usec / 1000;
        elapsed_day = (double) (timer_gettimeofday_end -
                      timer_gettimeofday_beg) / (1000.0 * 3600.0 * 24.0);
    }

    return elapsed_day;
}

    double
timerElapsedHour(void) // In hours
{
    double elapsed_hour = 0;
    if (timer_id == 0) { // Timer 0
        const clock_t now = clock();
        const clock_t elapsed = now - timer_clock_beg;
        elapsed_hour = ((double) elapsed / (double) CLOCKS_PER_SEC) / 3600.0;

    } else { // Timer 1
        gettimeofday(&timer_time_value, &timer_time_zone);
        long timer_gettimeofday_end = timer_time_value.tv_sec * 1000 +
                                      timer_time_value.tv_usec / 1000;
        elapsed_hour = (double) (timer_gettimeofday_end -
                       timer_gettimeofday_beg) / (1000.0 * 60.0 * 60.0);
    }

    return elapsed_hour;
}

    double
timerElapsedMin(void) // In minutes
{
    double elapsed_min = 0;
    if (timer_id == 0) { // Timer 0
        const clock_t now = clock();
        const clock_t elapsed = now - timer_clock_beg;
        elapsed_min = ((double) elapsed / (double) CLOCKS_PER_SEC) / 60.0;

    } else { // Timer 1
        gettimeofday(&timer_time_value, &timer_time_zone);
        long timer_gettimeofday_end = timer_time_value.tv_sec * 1000 +
                                      timer_time_value.tv_usec / 1000;
        elapsed_min = (double) (timer_gettimeofday_end -
                      timer_gettimeofday_beg) / (1000.0 * 60.0);
    }

    return elapsed_min;
}

    double
timerElapsedSec(void) // In seconds
{
    double elapsed_sec = 0;
    if (timer_id == 0) { // Timer 0
        const clock_t now = clock();
        const clock_t elapsed = now - timer_clock_beg;
        elapsed_sec = (double) elapsed / (double) CLOCKS_PER_SEC;
        //msg(MSG_DEBUG, "timer_clock_beg: %f", (double) timer_clock_beg);
        //msg(MSG_DEBUG, "now: %f", (double) now);
        //msg(MSG_DEBUG, "CLOCKS_PER_SEC: %f", (double) CLOCKS_PER_SEC);

    } else { // Timer 1
        gettimeofday(&timer_time_value, &timer_time_zone);
        long timer_gettimeofday_end = timer_time_value.tv_sec * 1000 +
                                      timer_time_value.tv_usec / 1000;
        elapsed_sec = (double) (timer_gettimeofday_end -
                                timer_gettimeofday_beg) / 1000.0;
    }

    return elapsed_sec;
}

    double
timerElapsedMs(void) // In milliseconds
{
    double elapsed_ms = 0;
    if (timer_id == 0) { // Timer 0
        const clock_t now = clock();
        const clock_t elapsed = now - timer_clock_beg;
        elapsed_ms = (double) elapsed / ((double) CLOCKS_PER_SEC / 1000.0);

    } else { // Timer 1
        gettimeofday(&timer_time_value, &timer_time_zone);
        long timer_gettimeofday_end = timer_time_value.tv_sec * 1000 +
                                      timer_time_value.tv_usec / 1000;
        elapsed_ms = (double) (timer_gettimeofday_end -
                               timer_gettimeofday_beg);
    }

    return elapsed_ms;
}

// Display the elapsed time in a beautiful way.
    string
timerElapsedTimeString(void)
{
    char elapsed_char[128];
    const double elapsed = timerElapsedSec();
    if (elapsed < 10) { // < 10 seconds
        sprintf(elapsed_char, "Elapsed time: %0.3f ms (%0.3f seconds)",
            timerElapsedMs(), elapsed);
    } else if (elapsed <= 60*10) { // < 10 mins
        sprintf(elapsed_char, "Elapsed time: %0.3f seconds (%0.3f minutes)",
            elapsed, timerElapsedMin());
    } else if (elapsed > 60*10 && elapsed <= 3600) { // < 1 hour
        sprintf(elapsed_char, "Elapsed time: %0.3f minutes (%0.3f hours)",
            timerElapsedMin(), timerElapsedHour());
    } else if (elapsed > 3600 && elapsed <= 3600*24) { // < 1 day
        sprintf(elapsed_char, "Elapsed time: %0.3f hours (%0.3f days)",
            timerElapsedHour(), timerElapsedDay());
    } else { // the rest
        sprintf(elapsed_char, "Elapsed time: %0.3f days (%0.3f weeks)",
            timerElapsedDay(), timerElapsedDay()/7.0);
    }

    return string(elapsed_char);
}

    void
timerTest(void)
{
    {
        timerInit(0);
        printf("before: %s\n", getCurrentDateTime().c_str());
        double i = 0;
        for (i = 0; i < 1000000000.0f; i++) {}
        printf("after: %s\n", getCurrentDateTime().c_str());
        double elapsed_min = timerElapsedMin();
        double elapsed_sec = timerElapsedSec();
        double elapsed_ms = timerElapsedMs();
        printf("time 0: %0.3f min, %0.3f sec, %0.3f ms (%0.0f)\n",
            elapsed_min, elapsed_sec, elapsed_ms, i);
    }

    {
        timerInit(1);
        printf("before: %s\n", getCurrentDateTime().c_str());
        double i = 0;
        for (i = 0; i < 1000000000.0f; i++) {}
        printf("after: %s\n", getCurrentDateTime().c_str());
        double elapsed_min = timerElapsedMin();
        double elapsed_sec = timerElapsedSec();
        double elapsed_ms = timerElapsedMs();
        printf("time 1: %0.3f min, %0.3f sec, %0.3f ms (%0.0f)\n",
            elapsed_min, elapsed_sec, elapsed_ms, i);
    }
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Returning the current time, date, or both.]                 */
/*  Description [Results:                                                    */
/*               GetCurrentTime(): 14:41:10                                  */
/*               GetCurrentDate(): 08/22/02                                  */
/*               GetCurrentDateTime(): 08/22/02 14:41:10                     */
/*  Note        [The return string can not be freed.]                        */
/*  See also    []                                                           */
/*                                                                           */
/*===========================================================================*/

    string
getCurrentTime(void)
{
    time_t t_now;
    struct tm *tm_now;
    char hour[3] = "\0";
    char min[3] = "\0";
    char sec[3] = "\0";

    time (&t_now);
    tm_now = localtime(&t_now);

    if (tm_now->tm_hour < 10) sprintf(hour, "0%d", tm_now->tm_hour);
    else sprintf(hour, "%d", tm_now->tm_hour);
    if (tm_now->tm_min < 10) sprintf(min, "0%d", tm_now->tm_min);
    else sprintf(min, "%d", tm_now->tm_min);
    if (tm_now->tm_sec < 10) sprintf(sec, "0%d", tm_now->tm_sec);
    else sprintf(sec, "%d", tm_now->tm_sec);

    string cur_time = string(hour) + ":" + string(min) + ":" + string(sec);
    return cur_time;
}

    string
getCurrentDate(void)
{
    time_t t_now;
    struct tm *tm_now;
    char year[3] = "\0";
    char mon[3] = "\0";
    char mday[3] = "\0";

    time (&t_now);
    tm_now = localtime(&t_now);

    if ((-100 + tm_now->tm_year) < 10)
        sprintf(year, "0%d", -100 + tm_now->tm_year);
    else
        sprintf(year, "%d", -100 + tm_now->tm_year);
    if (1 + tm_now->tm_mon < 10) sprintf(mon, "0%d", 1 + tm_now->tm_mon);
    else sprintf(mon, "%d", 1 + tm_now->tm_mon);
    if (tm_now->tm_mday < 10) sprintf(mday, "0%d", tm_now->tm_mday);
    else sprintf(mday, "%d", tm_now->tm_mday);

    string cur_time = string(mon) + "/" + string(mday) + "/" + string(year);
    return cur_time;
}

    string
getCurrentDateTime(void)
{
    time_t t_now;
    struct tm *tm_now;
    char year[3] = "\0";
    char mon[3] = "\0";
    char mday[3] = "\0";

    char hour[3] = "\0";
    char min[3] = "\0";
    char sec[3] = "\0";

    time (&t_now);
    tm_now = localtime(&t_now);

    // Get current date =======================================================

    if ((-100 + tm_now->tm_year) < 10)
        sprintf(year, "0%d", -100 + tm_now->tm_year);
    else
        sprintf(year, "%d", -100 + tm_now->tm_year);
    if (1 + tm_now->tm_mon < 10) sprintf(mon, "0%d", 1 + tm_now->tm_mon);
    else sprintf(mon, "%d", 1 + tm_now->tm_mon);
    if (tm_now->tm_mday < 10) sprintf(mday, "0%d", tm_now->tm_mday);
    else sprintf(mday, "%d", tm_now->tm_mday);

    // Get current time =======================================================

    if (tm_now->tm_hour < 10) sprintf(hour, "0%d", tm_now->tm_hour);
    else sprintf(hour, "%d", tm_now->tm_hour);
    if (tm_now->tm_min < 10) sprintf(min, "0%d", tm_now->tm_min);
    else sprintf(min, "%d", tm_now->tm_min);
    if (tm_now->tm_sec < 10) sprintf(sec, "0%d", tm_now->tm_sec);
    else sprintf(sec, "%d", tm_now->tm_sec);

    string cur_date_time =
           string(mon) + "/" + string(mday) + "/" + string(year) + " " +
           string(hour) + ":" + string(min) + ":" + string(sec);
    return cur_date_time;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Get the file/directory name out of the path.]               */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/
    EXIT_TYPE
getFileName(string &file_name, const string &file_path)
{
    msg(MSG_DEBUG, "getFileName(): %s", file_path.c_str());
    size_t found = file_path.find_last_of("/\\");
    file_name = file_path.substr(found+1);;

    return EXIT_NORMAL;
}

    string
getFileName(const string &file_path)
{
    msg(MSG_DEBUG, "getFileName(): %s", file_path.c_str());
    size_t found = file_path.find_last_of("/\\");
    string file_name = file_path.substr(found+1);;

    return file_name;
}

    EXIT_TYPE
getFileNames(
    string **file_names,
    const unsigned int num_files,
    const string *file_paths)
{
    if (file_names == NULL || num_files == 0 || file_paths == NULL) {
        outputMsg("Wrong input parameter for getFileNames().", false);
        return EXIT_WRONG_INPUTS;
    }

    *file_names = newArray1D<string>(num_files, false);
    for (unsigned int i = 0; i < num_files; i++) {
        EXIT_TYPE g = getFileName((*file_names)[i], file_paths[i]);
        if (g != EXIT_NORMAL) return g;
    }

    return EXIT_NORMAL;
}

    EXIT_TYPE
getDirName(string &file_dir, const string &file_path)
{
    msg(MSG_DEBUG, "getDirName(): %s", file_path.c_str());
    size_t found = file_path.find_last_of("/\\");
    file_dir = file_path.substr(0,found);

    return EXIT_NORMAL;
}

    string
getDirName(const string &file_path)
{
    msg(MSG_DEBUG, "getDirName(): %s", file_path.c_str());
    size_t found = file_path.find_last_of("/\\");
    string file_dir = file_path.substr(0,found);

    return file_dir;
}

    EXIT_TYPE
getDirNames(
    string **dir_names,
    const unsigned int num_dirs,
    const string *dir_paths)
{
    if (dir_names == NULL || num_dirs == 0 || dir_paths == NULL) {
        outputMsg("Wrong input parameter for getDirNames().", false);
        return EXIT_WRONG_INPUTS;
    }

    *dir_names = newArray1D<string>(num_dirs, false);
    for (unsigned int i = 0; i < num_dirs; i++) {
        EXIT_TYPE g = getDirName((*dir_names)[i], dir_paths[i]);
        if (g != EXIT_NORMAL) return g;
    }

    return EXIT_NORMAL;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Check the directory path.]                                  */
/*                                                                           */
/*  Description [For details, check "man 2 stat" and "man 2 mkdir" for more  */
/*      information.]                                                        */
/*                                                                           */
/*===========================================================================*/

    EXIT_TYPE
checkDirPath(const char *path,
    const bool if_create_dir) // If create the directory (default: false)
{
   const string path_ = path;
   struct stat st;
   if (stat(path, &st) < 0 || !S_ISDIR(st.st_mode)) {
       // No such directory path
       if (if_create_dir) {
           if (mkdir(path, 0755) != 0) return EXIT_FAIL_EXECUTION;
       } else {
           return EXIT_FAIL_EXECUTION;
       }
   }

   return EXIT_NORMAL;
}

    EXIT_TYPE
checkDirPath(const string &path,
    const bool if_create_dir) // If create the directory (default: false)
{
    return checkDirPath(path.c_str(), if_create_dir);
}

    EXIT_TYPE
checkFilePath(const char *path)
{
   const string path_ = path;
   struct stat64 st; // To handle files larger than 2GB.
   if (stat64(path, &st) < 0 || !S_ISREG(st.st_mode)) {
       // No such file path
       return EXIT_FAIL_EXECUTION;
   }

   return EXIT_NORMAL;
}

    EXIT_TYPE
checkFilePath(const string &path)
{
    return checkFilePath(path.c_str());
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [The corresponding ANSI fopen function while with more error */
/*               checks.]                                                    */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    FILE *
openFile(
    const char * const fn_p,
    const char * const open_mode_p,
    const int if_silent) // If not show messages
{
    FILE * f_p = NULL;

    if (fn_p == NULL) {
        msg(MSG_ERROR, "Null file name pointer.");
        exit (-1);
    }

    if (!if_silent) {
        msg(0, "Opening the file %s ... ", fn_p);
    }

    f_p = fopen(fn_p, open_mode_p);
    if (f_p == NULL) {
        if (!if_silent) {
            msg(0, "failed.\n");
        } else {
            msg(MSG_ERROR, "Opening the file %s ... failed.", fn_p);
        }
        msg(MSG_ERROR, "Please check file path.");
        exit (-1);
    }
    if (!if_silent) msg(0, "succeeded.\n");

    return (f_p);
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Merge multiple files into one.]                             */
/*                                                                           */
/*  Description [The destination file will be cleared (removed) before the   */
/*      process.                                                             */
/*      On success, it returns true. Return false or abort, otherwise.]      */
/*                                                                           */
/*===========================================================================*/

    EXIT_TYPE
mergeFiles(
    const string &merged_file_path,
    const unsigned int &file_num, const string *file_paths,
    const bool if_interrupt, // If enable interrupt during computation
    const bool if_verbose)
{
    #if XCPPLIB_DEBUG_MODE
    const bool if_debug = true;
    #endif
    xcpplibMsgDebugHead(if_debug);

    try { // Begin of try *****************************************************

    if (file_paths == NULL) {
        msg(MSG_ERROR, "Null pointer of file paths.");
        throw EXIT_WRONG_INPUTS;
    }

    // Remove the merged file before adding new contents.
    removeFile(merged_file_path);

    for (unsigned int i = 0; i < file_num; i++) {
        if (merged_file_path != file_paths[i]) {
            const string merge_cmd = "cat " + file_paths[i] + " >> " +
                                     merged_file_path;
            if (if_verbose) msg(2, "Merge cmd: %s", merge_cmd.c_str());
            if (system(merge_cmd.c_str()) != 0) {
                msg(MSG_ERROR, "Failed to merge file \"%s\".",
                    file_paths[i].c_str());
                msg(MSG_ERROR, "Please check disk quota or file I/O.");
                throw EXIT_FAIL_EXECUTION;
            }
        } else {
            if (if_verbose) {
                msg(MSG_WARNING, "The merged file path is the same as the input file path.");
            }
        }

        // Check if keyboard is pressed for termination.
        if (if_interrupt && keyboardInputKeyPressed()) {
            msg(1, "Program is terminated as you wish.");
            throw EXIT_USER_INTERRUPT;
        }
    }

    } // End of try ***********************************************************

    catch (EXIT_TYPE error_code) {
        switch (error_code) {
        case EXIT_NORMAL: break; // Do nothing
        default:
            xcpplibMsgDebugTail(if_debug);
            return error_code;
        }
    }

    xcpplibMsgDebugTail(if_debug);
    return EXIT_NORMAL;
}

    EXIT_TYPE
mergeFiles(
    const string &merged_file_path,
    const string &file_path1, const string &file_path2,
    const bool if_interrupt, // If enable interrupt during computation
    const bool if_verbose)
{
    #if XCPPLIB_DEBUG_MODE
    const bool if_debug = true;
    #endif
    xcpplibMsgDebugHead(if_debug);

    try { // Begin of try *****************************************************

    string file_paths[2] = {file_path1, file_path2};
    EXIT_TYPE m = mergeFiles(merged_file_path, 2, file_paths,
                             if_interrupt, if_verbose);
    if (m != EXIT_NORMAL) throw m;

    } // End of try ***********************************************************

    catch (EXIT_TYPE error_code) {
        switch (error_code) {
        case EXIT_NORMAL: break; // Do nothing
        default:
            xcpplibMsgDebugTail(if_debug);
            return error_code;
        }
    }

    xcpplibMsgDebugTail(if_debug);
    return EXIT_NORMAL;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Move source file into destination file.]                    */
/*                                                                           */
/*  Description [On success, it returns true. Abort, otherwise.]             */
/*                                                                           */
/*===========================================================================*/

    EXIT_TYPE
moveFile(
    const string &dst, const string &src, const bool if_verbose)
{
    const string move_cmd = "mv -f " + src + " " + dst;
    if (if_verbose) msg(2, "moveFile(): %s", move_cmd.c_str());
    ensure(system(move_cmd.c_str()) == 0,
        "Failed to move file from \"" + src + "\" to \"" + dst + "\".");

    return EXIT_NORMAL;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Copy source file into destination file.]                    */
/*                                                                           */
/*  Description [On success, it returns true. Abort, otherwise.]             */
/*                                                                           */
/*===========================================================================*/

    EXIT_TYPE
copyFile(
    const string &dst, const string &src, const bool if_verbose)
{
    const string copy_cmd = "cp -f " + src + " " + dst;
    if (if_verbose) msg(2, "copyFile(): %s", copy_cmd.c_str());
    ensure(system(copy_cmd.c_str()) == 0,
        "Failed to copy file from \"" + src + "\" to \"" + dst + "\".");

    return EXIT_NORMAL;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Append (Cat) source file into destination file.]            */
/*                                                                           */
/*  Description [On success, it returns true. Abort, otherwise.]             */
/*                                                                           */
/*===========================================================================*/

    EXIT_TYPE
appendFile(
    const string &dst, const string &src, const bool if_verbose)
{
    const string append_cmd = "cat " + src + " >> " + dst;
    if (if_verbose) msg(2, "appendFile(): %s", append_cmd.c_str());
    ensure(system(append_cmd.c_str()) == 0,
        "Failed to append file from \"" + src + "\" to \"" + dst + "\".");

    return EXIT_NORMAL;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Remove everything under the given directory.]               */
/*                                                                           */
/*  Description [On success, it returns true. Abort, otherwise.]             */
/*                                                                           */
/*===========================================================================*/

    EXIT_TYPE
cleanDir(const string &dir_path, const bool if_verbose)
{
    // Skip cleaning if the directory doesn't exist.
    if (checkDirPath(dir_path) != EXIT_NORMAL) return EXIT_NORMAL;

    string clean_cmd;
    clean_cmd = "rm -rf " + dir_path + "/*";
    if (if_verbose) msg(2, "cleanDir(): %s", clean_cmd.c_str());
    ensure(system(clean_cmd.c_str()) == 0,
        "Failed to clean directory \"" + dir_path + "\".");

    return EXIT_NORMAL;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Remove directory.]                                          */
/*                                                                           */
/*  Description [On success, it returns true. Abort, otherwise.]             */
/*                                                                           */
/*===========================================================================*/

    EXIT_TYPE
removeDir(const string &dir_path, const bool if_verbose)
{
    // Skip removing if the directory doesn't exist.
    if (checkDirPath(dir_path) != EXIT_NORMAL) return EXIT_NORMAL;

    cleanDir(dir_path, if_verbose);
    ensure(rmdir(dir_path.c_str()) == 0,
        "Failed to remove directory \"" + dir_path + "\".");

    return EXIT_NORMAL;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Remove file.]                                               */
/*                                                                           */
/*  Description [On success, it returns true. Abort, otherwise.]             */
/*                                                                           */
/*===========================================================================*/

    EXIT_TYPE
removeFile(const string &file_path, const bool if_verbose)
{
    // Skip removing if the file doesn't exist.
    if (checkFilePath(file_path) != EXIT_NORMAL) return EXIT_NORMAL;

    string clean_cmd;
    if (if_verbose) clean_cmd = "rm -vf " + file_path;
    else clean_cmd = "rm -f " + file_path;
    msg(MSG_DEBUG, "clean_cmd: %s", clean_cmd.c_str());
    ensure(system(clean_cmd.c_str()) == 0,
        "Failed to remove file \"" + file_path + "\".");

    return EXIT_NORMAL;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Grep the string in the given file.]                         */
/*                                                                           */
/*  Description [On success, it returns true. Abort, otherwise.]             */
/*                                                                           */
/*===========================================================================*/

    EXIT_TYPE
grepString(
    const string &log_file, // File to store the grepped string.
                            // Grepped string is appended.
    const string &str, const string &file_path, const bool if_verbose)
{
    ensure(checkFilePath(file_path) == EXIT_NORMAL,
        "File \"" + file_path + "\" doesn't exist.");

    string grep_cmd;
    if (if_verbose) {
        grep_cmd = "grep \"" + str + "\" " + file_path + " | tee -a " +
                   log_file;
    } else {
        grep_cmd = "grep \"" + str + "\" " + file_path + " >> " + log_file;
    }
    msg(MSG_DEBUG, "grep_cmd: %s", grep_cmd.c_str());
    ensure(system(grep_cmd.c_str()) == 0,
        "Failed to grep file \"" + file_path + "\".");

    return EXIT_NORMAL;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [The corresponding ANSI fopen function while with more error */
/*               checks.]                                                    */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    void
checkPermission(
    const char * const fn_p,
    const char * const open_mode_p,
    const int if_silent             // If not show messages
    )
{
    FILE * f_p = NULL;

    if (fn_p == NULL) {
        msg(MSG_ERROR, "Null file name pointer.");
        exit (-1);
    }

    if (!if_silent) {
        msg(0, "Checking the permission on file %s ... ", fn_p);
    }

    f_p = fopen(fn_p, open_mode_p);
    if (f_p == NULL) {
        if (!if_silent) {
            msg(0, "failed.\n");
        } else {
            msg(MSG_ERROR, "Checking the permission on file %s ... failed.", fn_p);
        }
        msg(MSG_ERROR, "Please check file path.");
        exit (-1);
    }
    if (!if_silent) msg(0, "succeeded.");

    fclose(f_p);
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Compare two arrays of floating-point numbers.]              */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    EXIT_TYPE
compareArray(
    const float *data1, const float *data2, const int ele_num,
    const float precision)
{
    //printf("precision: %f\n", precision);
    for (int i = 0; i < ele_num; i++) {
        if (abs(data1[i] - data2[i]) > precision) {
            return EXIT_FAIL_EXECUTION;
        }
    }
    return EXIT_NORMAL;
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    []                                                           */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

    void
printDiff(
    const float *data1, const float *data2, const int ele_num,
    const float precision)
{
    int i;
    int error_count=0;
    for (i=0; i<ele_num; i++) {
        if (abs(data1[i] - data2[i]) > precision && error_count < 10) {
            printf("diff(%d,%d) CPU=%4.4f, GPU=%4.4f\n",
                i, i, data1[i], data2[i]);
        }
        if (data1[i] != data2[i]) {
            error_count++;
        }
    }

    printf("\nTotal Errors = %d\n", error_count);
}

#if 0
// ==================== isPowerOfTwo ====================
    bool 
isPowerOfTwo(int n)
{
    return ((n & (n - 1)) == 0);
}

// ==================== getLeastPowerOfTwo ====================
// Get the least number of power two that is greater than value.
// E.g., value = 8, p = 8.
//       value = 12, p = 16.
//       value = 16, p = 16.
//       value = 18, p = 32.
    int 
getLeastPowerOfTwo(const int value)
{
    int num = value;
    int exp = 0;
    frexp((float) num, &exp);
    int p = (int) pow(2.0, exp);

    // frexp may generate larger value
    if (num == (int) pow(2.0, exp - 1)) {
        p = (int) pow(2.0, exp - 1);
    }

    return p;
}

// FIXME: Should be moved to GPU source files.
// ==================== padVectorPowerOfTwo ====================
FLOAT_T *padVectorPowerOfTwo(FLOAT_T *array, const int element_num)
{
    FLOAT_T *a = NULL;

    if (!isPowerOfTwo(element_num)) {
        int size_v = getLeastPowerOfTwo(element_num);

        // For example, we must pad 3770 to 4096 for easy manipulation in GPU.
        cutilSafeMalloc(a = (FLOAT_T *) malloc(size_v * sizeof(FLOAT_T)));
        ensure(a, "Failed to allocate memory.");
        for (int i = 0; i < element_num; i++) {
            a[i] = array[i];
        }
        for (int i = element_num; i < size_v; i++) {
            a[i] = 0.0;
        }
        // Not free original array
        //free(array);

        // Input size is multiples of vectorProductGpu_BLOCK_SIZE.
    } else {
        a = array;
    }

    return a;
}
#endif

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [This function is used for testing purpose only.]            */
/*                                                                           */
/*  Description [It can be called by other functions to facilitate the       */
/*      testing process.]                                                    */
/*                                                                           */
/*===========================================================================*/

    void
xcpplibOthersTest(void)
{

}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

}

