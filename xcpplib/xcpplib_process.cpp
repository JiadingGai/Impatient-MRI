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

    File Name   [xcpplib_process.cpp]

    Synopsis    [Error checking and exception handling.]

    Description [See the corresponding header file for details.]

    Revision    [0.1; Initial build; Xiao-Long Wu, ECE UIUC]
    Date        [01/25/2009]

 *****************************************************************************/

/*---------------------------------------------------------------------------*/
/*  Included library headers                                                 */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cassert>     // for assert()
#include <cstring>
#include <unistd.h>     // for keyboard input manipulation
#include <termios.h>    // for keyboard input manipulation
#include <poll.h>       // for keyboard input manipulation

// XCPPLIB libraries
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
/*  Class/Function implementations                                           */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Show messages in a professional way.]                       */
/*                                                                           */
/*  Description [Printf wrapper.]                                            */
/*                                                                           */
/*  FIXME       [Should use C++ class to avoid file opening/closing on the   */
/*      calling side. It's better to use file names than file pointers.      */
/*      Also a global object can be defined to avoid object definition.]     */
/*                                                                           */
/*===========================================================================*/

static string msg_header = "";
static FILE *msg_out_fp = stdout;
static FILE *msg_err_fp = stderr;
static FILE *msg_log_fp = NULL;
static bool msg_debug_mode = false;

// Set this to show program header before every message.
// You should setup the message header in the very beginning of the program.
    void
msgSetHeader(const string &header)
{
    ensure(header != "", "Empty message header.");
    msg(MSG_DEBUG, "Set message header: \"%s\"", header.c_str());
    msg_header = header + ": ";
}

// Set the output file pointer. Default is stdout.
    void
msgSetOutput(FILE *fp)
{
    ensure(fp != NULL, "Null message output pointer.");
    msg_out_fp = fp;
}

// Set the output log file pointer. Default is NULL.
// You should setup this in the very beginning if you want to list everything.
    void
msgSetLog(FILE *fp)
{
    msg_log_fp = fp;
}

// Set this in the beginning of each function to enable debugging output.
    void
msgSetDebug(const bool &d)
{
    msg_debug_mode = d;
}

    bool
msgGetDebug(void)
{
    return msg_debug_mode;
}

// The main message output routine.
    void
msg(const int level, const char *format, ...)
{
    ensure(level < 12, "Given message level is too big.");

    if (level == MSG_DEBUG && !msg_debug_mode) return;

    // Note: This number can be too small for some huge messages.
    //       Make sure you find a good balance.
    const unsigned int MSG_MAX_SIZE = 8192;

    // Fill the contents of msg_buf.
    static char msg_buf[MSG_MAX_SIZE] = "\0";
    msg_buf[MSG_MAX_SIZE-1] = '\0';
    va_list arg;
    va_start(arg, format);
    vsprintf(msg_buf, format, arg);
    va_end(arg);
    // Message is too big to print out.
    // FIXME: Is there a better way to show errors? All the printing functions
    // failed after msg_buf is blown out.
    assert(msg_buf[MSG_MAX_SIZE-1] == '\0');

    // Combine the message header and the rest together into out_buf.
    static char out_buf[MSG_MAX_SIZE+1024] = "\0";
    switch (level) {
    case MSG_PLAIN:
        sprintf(out_buf, "%s", msg_buf);
        break;
    case MSG_DEBUG: // Enable debugging messages on debug mode.
                    // msg(MSG_DEBUG, ...) will be void on other modes.
        if (msg_debug_mode) {
            sprintf(out_buf, "%sDebug: %s\n", msg_header.c_str(), msg_buf);
        }
        break;
    case MSG_WARNING: // Warning caused by users
        sprintf(out_buf, "%sWarning: %s\n", msg_header.c_str(), msg_buf);
        break;
    case MSG_ERROR: // Error caused by users
        sprintf(out_buf, "%sError: %s\n", msg_header.c_str(), msg_buf);
        break;
    case MSG_INTERNAL_ERROR: // Error caused by programmers
        sprintf(out_buf, "%sInternal Error: %s\n", msg_header.c_str(), msg_buf);
        break;
    default: // NORMAL: Level >= 0
        char *out_buf_ptr = out_buf;
        if (strlen(msg_buf) > 0) {
            for (int i = 0; i < level; i++) {
                sprintf(out_buf_ptr, "  "); out_buf_ptr += 2;
            }
            // Only print out header when level is 0 or 1.
            if (level < 2) {
                sprintf(out_buf_ptr, "%s", msg_header.c_str());
                out_buf_ptr += strlen(msg_header.c_str());
            } else
            for (unsigned int i = 0; i < strlen(msg_header.c_str()); i++) {
                sprintf(out_buf_ptr, " "); out_buf_ptr += 1;
            }
            sprintf(out_buf_ptr, "%s\n", msg_buf);
            out_buf_ptr += strlen(msg_buf);
        } else {
            sprintf(out_buf_ptr, "\n"); out_buf_ptr += 1;
        }
    } // End of switch (level)

    // Set the output place.
    static FILE *fp = NULL;
    switch (level) {
    case MSG_INTERNAL_ERROR: fp = msg_err_fp; break;
    case MSG_ERROR:          fp = msg_err_fp; break;
    case MSG_WARNING:        fp = msg_err_fp; break;
    case MSG_DEBUG:          fp = msg_out_fp; break;
    case MSG_PLAIN:          fp = msg_out_fp; break;
    default:                 fp = msg_out_fp;
    }

    fprintf(fp, "%s", out_buf); fflush(fp);

    // If the additional message log is enabled, print out.
    if (msg_log_fp != NULL) {
        fprintf(msg_log_fp, "%s", out_buf); fflush(msg_log_fp);
    }

    // Clean the message buffers for the next run.
    msg_buf[0] = '\0'; out_buf[0] = '\0';
}

/*---------------------------------------------------------------------------*/
/*  Functions for error message printing and exitting.                       */
/*  Note: Use true/false for bool type arguments.                            */
/*---------------------------------------------------------------------------*/

// Function definition =======================================================

    void
outputMsg(const string &msg, bool if_exit)
{
    if (if_exit) {
        cerr<< endl<< "***Fatal error: "<< msg<< endl;
        if (XCPPLIB_ENABLE_GDB_TRACE) { assert(0); }
        else exit(1);
    } else {
        outputWarn(msg);
        //cerr<< "***Warning: "<< msg<< endl;
    }
}
    void
outputMsg(const string &msg,
    const string &fn,           // file name of the error
    const int &line,            // line number of the error
    bool if_exit)
{
    if (if_exit) {
        cerr<< endl<< "***Fatal error: "<< msg<< endl;
        if (line != 0)
            cerr<< "Internal error at file \""<< fn<< "\", line "<< line
                << ".\n";
        if (XCPPLIB_ENABLE_GDB_TRACE) { assert(0); }
        else exit(1);
    } else {
        outputWarn(msg);
        //cerr<< "***Warning: "<< msg<< endl;
        if (line != 0)
            cerr<< "Internal warning at file \""<< fn<< "\", line "<< line
                << ".\n";
    }
}
    void
outputMsg(const string &fn, const int &line, bool if_exit)
{
    const string msg = "Unknown error.";
    if (if_exit) {
        cerr<< endl<< "***Fatal error: "<< msg<< endl;
        if (line != 0)
            cerr<< "Internal error at file \""<< fn<< "\", line "<< line
                << ".\n";
        if (XCPPLIB_ENABLE_GDB_TRACE) { assert(0); }
        else exit(1);
    } else {
        outputWarn(msg);
        //cerr<< "***Warning: "<< msg<< endl;
        if (line != 0)
            cerr<< "Internal warning at file \""<< fn<< "\", line "<< line
                << ".\n";
    }
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Keyboard input manipulation during the computation process.]*/
/*                                                                           */
/*  Description [See keyboardInputTest() for usage.]                         */
/*                                                                           */
/*  Note        [The following code snippit is revised from the discussion   */
/*      of the web site: http://www.cplusplus.com/forum/general/5304/        */
/*      Which is provided by Duoas. All rights are reserved by the original  */
/*      author.]                                                             */
/*                                                                           */
/*===========================================================================*/

// The first function to call before starting the keyboard input checking
    bool
keyboardInputCheckBegin(const string key) // Only the first two chars are used.
{
    try {
        bool r = false;
        r = keyboardInputInitialize(key);
        if (!r) throw r;
        r = keyboardInputLineBuffered(false);
        if (!r) throw r;
        r = keyboardInputEcho(false);
        if (!r) throw r;
    }
    catch (bool result) {
        return result;
    }

    return true;
}

// The last function to call to end the keyboard input checking
    bool
keyboardInputCheckEnd(void)
{
    try {
        bool r = false;
        r = keyboardInputLineBuffered();
        if (!r) throw r;
        r = keyboardInputEcho();
        if (!r) throw r;
        keyboardInputFinalize();
    }
    catch (bool result) {
        return result;
    }

    return true;
}

static bool keyboard_input_initialized = false;
static struct termios keyboard_input_initial_settings;
static string keyboard_input_key = "cC";

    bool
keyboardInputInitialize(const string key) // Only the first two chars are used.
{
    if (!keyboard_input_initialized) {
        keyboard_input_key = key;
        keyboard_input_initialized = (bool)isatty(STDIN_FILENO);
        if (keyboard_input_initialized) {
            keyboard_input_initialized = (0 == tcgetattr(STDIN_FILENO,
                                          &keyboard_input_initial_settings));
        }
        if (keyboard_input_initialized) std::cin.sync_with_stdio();
    }

    return keyboard_input_initialized;
}

    void
keyboardInputFinalize()
{
    ensure(keyboard_input_initialized,
        "keyboardInputInitialize() should be called first.");

    if (keyboard_input_initialized) {
        tcsetattr(STDIN_FILENO, TCSANOW, &keyboard_input_initial_settings);
        keyboard_input_initialized = false;
    }
}

// Decide if the inputs are buffered or not ("on" is false).
// The buffered input can be used later by keyboardInputKeyPressed().
    bool
keyboardInputLineBuffered(const bool on) // default: true
{
    struct termios settings;

    ensure(keyboard_input_initialized,
        "keyboardInputInitialize() should be called first.");

    if (tcgetattr(STDIN_FILENO, &settings)) return false;

    if (on) settings.c_lflag |= ICANON;
    else settings.c_lflag &= ~(ICANON);

    if (tcsetattr(STDIN_FILENO, TCSANOW, &settings)) return false;

    if (on) setlinebuf(stdin);  // Output is line-buffered.
    else setbuf(stdin, NULL);   // Input/Output is set unbuffered.
                                // (Print to screen directly).

    return true;
}

// Decide if the inputs are displayed to the screen or not ("on" is false).
    bool
keyboardInputEcho(const bool on) // default: true
{
    struct termios settings;

    ensure(keyboard_input_initialized,
        "keyboardInputInitialize() should be called first.");

    if (tcgetattr(STDIN_FILENO, &settings)) return false;

    if (on) settings.c_lflag |= ECHO;
    else settings.c_lflag &= ~(ECHO);

    return 0 == tcsetattr(STDIN_FILENO, TCSANOW, &settings);
}

// Is this used somewhere in the standard library?
#define INFINITE (-1)

    bool
keyboardInputKeyPressed(
    const unsigned timeout_ms) // Period of time to poll. Default: 0 ms.
{
    makeSure(keyboard_input_initialized,
        "keyboardInputInitialize() should be called first.", false);

    if (keyboard_input_initialized) {
        struct pollfd pls[1];
        pls[0].fd     = STDIN_FILENO;
        pls[0].events = POLLIN | POLLPRI;
        if (poll(pls, 1, timeout_ms) > 0 && 
            (cin.get() == keyboard_input_key.c_str()[0] ||
             cin.get() == keyboard_input_key.c_str()[1])) {
            return true;
        }
    }

    return false;
}

    EXIT_TYPE
keyboardInputTest()
{
    if (!keyboardInputInitialize()) {
        cout << "You must be a human to use this program.\n";
        return EXIT_WRONG_INPUTS;
    }

    try {
        string name, password;

        keyboardInputLineBuffered(false); // Turn off the input buffering.
        keyboardInputEcho(false); // Turn off the input displaying.

        cout << "Press any key to wake me up.\n";
        while (true) {
            cout << "Zzz..." << flush;
            if (keyboardInputKeyPressed(500)) break;
        }
        if (cin.get() == '\n') cout << "\nHmm... Not much of an 'any' key"
            " test when you press ENTER...\n";
        else cout << "\nThanks!\n";

        keyboardInputEcho(); // Turn on the input displaying.
        keyboardInputLineBuffered(); // Turn on the input buffering.
        cout << "\nPlease enter your name> " << flush;
        getline(cin, name);

        cout << "Please enter a (fake) password> " << flush;
        keyboardInputEcho(false);
        getline(cin, password);
        keyboardInputEcho();
        cout << "\nSo, your password is \"" << password << "\", hmm?\n";

        // http://www.cppreference.com/cppio/ignore.html
        // The ignore() function is used with input streams. It reads and
        // throws away characters until num characters have been read (where
        // num defaults to 1) or until the character delim is read (where delim
        // defaults to EOF).
        cout << "\nPress ENTER to quit> " << flush;
        cin.ignore(numeric_limits <streamsize> ::max(), '\n');

        cout << "\nGood-bye " << name << ".\n";
    }
    catch (...) { }

    keyboardInputFinalize();
    return EXIT_NORMAL;
}

/*---------------------------------------------------------------------------*/
/*  Class Indents member implementations                                     */
/*---------------------------------------------------------------------------*/

    string
Indents::str(void) const
{
    string spaces;
    for (int i = 0; i < space_num; i++) {
        spaces += " ";
    }
    return spaces;
}

    ostream &
Indents::print(ostream &out) const
{
    string spaces;
    for (int i = 0; i < space_num; i++) {
        spaces += " ";
    }
    return out<< spaces;
}

    ostream &
operator<<(ostream &out, const Indents &ind)
{
    return ind.print(out);
}

    void
Indents::test(ostream &out) const
{
    Indents ind0, ind2(2), ind4(2);
    ind4 += 2;

    out<< "Indents: \n";
    out<< "ind0: \""<< ind0<< "\"\n";
    out<< "ind2: \""<< ind2<< "\"\n";
    out<< "ind4: \""<< ind4<< "\"\n";
    out<< "ind0++: \""<< ind0++<< "\"\n";
    out<< "++ind0: \""<< ++ind0<< "\"\n";
    ind2 += 4;
    out<< "ind2+=4: \""<< ind2<< "\"\n";
    ind4 -= 2;
    out<< "ind4-=2: \""<< ind4<< "\"\n";
}

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [This function is used for testing purpose only.]            */
/*                                                                           */
/*  Description [It can be called by other functions to facilitate the       */
/*      testing process.]                                                    */
/*                                                                           */
/*===========================================================================*/

    void
xcpplibProcessTest(void)
{

}

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

}

