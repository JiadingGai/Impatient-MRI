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

    File Name   [xcpplib_process.h]

    Synopsis    [Error checking and exception handling.]

    Description [This file defines error checking and exception handling
        procedures. This file shall be used right after xcpplib_global.h and
        before other source files.]

    Revision    [0.1; Initial build; Xiao-Long Wu, ECE UIUC]
    Date        [01/25/2009]

 *****************************************************************************/

#ifndef XCPPLIB_PROCESS_H
#define XCPPLIB_PROCESS_H

/*---------------------------------------------------------------------------*/
/*  Included libraries from standard libraries                               */
/*---------------------------------------------------------------------------*/

// System libraries
#include <iostream>
#include <cstdarg>
#include <vector>
#include <limits>       // for keyboard input manipulation (numeric_limits())

// XCPPLIB libraries
#include <xcpplib_global.h>

/*---------------------------------------------------------------------------*/
/*  Namespace declared - begin                                               */
/*---------------------------------------------------------------------------*/

namespace xcpplib {

/*---------------------------------------------------------------------------*/
/*  Namespace used                                                           */
/*---------------------------------------------------------------------------*/

using namespace std;

/*---------------------------------------------------------------------------*/
/*  Macros and types                                                         */
/*---------------------------------------------------------------------------*/

enum EXIT_TYPE {
    EXIT_NORMAL         =  0,   // This must be zero by convention.
    EXIT_USER_INTERRUPT = -1,   // Errors must be negative.
    EXIT_WRONG_INPUTS   = -2,   // Error caused by users.
    EXIT_FAIL_EXECUTION = -3,   // Error caused by users.
    EXIT_INTERNAL_ERROR = -4};  // Error caused by programmers.

/*---------------------------------------------------------------------------*/
/*  Function prototypes                                                      */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*  Functions for message printing.                                          */
/*---------------------------------------------------------------------------*/

const int MSG_PLAIN          = -1; // No program header is provided.
const int MSG_DEBUG          = -2; // Debug mode message printing.
                                   // Which is disabled at release mode.
const int MSG_WARNING        = -3; // Warning caused by users.
const int MSG_ERROR          = -4; // Error caused by users.
const int MSG_INTERNAL_ERROR = -5; // Error caused by programmers.

// You should setup the message header in the very beginning of the program.
    void
msgSetHeader(const string &header);

    void
msgSetOutput(FILE *fp);

// You should setup the message header in the very beginning of the program.    
    void
msgSetLog(FILE *fp);

    void
msgSetDebug(const bool &d);

    bool
msgGetDebug(void);

    void
msg(const int level, const char *format, ...);

// You can put the following lines in your program to disable debugging message
// functions when performance is an issue.
// ============================================================================

#if 0
    #if ENABLE_DEBUG_MSG
        #define msgDebug(args...)       msg(MSG_DEBUG, ##args)
    #else
        #define msgDebug(args...)
        #undef msgDebugHead
        #define msgDebugHead(DEBUG_VARIABLE)
        #undef msgDebugTail
        #define msgDebugTail(DEBUG_VARIABLE)
    #endif
#endif

// Enable the debugging message printing within each function scope.
// msgDebugHead() and msgDebugTail() must be paired.
// The DEBUG_VARIABLE must be a variable instead of a macro.
// For XCPPLIB internal use, xcpplibMsgDebugHead() xcpplibMsgDebugTail() are
// defined in xcpplib_global.h.
// ============================================================================

#define msgDebugHead(DEBUG_VARIABLE) \
    bool pre_msg_debug_mode; \
    if (DEBUG_VARIABLE) { \
        pre_msg_debug_mode = msgGetDebug(); \
        msgSetDebug(true); \
    } else { \
        pre_msg_debug_mode = msgGetDebug(); \
        msgSetDebug(false); \
    }

#define msgDebugTail(DEBUG_VARIABLE) \
    msgSetDebug(pre_msg_debug_mode);

/*---------------------------------------------------------------------------*/
/*  Functions for error message printing and exitting.                       */
/*  Note: Use true/false for bool type arguments.                            */
/*---------------------------------------------------------------------------*/

    inline void
outputWarn(const string &msg)
{
    cerr<< "***Warning: "<< msg<< endl;
}
    void
outputMsg(const string &msg, bool if_exit = true);

    void
outputMsg(const string &msg,
    const string &fn, const int &line, bool if_exit = true);

    void
outputMsg(const string &fn, const int &line, bool if_exit = true);

#define message(msg) outputMsg(msg, __FILE__, __LINE__);

/*
 * For boolean status check
 */

/*
 * To replace assert() while providing more friendly options.
 */

// For boolean status check

    inline void
makeSure(bool check, const string &msg, bool if_exit = true)
{
    if (check == false) {
        if (!if_exit) outputWarn(msg);
        else outputMsg(msg, true);
    }
}
// Used internally for programmers.
    inline void
makeSure(bool check, const string &msg,
    const string &fn, const int &line, bool if_exit = true)
{
    if (check == false) { outputMsg(msg, fn, line, if_exit); }
}
// Used internally for programmers.
    inline void
makeSure(bool check, const string &fn, const int &line, bool if_exit = true)
{
    if (check == false) { outputMsg(fn, line, if_exit); }
}

// For pointer NULL check

    inline void
makeSure(const void * check, const string &msg, bool if_exit = true)
{
    if (check == NULL) {
        if (!if_exit) outputWarn(msg);
        else outputMsg(msg, true);
    }
}
// Used internally for programmers.
    inline void
makeSure(const void * check, const string &msg,
    const string &fn, const int &line, bool if_exit = true)
{
    if (check == NULL) { outputMsg(msg, fn, line, if_exit); }
}
// Used internally for programmers.
    inline void
makeSure(const void * check, const string &fn, const int &line,
    bool if_exit = true)
{
    if (check == NULL) { outputMsg(fn, line, if_exit); }
}

// Used internally for programmers.
// Note: These can cause slow execution when this is called many times.
#define ensure(check, msg) makeSure(check, msg, __FILE__, __LINE__);
#define warn(check, msg)   makeSure(check, msg, false);

/*---------------------------------------------------------------------------*/
/*  Keyboard input manipulation during the computation process.              */
/*---------------------------------------------------------------------------*/

// The first function to call before starting the keyboard input checking
// Only the first two chars are used.
    bool
keyboardInputCheckBegin(const string key = "cC");

// The last function to call to end the keyboard input checking
    bool
keyboardInputCheckEnd(void);

// Only the first two chars are used.
    bool
keyboardInputInitialize(const string key = "cC");

    void
keyboardInputFinalize();

    bool
keyboardInputLineBuffered(const bool on = true);

    bool
keyboardInputEcho(const bool on = true);

    bool
keyboardInputKeyPressed(const unsigned timeout_ms = 0);

/*---------------------------------------------------------------------------*/
/*  Class/Type implementations                                               */
/*---------------------------------------------------------------------------*/

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Message indentation class.]                                 */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*===========================================================================*/

class Indents
{
private:
    int space_num;
    int tab_stop;

public:
    // Constructors and destructors
    // ============================

    Indents(const int s = 0, const int t = 2) {
        space_num = s;
        tab_stop = t;
    }

    Indents(const Indents &copy) {
        space_num = copy.space_num;
        tab_stop = copy.tab_stop;
    }

    ~Indents() {};

    // Main functions to do the identation
    // ===================================

        string
    str(void) const;
        ostream &
    print(ostream &out) const;

        void
    operator =(const Indents &ind) {
        space_num = ind.space_num;
        tab_stop = ind.tab_stop;
    }

        Indents
    operator +(Indents &ind) const {
        int new_space_num = space_num + ind.space_num;
        return Indents(new_space_num, tab_stop);
    }

        Indents
    operator +(const int t) const {
        int new_space_num = space_num + t*tab_stop;
        return Indents(new_space_num, tab_stop);
    }

        void
    operator +=(const int tab) {
        space_num = space_num + tab;
    }

        void
    operator -=(const int tab) {
        space_num = space_num - tab;
    }

        Indents
    operator -(Indents &ind) const {
        int new_space_num = space_num - ind.space_num;
        if (new_space_num < 0) new_space_num = 0;
        return Indents(new_space_num);
    }

    // Prefix ++, e.g., ++Ind. This increases the space_num by tab_stop.
        Indents
    operator ++(void) {
        space_num = space_num + tab_stop;
        return Indents(space_num, tab_stop);
    }

    // Postfix ++, e.g., Ind++. This increases the space_num by tab_stop.
        Indents
    operator ++(int) {
        Indents new_ind(space_num, tab_stop);
        space_num = space_num + tab_stop;
        return new_ind;
    }

    // Prefix --, e.g., --Ind. This decreases the space_num by tab_stop.
        Indents
    operator --(void) {
        space_num = space_num - tab_stop;
        if (space_num < 0) space_num = 0;
        return Indents(space_num, tab_stop);
    }

    // Postfix --, e.g., Ind--. This decreases the space_num by tab_stop.
        Indents
    operator --(int) {
        Indents new_ind(space_num, tab_stop);
        space_num = space_num - tab_stop;
        if (space_num < 0) space_num = 0;
        return new_ind;
    }

        friend ostream &
    operator<<(ostream &out, const Indents &ind);

public:
    // Test function
    // =============

        void
    test(ostream &out) const;
};

/*===========================================================================*/
/*                                                                           */
/*  Synopsis    [Messenger output handling class.]                           */
/*                                                                           */
/*  Description []                                                           */
/*                                                                           */
/*  Usage       [                                                            */
/*      For standard output stream: Messenger xlog(MESSENGER_STREAM::COUT);  */
/*      For standard error stream: Messenger xlog(MESSENGER_STREAM::CERR);   */
/*               ]                                                           */
/*                                                                           */
/*===========================================================================*/

namespace MESSENGER_STREAM {
    enum T {
        COUT = 1 << 0,      // standard output stream
        CERR = 1 << 1,      // standard error stream
        DUMMY = 1 << 2      // empty stream
    };
}

// A class of null stream output
class NullStream: public ostream
{
public:
    NullStream(): ostream(0) { /* empty */ };
};

// FIXME: This class got problems.
class Messenger: public ostream
{
public:
    // Constructors and destructors
    // ============================

    Messenger(const MESSENGER_STREAM::T &m = MESSENGER_STREAM::COUT) {
        setMode(m);
        switch (m) {
        case MESSENGER_STREAM::COUT: addStream(cout); break;
        case MESSENGER_STREAM::CERR: addStream(cerr); break;
        case MESSENGER_STREAM::DUMMY: {
            static NullStream dummy;
            addStream(dummy); break;
        }
        default: message("Undefined MESSAGE_STREAM type.");
        }
    }

    // Member functions
    // ================

        int
    addStream(ostream &s) {
        streams.push_back(&s);
        return streams.size() - 1;
    }

        void
    setMode(int m) { mode = m; }

    // FIXME: Not verified yet.
        void
    activateStream(int index) { mode |= 1 << index; }

    // FIXME: Not verified yet.
        void
    deactivateStream(int index) { mode &= ~(1 << index); }

        template<class T> Messenger &
    operator << (const T &msg) {
        int flags = 1;
        for (unsigned int i = 0; i < streams.size(); i++) {
            if (flags & mode) (*streams[i]) << msg;
            flags <<= 1;
        }
        return *this;
    }

private:
    int mode; // Output stream type. E.g., (MESSENGER_STREAM::COUT) or
              // (MESSENGER_STREAM::COUT | MESSENGER_STREAM::CERR).
    vector<ostream *> streams;
};

/*---------------------------------------------------------------------------*/
/*  Testing functions                                                        */
/*---------------------------------------------------------------------------*/

    void
xcpplibProcessTest(void);

/*---------------------------------------------------------------------------*/
/*  Namespace declared - end                                                 */
/*---------------------------------------------------------------------------*/

}

#endif // XCPPLIB_PROCESS_H

