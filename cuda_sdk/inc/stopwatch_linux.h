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

/* CUda UTility Library */

#ifndef _STOPWATCH_LINUX_H_
#define _STOPWATCH_LINUX_H_

// includes, system
#include <ctime>
#include <sys/time.h>

//! Windows specific implementation of StopWatch
class StopWatchLinux {

protected:

    //! Constructor, default
    StopWatchLinux();

    // Destructor
    ~StopWatchLinux();

public:

    //! Start time measurement
    inline void start();

    //! Stop time measurement
    inline void stop();

    //! Reset time counters to zero
    inline void reset();

    //! Time in msec. after start. If the stop watch is still running (i.e. there
    //! was no call to stop()) then the elapsed time is returned, otherwise the
    //! time between the last start() and stop call is returned
    inline const float getTime() const;

    //! Mean time to date based on the number of times the stopwatch has been 
    //! _stopped_ (ie finished sessions) and the current total time
    inline const float getAverageTime() const;

private:

    // helper functions
  
    //! Get difference between start time and current time
    inline float getDiffTime() const;

private:

    // member variables

    //! Start of measurement
    struct timeval  start_time;

    //! Time difference between the last start and stop
    float  diff_time;

    //! TOTAL time difference between starts and stops
    float  total_time;

    //! flag if the stop watch is running
    bool running;

    //! Number of times clock has been started
    //! and stopped to allow averaging
    int clock_sessions;
};

// functions, inlined

////////////////////////////////////////////////////////////////////////////////
//! Start time measurement
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchLinux::start() {

  gettimeofday( &start_time, 0);
  running = true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop time measurement and increment add to the current diff_time summation
//! variable. Also increment the number of times this clock has been run.
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchLinux::stop() {

  diff_time = getDiffTime();
  total_time += diff_time;
  running = false;
  clock_sessions++;
}

////////////////////////////////////////////////////////////////////////////////
//! Reset the timer to 0. Does not change the timer running state but does 
//! recapture this point in time as the current start time if it is running.
////////////////////////////////////////////////////////////////////////////////
inline void
StopWatchLinux::reset() 
{
  diff_time = 0;
  total_time = 0;
  clock_sessions = 0;
  if( running )
    gettimeofday( &start_time, 0);
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. after start. If the stop watch is still running (i.e. there
//! was no call to stop()) then the elapsed time is returned added to the 
//! current diff_time sum, otherwise the current summed time difference alone
//! is returned.
////////////////////////////////////////////////////////////////////////////////
inline const float 
StopWatchLinux::getTime() const 
{
    // Return the TOTAL time to date
    float retval = total_time;
    if( running) {

        retval += getDiffTime();
    }

    return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. for a single run based on the total number of COMPLETED runs
//! and the total time.
////////////////////////////////////////////////////////////////////////////////
inline const float 
StopWatchLinux::getAverageTime() const
{
    return total_time/clock_sessions;
}



////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
inline float
StopWatchLinux::getDiffTime() const 
{
  struct timeval t_time;
  gettimeofday( &t_time, 0);

  // time difference in milli-seconds
  return  (float) (1000.0 * ( t_time.tv_sec - start_time.tv_sec) 
                + (0.001 * (t_time.tv_usec - start_time.tv_usec)) );
}

#endif // _STOPWATCH_LINUX_H_

