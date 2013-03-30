

#include "WKFUtils.h"

#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_MSC_VER)
#include <windows.h>
#include <conio.h>
#else
#include <unistd.h>
#include <sys/time.h>
#include <errno.h>

#if defined(ARCH_AIX4)
#include <strings.h>
#endif

#if defined(__irix)
#include <bstring.h>
#endif

#if defined(__hpux)
#include <time.h>
#endif // HPUX
#endif // _MSC_VER


#if defined(_MSC_VER)
typedef struct {
  DWORD starttime;
  DWORD endtime;
} wkf_timer;


void wkf_timer_start(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  t->starttime = GetTickCount();
}

void wkf_timer_stop(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  t->endtime = GetTickCount();
}

double wkf_timer_time(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  double ttime;

  ttime = ((double) (t->endtime - t->starttime)) / 1000.0;

  return ttime;
}

double wkf_timer_start_time(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  double ttime;
  ttime = ((double) (t->starttime)) / 1000.0;
  return ttime;
}

double wkf_timer_stop_time(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  double ttime;
  ttime = ((double) (t->endtime)) / 1000.0;
  return ttime;
}

#else

// Unix with gettimeofday()
typedef struct {
  struct timeval starttime, endtime;
  struct timezone tz;
} wkf_timer;

void wkf_timer_start(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  gettimeofday(&t->starttime, &t->tz);
}

void wkf_timer_stop(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  gettimeofday(&t->endtime, &t->tz);
}

double wkf_timer_time(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  double ttime;
  ttime = ((double) (t->endtime.tv_sec - t->starttime.tv_sec)) +
          ((double) (t->endtime.tv_usec - t->starttime.tv_usec)) / 1000000.0;
  return ttime;
}

double wkf_timer_start_time(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  double ttime;
  ttime = ((double) t->starttime.tv_sec) +
          ((double) t->starttime.tv_usec) / 1000000.0;
  return ttime;
}

double wkf_timer_stop_time(wkf_timerhandle v) {
  wkf_timer * t = (wkf_timer *) v;
  double ttime;
  ttime = ((double) t->endtime.tv_sec) +
          ((double) t->endtime.tv_usec) / 1000000.0;
  return ttime;
}

#endif

// system independent routines to create and destroy timers
wkf_timerhandle wkf_timer_create(void) {
  wkf_timer * t;
  t = (wkf_timer *) malloc(sizeof(wkf_timer));
  memset(t, 0, sizeof(wkf_timer));
  return t;
}

void wkf_timer_destroy(wkf_timerhandle v) {
  free(v);
}

double wkf_timer_timenow(wkf_timerhandle v) {
  wkf_timer_stop(v);
  return wkf_timer_time(v);
}

/// initialize status message timer
wkfmsgtimer * wkf_msg_timer_create(double updatetime) {
  wkfmsgtimer *mt;
  mt = (wkfmsgtimer *) malloc(sizeof(wkfmsgtimer));
  if (mt != NULL) {
    mt->timer = wkf_timer_create();
    mt->updatetime = updatetime;
    wkf_timer_start(mt->timer);
  }
  return mt;
}

/// return true if it's time to print a status update message
int wkf_msg_timer_timeout(wkfmsgtimer *mt) {
  double elapsed = wkf_timer_timenow(mt->timer);
  if (elapsed > mt->updatetime) {
    // reset the clock and return true that our timer expired
    wkf_timer_start(mt->timer);
    return 1;
  } else if (elapsed < 0) {
    // time went backwards, best reset our clock!
    wkf_timer_start(mt->timer);
  }
  return 0;
}

/// destroy message timer
void wkf_msg_timer_destroy(wkfmsgtimer * mt) {
  wkf_timer_destroy(mt->timer);
  free(mt);
}

