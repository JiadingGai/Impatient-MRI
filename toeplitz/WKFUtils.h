#ifndef WKFUTILS_H
#define WKFUTILS_H
typedef void* wkf_timerhandle;            ///< a timer handle
wkf_timerhandle wkf_timer_create(void);    ///< create a timer (clears timer)
void wkf_timer_destroy(wkf_timerhandle);   ///< create a timer (clears timer)
void wkf_timer_start(wkf_timerhandle);     ///< start a timer  (clears timer)
void wkf_timer_stop(wkf_timerhandle);      ///< stop a timer
double wkf_timer_time(wkf_timerhandle);    ///< report elapsed time in seconds
double wkf_timer_timenow(wkf_timerhandle); ///< report elapsed time in seconds
double wkf_timer_start_time(wkf_timerhandle); ///< report wall starting time
double wkf_timer_stop_time(wkf_timerhandle); ///< report wall stopping time

typedef struct {
  wkf_timerhandle timer;
  double updatetime;
} wkfmsgtimer;

/// initialize periodic status message timer
extern wkfmsgtimer * wkf_msg_timer_create(double updatetime);

/// return true if it's time to print a status update message
extern int wkf_msg_timer_timeout(wkfmsgtimer *time);

/// destroy message timer
void wkf_msg_timer_destroy(wkfmsgtimer * mt);
#endif
