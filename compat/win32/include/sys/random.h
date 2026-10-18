/* Win32 shadow of <sys/random.h>; getrandom() is BCryptGenRandom. */
#pragma once
#include "gufo_posix.h"
#define GRND_NONBLOCK 0x01
#define GRND_RANDOM 0x02
#ifdef __cplusplus
extern "C"
#endif
ssize_t getrandom(void* buffer, size_t length, unsigned int flags);
