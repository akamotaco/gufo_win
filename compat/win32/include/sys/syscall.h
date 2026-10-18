/* Win32 shadow of <sys/syscall.h>; see gufo_posix.h. */
#pragma once
#include "gufo_posix.h"
/* No memfd on Windows: syscall(SYS_memfd_create) fails with ENOSYS. */
#define SYS_memfd_create 319
/* syscall(SYS_renameat2, olddirfd, oldpath, newdirfd, newpath, flags):
 * RENAME_NOREPLACE (<linux/fs.h>) is honored; other flags are EINVAL. */
#define SYS_renameat2 316
#ifdef __cplusplus
extern "C"
#endif
long syscall(long number, ...);
