/* Win32 shadow of <dirent.h>; see gufo_posix.h. Directory streams are
 * implemented in compat/win32/posix.cpp over FindFirstFileExW. */
#pragma once
#include "gufo_posix.h"

#define DT_UNKNOWN 0
#define DT_DIR 4
#define DT_REG 8
#define DT_LNK 10

struct dirent {
  unsigned char d_type;
  char d_name[1024]; /* UTF-8 */
};
typedef struct gufo_dir DIR;

#ifdef __cplusplus
extern "C" {
#endif
DIR* opendir(const char* path);
/* Takes ownership of `fd`: closedir() closes it. */
DIR* fdopendir(int fd);
struct dirent* readdir(DIR* dir);
int closedir(DIR* dir);
#ifdef __cplusplus
}
#endif
