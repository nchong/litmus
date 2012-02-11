#ifndef LOG_H
#define LOG_H

#include <cstdlib>
#include <cstdio>

enum LogLevel {
  LOG_FATAL,
  LOG_ERR,
  LOG_WARN,
  LOG_INFO,
  LOG_DBG
};

const char *getLogString(LogLevel level);

#ifdef LOG_LEVEL
#define LOG(level, ...) do {                                           \
  if (level <= LOG_LEVEL ) {                                                   \
    fprintf(stderr,"[%s] [%s:%d]: ", getLogString(level), __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__);                                              \
    fprintf(stderr, "\n");                                                     \
    fflush(stderr);                                                            \
  }                                                                            \
  if (level == LOG_FATAL) {                                                    \
    exit(1);                                                                   \
  }                                                                            \
} while (0)
#else
#define LOG(level, ...)  do { } while(0)
#endif

#endif
