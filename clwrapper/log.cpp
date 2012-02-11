#include "log.h"

const char *getLogString(LogLevel level) {
  switch (level) {
    case LOG_FATAL: return "FATAL";
    case LOG_ERR:   return "ERR  ";
    case LOG_WARN:  return "WARN ";
    case LOG_INFO:  return "INFO ";
    case LOG_DBG:   return "DEBUG";
    default:        return "-----";
  }
}
