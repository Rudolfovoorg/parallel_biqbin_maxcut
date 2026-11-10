#ifndef BIQBIN_LOGGING_H
#define BIQBIN_LOGGING_H

/*
 * C/C++ logging macros backed by the existing Python BiqBin logger.
 *
 * Include this header in files that use BIQBIN_LOG_* calls. The Python bridge
 * is implemented in wrapper.cpp.
 *
 * Native records use logging.getLogger("biqbin") directly.
 * Python configures levels, filters, handlers, formatting and log files.
 * No initialization or MPI calls are needed; no Python references are retained.
 * Use only while Python is running. PURE_C builds use printf instead and need
 * no Python or pybind11 headers. PURE_C appends a newline to each record;
 * Python handlers control record termination. Omit trailing message newlines.
 *
 * BIQBIN_LOG_MIN_LEVEL sets the minimum severity compiled in (default INFO).
 * Example: -DBIQBIN_LOG_MIN_LEVEL=10 keeps DEBUG and higher; Python still
 * filters these records at runtime. Disabled calls do not evaluate arguments.
 * TRACE is a custom Python logging level (5); DEBUG through CRITICAL use the
 * standard Python levels. FATAL is an alias for CRITICAL and does not abort.
 *
 * Example: BIQBIN_LOG_INFO("node %d: bound %.2f", node->level, bound);
 */

/* Macros allow the preprocessor to remove calls below the build threshold. */
#define BIQBIN_LOG_LEVEL_TRACE 5
#define BIQBIN_LOG_LEVEL_DEBUG 10
#define BIQBIN_LOG_LEVEL_INFO 20
#define BIQBIN_LOG_LEVEL_WARNING 30
#define BIQBIN_LOG_LEVEL_ERROR 40
#define BIQBIN_LOG_LEVEL_CRITICAL 50

#ifndef BIQBIN_LOG_MIN_LEVEL
#define BIQBIN_LOG_MIN_LEVEL BIQBIN_LOG_LEVEL_INFO
#endif

#ifdef PURE_C

#include <stdio.h>

#define BIQBIN_LOG_AT(level, ...) \
    do { \
        if ((level) >= BIQBIN_LOG_MIN_LEVEL) { \
            (void)printf(__VA_ARGS__); \
            (void)putchar('\n'); \
        } \
    } while (0)

#else /* Python extension */

#ifdef __cplusplus
extern "C" {
#endif

void biqbin_log_write(int level, const char *format, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

#ifdef __cplusplus
}
#endif

#define BIQBIN_LOG_AT(level, ...) \
    do { \
        const int biqbin_logging_macro_level_ = (level); \
        if (biqbin_logging_macro_level_ >= BIQBIN_LOG_MIN_LEVEL) \
            biqbin_log_write(biqbin_logging_macro_level_, __VA_ARGS__); \
    } while (0)

#endif /* PURE_C */

#if BIQBIN_LOG_MIN_LEVEL <= BIQBIN_LOG_LEVEL_TRACE
#define BIQBIN_LOG_TRACE(...) BIQBIN_LOG_AT(BIQBIN_LOG_LEVEL_TRACE, __VA_ARGS__)
#else
#define BIQBIN_LOG_TRACE(...) ((void)0)
#endif

#if BIQBIN_LOG_MIN_LEVEL <= BIQBIN_LOG_LEVEL_DEBUG
#define BIQBIN_LOG_DEBUG(...) BIQBIN_LOG_AT(BIQBIN_LOG_LEVEL_DEBUG, __VA_ARGS__)
#else
#define BIQBIN_LOG_DEBUG(...) ((void)0)
#endif

#if BIQBIN_LOG_MIN_LEVEL <= BIQBIN_LOG_LEVEL_INFO
#define BIQBIN_LOG_INFO(...) BIQBIN_LOG_AT(BIQBIN_LOG_LEVEL_INFO, __VA_ARGS__)
#else
#define BIQBIN_LOG_INFO(...) ((void)0)
#endif

#if BIQBIN_LOG_MIN_LEVEL <= BIQBIN_LOG_LEVEL_WARNING
#define BIQBIN_LOG_WARNING(...) BIQBIN_LOG_AT(BIQBIN_LOG_LEVEL_WARNING, __VA_ARGS__)
#else
#define BIQBIN_LOG_WARNING(...) ((void)0)
#endif

#if BIQBIN_LOG_MIN_LEVEL <= BIQBIN_LOG_LEVEL_ERROR
#define BIQBIN_LOG_ERROR(...) BIQBIN_LOG_AT(BIQBIN_LOG_LEVEL_ERROR, __VA_ARGS__)
#else
#define BIQBIN_LOG_ERROR(...) ((void)0)
#endif

#if BIQBIN_LOG_MIN_LEVEL <= BIQBIN_LOG_LEVEL_CRITICAL
#define BIQBIN_LOG_CRITICAL(...) BIQBIN_LOG_AT(BIQBIN_LOG_LEVEL_CRITICAL, __VA_ARGS__)
#else
#define BIQBIN_LOG_CRITICAL(...) ((void)0)
#endif
/* FATAL records a critical message; it does not abort the solver or MPI. */
#define BIQBIN_LOG_FATAL(...) BIQBIN_LOG_CRITICAL(__VA_ARGS__)

#endif /* BIQBIN_LOGGING_H */
