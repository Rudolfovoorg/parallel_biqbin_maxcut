#ifndef WRAPPER_H
#define WRAPPER_H

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif
EXTERN_C int wrapped_read_data(void);
EXTERN_C void clean_python_references(void);
EXTERN_C void copy_solution(void);
EXTERN_C void copy_root_solution(void);
EXTERN_C void record_time(double time_taken);
EXTERN_C int get_time_limit(void);
#endif
