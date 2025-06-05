#ifndef WRAPPER_H
#define WRAPPER_H

#ifdef __cplusplus
# define EXTERN_C extern "C"
#else
# define EXTERN_C
#endif

EXTERN_C double wrapped_heuristic(Problem *P0, Problem *P, BabNode *node, int *x);
EXTERN_C void wrapped_read_data();
EXTERN_C int process_adj_matrix(double* Adj, int Adj_N);
EXTERN_C void clean_python_references(void);
EXTERN_C void copy_solution(void);
EXTERN_C void record_time(double time_taken);
EXTERN_C void set_rank(int r);
#endif
