#ifndef BIQBIN_CPP_API_H
#define BIQBIN_CPP_API_H


#ifdef __cplusplus
# define EXTERN_C extern "C"
#else
# define EXTERN_C
#endif

/* macros for allocating vectors and matrices */
#define alloc_vector(var, size, type)                                                       \
    var = (type *)calloc((size), sizeof(type));                                             \
    if (var == NULL)                                                                        \
    {                                                                                       \
        fprintf(stderr,                                                                     \
                "\nError: Memory allocation problem for variable " #var " in %s line %d\n", \
                __FILE__, __LINE__);                                                        \
        abort_alloc_fail(10);                                                          \
    }

#define alloc(var, type) alloc_vector(var, 1, type)
#define alloc_matrix(var, size, type) alloc_vector(var, (size) * (size), type)

/************************************************************************************************************/
/* The main problem and any subproblems are stored using the following structure. */
typedef struct Problem
{
    double *L;      // Objective matrix
    int n;          // size of L
    int NIneq;      // number of triangle inequalities
    int NPentIneq;  // number of pentagonal inequalities
    int NHeptaIneq; // number of heptagonal inequalities
    int bundle;     // size of bundle
} Problem;

/* Maximum number of variables */
#define NMAX 1024

/* Solution of the problem */
typedef struct BabSolution
{
    /*
     * Vector X: Binary vector that stores the solution of the branch-and-bound algorithm
     */
    int X[NMAX];
} BabSolution;

/*
 * Node of the branch-and-bound tree.
 * Structure that represent a node of the branch-and-bound tree and stores all the
 * useful information.
 */
typedef struct BabNode
{
    int xfixed[NMAX];     // 0-1 vector specifying which nodes are fixed
    BabSolution sol;      // 0-1 solution vector
    double fracsol[NMAX]; // fractional vector obtained from primal matrix X (last column except last element)
                          // from bounding routine. Used for determining the next branching variable.
    int level;            // level (depth) of the node in B&B tree
    double upper_bound;   // upper bound on solution value of max-cut, i.e. MC <= upper_bound.
                          // Used for determining the next node in priority queue.
} BabNode;

EXTERN_C double runHeuristic_unpacked(double *P0_L, int P0_N , double *P_L, int P_N, int *node_xfixed, int *node_sol_X, int *x);
EXTERN_C int wrapped_main(int argc, char **argv);
EXTERN_C double* readData(const char *instance, int *adj_N);
EXTERN_C int read_data_BQP(const char *instance);
EXTERN_C double Bab_LBGet(void);                              // returns global lower bound
EXTERN_C int update_best(int *xbest, int *xnew, double *best, int P0_N);
EXTERN_C double evaluateSolution(int *sol);
EXTERN_C void abort_alloc_fail(int abort_code);
EXTERN_C int process_adj_matrix(double* Adj, int Adj_N);
#endif
