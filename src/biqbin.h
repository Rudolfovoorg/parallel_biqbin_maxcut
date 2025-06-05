#ifndef BIQBIN_H
#define BIQBIN_H

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "biqbin_cpp_api.h"
#include "blas_laplack.h"
/************************************************************************************************************/

// MESSAGES for MPI
typedef enum Message
{
    SEND_FREEWORKERS, // send ranks of free workers
    IDLE,             // worker is free, his local queue of subproblems is empty
    NEW_VALUE         // better lower bound found
} Message;

// TAGS in MPI messages
typedef enum Tags
{
    OVER,       // info to finish
    MESSAGE,    // type of message
    FREEWORKER, // when receiving/sending rank of free worker
    NUM_FREE_WORKERS,
    PROBLEM,
    LOWER_BOUND, // new lower bound
    SOLUTION     // solution vector
} Tags;

/************************************************************************************************************/

#define BIG_NUMBER 1e+9

/* Maximum number of cutting planes (triangle, pentagonal and heptagonal inequalities) allowed to add */
#define MaxTriIneqAdded 50000
#define MaxPentIneqAdded 50000
#define MaxHeptaIneqAdded 50000

/* Maximum size of bundle */
#define MaxBundle 400

/* Branching strategies */
#define LEAST_FRACTIONAL 0
#define MOST_FRACTIONAL 1

// BiqBin parameters and default values
#ifndef PARAM_FIELDS
#define PARAM_FIELDS                         \
    P(int, init_bundle_iter, "%d", 3)        \
    P(int, max_bundle_iter, "%d", 15)        \
    P(int, triag_iter, "%d", 5)              \
    P(int, pent_iter, "%d", 5)               \
    P(int, hept_iter, "%d", 5)               \
    P(int, max_outer_iter, "%d", 20)         \
    P(int, extra_iter, "%d", 10)             \
    P(double, violated_TriIneq, "%lf", 1e-3) \
    P(int, TriIneq, "%d", 5000)              \
    P(int, adjust_TriIneq, "%d", 1)          \
    P(int, PentIneq, "%d", 5000)             \
    P(int, HeptaIneq, "%d", 5000)            \
    P(int, Pent_Trials, "%d", 60)            \
    P(int, Hepta_Trials, "%d", 50)           \
    P(int, include_Pent, "%d", 1)            \
    P(int, include_Hepta, "%d", 1)           \
    P(int, root, "%d", 0)                    \
    P(int, use_diff, "%d", 1)                \
    P(int, time_limit, "%d", 0)              \
    P(int, branchingStrategy, "%d", MOST_FRACTIONAL)
#endif

typedef struct BiqBinParameters
{
#define P(type, name, format, def_value) type name;
    PARAM_FIELDS
#undef P
} BiqBinParameters;

/* Structure for storing triangle inequalities */
typedef struct Triangle_Inequality
{
    int i;
    int j;
    int k;
    int type;     // type: 1-4
    double value; // cut violation
    double y;     // corresponding dual multiplier
} Triangle_Inequality;

/* Structure for storing pentagonal inequalities */
typedef struct Pentagonal_Inequality
{
    int type; // type: 1-3 (based on H1 = ee^T, ...)
    int permutation[5];
    double value; // cut violation
    double y;     // corresponding dual multiplier
} Pentagonal_Inequality;

/* Structure for storing heptagonal inequalities */
typedef struct Heptagonal_Inequality
{
    int type; // type: 1-4 (based on H1 = ee^T, ...)
    int permutation[7];
    double value; // cut violation
    double y;     // corresponding dual multiplier
} Heptagonal_Inequality;

/* heap (data structure) declaration */
typedef struct Heap
{
    int size;       /* maximum number of elements in heap */
    int used;       /* current number of elements in heap */
    BabNode **data; /* array of BabNodes                  */
} Heap;

/**** Declarations of functions per file ****/

/* allocate_free.c */
void allocMemory(void);
void freeMemory(void);

/* bab_functions.c */
void initializeBabSolution(void);
int Init_PQ(void);
int Bab_Init(int argc, char **argv, int rank);
int updateSolution(int *x);
void master_Bab_Main(Message message, int source, int *busyWorkers, int numbWorkers, int *numbFreeWorkers, MPI_Datatype BabSolutiontype);
void worker_Bab_Main(MPI_Datatype BabSolutiontype, MPI_Datatype BabNodetype, int rank);
void printSolution(FILE *file);
void printFinalOutput(FILE *file, int num_nodes);
void Bab_End(void);
int getBranchingVariable(BabNode *node);
int countFixedVariables(BabNode *node);

/* bounding.c */
double SDPbound(BabNode *node, Problem *SP, Problem *PP, int rank);

/* bundle.c */
double fct_eval(const Problem *PP, double *dual_gamma, double *X, double *g);
void solve_lambda(int k, double *Q, double *c, double *lambda);
void lambda_eta(const Problem *PP, double *zeta, double *G, double *dual_gamma, double *dgamma, double *lambda, double *eta, double *t);
void bundle_method(Problem *PP, double *t, int bdl_iter);

/* cutting_planec.c */
double evaluateTriangleInequality(double *XX, int N, int type, int ii, int jj, int kk);
double getViolated_TriangleInequalities(double *X, int N, Triangle_Inequality *List, int *ListSize);
double updateTriangleInequalities(Problem *PP, double *y, int *NumAdded, int *NumSubtracted);
double getViolated_PentagonalInequalities(double *X, int N, Pentagonal_Inequality *Pent_List, int *ListSize);
double updatePentagonalInequalities(Problem *PP, double *y, int *NumAdded, int *NumSubtracted, int triag);
double getViolated_HeptagonalInequalities(double *X, int N, Heptagonal_Inequality *Hepta_List, int *ListSize);
double updateHeptagonalInequalities(Problem *PP, double *y, int *NumAdded, int *NumSubtracted, int hept_index);

/* evaluate.c */
double Evaluate(BabNode *node, Problem *SP, Problem *PP, int rank);
void createSubproblem(BabNode *node, Problem *SP, Problem *PP);
double getFixedValue(BabNode *node, Problem *SP);

/* heap.c */
int Bab_numEvalNodes(void);                          // returns number of evaluated nodes
void Bab_incEvalNodes(void);                         // increment the number of evaluated nodes
int isPQEmpty(void);                                 // checks if queue is empty
int Bab_LBUpd(double new_lb, BabSolution *bs);       // checks and updates lower bound if better found, returns 1 if success
BabNode *newNode(BabNode *parentNode);               // create child node from parent
BabNode *Bab_PQPop(void);                            // take and remove the node with the highest priority
void Bab_PQInsert(BabNode *node);                    // insert node into priority queue based on intbound and level
void Bab_LBInit(double lowerBound, BabSolution *bs); // initialize global lower bound and solution vector
Heap *Init_Heap(int size);                           // allocates space for heap (array of BabNode*)

/* heuristic.c */
double runHeuristic(Problem *P0, Problem *P, BabNode *node, int *x);
double GW_heuristic(double *P0_L, int P0_N , double *P_L, int P_N, int *node_xfixed, int *node_sol_X, int *x, int num); // RK
// RK double mc_1opt(int *x, Problem *P0);
double mc_1opt(int *x, double *P_L, int P_N);

/* ipm_mc_pk.c */
void ipm_mc_pk(double *L, int n, double *X, double *phi, int print);

/* operators.c */
void diag(const double *X, double *y, int n);
void Diag(double *X, const double *y, int n);
void op_B(const Problem *P, double *y, const double *X);
void op_Bt(const Problem *P, double *X, const double *tt);

/* process_input.c */
void print_symmetric_matrix(double *Mat, int N);
int processCommandLineArguments(int argc, char **argv, int rank);
int readParameters(const char *path, int rank);


/* qap_simuted_annealing.c */
double qap_simulated_annealing(int *H, int k, double *X, int n, int *pent);

/* main.c */

#endif /*BIQBIN_H */
