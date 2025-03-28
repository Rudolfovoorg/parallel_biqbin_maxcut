#ifndef BIQBIN_H
#define BIQBIN_H

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/************************************************************************************************************/

// MESSAGES for MPI
typedef enum Message{              
    SEND_FREEWORKERS,           // send ranks of free workers
    IDLE,                       // worker is free, his local queue of subproblems is empty
    NEW_VALUE                   // better lower bound found
} Message;

// TAGS in MPI messages
typedef enum Tags{
    OVER,                       // info to finish
    MESSAGE,                    // type of message
    FREEWORKER,                 // when receiving/sending rank of free worker
    NUM_FREE_WORKERS,
    PROBLEM,
    LOWER_BOUND,                // new lower bound
    SOLUTION                    // solution vector
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
#define LEAST_FRACTIONAL  0
#define MOST_FRACTIONAL   1

/* macros for allocating vectors and matrices */
#define alloc_vector(var, size, type)\
    var = (type *) calloc((size) , sizeof(type));\
    if(var == NULL){\
        fprintf(stderr,\
                "\nError: Memory allocation problem for variable "#var" in %s line %d\n",\
                __FILE__,__LINE__);\
        MPI_Abort(MPI_COMM_WORLD,10);\
    }

#define alloc(var, type) alloc_vector(var, 1, type)
#define alloc_matrix(var, size, type) alloc_vector(var, (size)*(size), type)


// BiqBin parameters and default values
#ifndef PARAM_FIELDS
#define PARAM_FIELDS \
    P(int,      init_bundle_iter,    "%d",                 3) \
    P(int,      max_bundle_iter,     "%d",                15) \
    P(int,      triag_iter,          "%d",                 5) \
    P(int,      pent_iter,           "%d",                 5) \
    P(int,      hept_iter,           "%d",                 5) \
    P(int,      max_outer_iter,      "%d",                20) \
    P(int,      extra_iter,          "%d",                10) \
    P(double,   violated_TriIneq,    "%lf",             1e-3) \
    P(int,      TriIneq,             "%d",              5000) \
    P(int,      adjust_TriIneq,      "%d",                 1) \
    P(int,      PentIneq,            "%d",              5000) \
    P(int,      HeptaIneq,           "%d",              5000) \
    P(int,      Pent_Trials,         "%d",                60) \
    P(int,      Hepta_Trials,        "%d",                50) \
    P(int,      include_Pent,        "%d",                 1) \
    P(int,      include_Hepta,       "%d",                 1) \
    P(int,      root,                "%d",                 0) \
    P(int,      use_diff,            "%d",                 1) \
    P(int,      time_limit,          "%d",                 0) \
    P(int,      branchingStrategy,   "%d",   MOST_FRACTIONAL) \
    P(int,      detailed_output,     "%d",                 0)
#endif


typedef struct BiqBinParameters { 
#define P(type, name, format, def_value) type name;
    PARAM_FIELDS
#undef P
} BiqBinParameters;

/* Structure for storing triangle inequalities */
typedef struct Triangle_Inequality {
    int i;
    int j;
    int k;
    int type;           // type: 1-4
    double value;       // cut violation 
    double y;           // corresponding dual multiplier
} Triangle_Inequality;


/* Structure for storing pentagonal inequalities */
typedef struct Pentagonal_Inequality {
    int type;           // type: 1-3 (based on H1 = ee^T, ...)
    int permutation[5];
    double value;       // cut violation 
    double y;           // corresponding dual multiplier
} Pentagonal_Inequality;


/* Structure for storing heptagonal inequalities */
typedef struct Heptagonal_Inequality {
    int type;           // type: 1-4 (based on H1 = ee^T, ...)
    int permutation[7];
    double value;       // cut violation 
    double y;           // corresponding dual multiplier
} Heptagonal_Inequality;


/* The main problem and any subproblems are stored using the following structure. */
typedef struct Problem {
    double *L;          // Objective matrix 
    int n;              // size of L
    int NIneq;          // number of triangle inequalities
    int NPentIneq;      // number of pentagonal inequalities
    int NHeptaIneq;     // number of heptagonal inequalities
    int bundle;         // size of bundle
} Problem;

typedef struct GlobalVariables {
    Problem *SP;                        // original problem instance
    Problem *PP;                        // subproblem instance
    int stopped;                    // true if the algorithm stopped at root node or after a time limit
    double root_bound;                  // SDP upper bound at root node
    double TIME;                        // CPU time
    double diff;			            // difference between basic SDP relaxation and bound with added cutting planes  	
    /********************************************************/
    
    
    /********************************************************/
    /*************** Specific to node ***********************/
    /********************************************************/
    /* PRIMAL variables */
    double *X;                          // Stores current (psd) X (primal solution). Violated inequalities are computed from X.
    double *Z;                          // Cholesky factorization: X = ZZ^T (used for heuristic)
    double *X_bundle;                   // containts bundle matrices as columns
    double *X_test;                     // matching pair X for gamma_test
    
    /* DUAL variables */
    double *dual_gamma;                      // (nonnegative) dual multiplier to cutting planes
    double *dgamma;                     // step direction vector
    double *gamma_test;
    double *lambda;                     // vector containing scalars of convex combinations of bundle matrices X_i
    double *eta;                        // dual multiplier to dual_gamma >= 0 constraint
    double *F;                          // vector of values <L,X_i>
    double *g;                          // subgradient
    double *G;                          // matrix of subgradients
    
    double f;                           // objective value of relaxation                      
    
    /* Triangle Inequalities variables */
    Triangle_Inequality *Cuts;          // vector (MaxTriIneqAdded) of current triangle inequality constraints
    Triangle_Inequality *List;          // vector (params.TriIneq) of new violated triangle inequalities
    
    /* Pentagonal Inequalities variables */
    Pentagonal_Inequality *Pent_Cuts;   // vector (MaxPentIneqAdded) of current pentagonal inequality constraints
    Pentagonal_Inequality *Pent_List;   // vector (params.PentIneq) of new violated pentagonal inequalities
    
    /* Heptagonal Inequalities variables */
    Heptagonal_Inequality *Hepta_Cuts;   // vector (MaxHeptaIneqAdded) of current heptagonal inequality constraints
    Heptagonal_Inequality *Hepta_List;   // vector (params.HeptaIneq) of new violated heptagonal inequalities    
} GlobalVariables;


/* Maximum number of variables */
#define NMAX 1024

/* Solution of the problem */
typedef struct BabSolution {
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
typedef struct BabNode {
    int xfixed[NMAX];       // 0-1 vector specifying which nodes are fixed
    BabSolution sol;        // 0-1 solution vector
    double fracsol[NMAX];   // fractional vector obtained from primal matrix X (last column except last element)
                            // from bounding routine. Used for determining the next branching variable.
    int level;              // level (depth) of the node in B&B tree     
    double upper_bound;     // upper bound on solution value of max-cut, i.e. MC <= upper_bound.
                            // Used for determining the next node in priority queue.  
} BabNode;


/* heap (data structure) declaration */
typedef struct Heap {   
    int size;               /* maximum number of elements in heap */
    int used;               /* current number of elements in heap */
    BabNode** data;         /* array of BabNodes                  */  
} Heap;



/****** BLAS  ******/

// level 1 blas 
extern void dscal_(int *n, double *alpha, double *X, int *inc);
extern void dcopy_(int *n, double *X, int *incx, double *Y, int *incy);
extern double dnrm2_(int *n, double *x, int *incx);
extern void daxpy_(int *n, double *alpha, double *X, int *incx, double *Y, int *incy);
extern double ddot_(int *n, double *X, int *incx, double *Y, int *incy);

// level 2 blas
extern void dsymv_(char *uplo, int *n, double *alpha, double *A, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
extern void dgemv_(char *uplo, int *m, int *n, double *alpha, double *A, int *lda, double *X, int *incx, double *beta, double *Y, int *incy);
extern void dsyr_(char *uplo, int *n, double *alpha, double *x, int *incx, double *A, int *lda);

// level 3 blas
extern void dsymm_(char *side, char *uplo, int *m, int *n, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
extern void dsyrk_(char *UPLO, char *TRANS, int *N, int *K, double *ALPHA, double *A, int *LDA, double *BETA, double *C, int *LDC);


/****** LAPACK  ******/

// computes Cholesky factorization of positive definite matrix
extern void dpotrf_(char *uplo, int *n, double *X, int *lda, int *info);

// computes the inverse of a real symmetric positive definite
// matrix  using the Cholesky factorization  
extern void dpotri_(char *uplo, int *n, double *X, int *lda, int *info);

// computes solution to a real system of linear equations with symmetrix matrix
extern void dsysv_(char *uplo, int *n, int *nrhs, double *A, int *lda, int *ipiv, double *B, int *ldb, double *work, int *lwork, int *info);  

// computes solution to a real system of linear equations with positive definite matrix  
extern void dposv_(char *uplo, int *n, int *nrhs, double *A, int *lda, double *B, int *ldb, int *info);


/**** Declarations of functions per file ****/

/* allocate_free.c */
void allocMemory(GlobalVariables *globals_in);
void freeMemory(GlobalVariables *globals_in);

/* bab_functions.c */
void initializeBabSolution(void);
int Init_PQ(void);
int Bab_Init(int argc, char **argv, int rank);
double evaluateSolution(int *sol, Problem *SP);
int updateSolution(int *x, Problem *SP);
void master_Bab_Main(Message message, int source, int *busyWorkers, int numbWorkers, int *numbFreeWorkers, MPI_Datatype BabSolutiontype);
void worker_Bab_Main(MPI_Datatype BabSolutiontype, MPI_Datatype BabNodetype, int rank);
void printSolution(FILE *file);
void printFinalOutput(FILE *file, int num_nodes);
void Bab_End(void);
int getBranchingVariable(BabNode *node);
int countFixedVariables(BabNode *node);

/* bounding.c */
double SDPbound(BabNode *node, Problem *PP, int rank, GlobalVariables *globals);

/* bundle.c */
double fct_eval(const Problem *PP, double *gamma, double *X, double *g, GlobalVariables *globals);
void solve_lambda(int k, double *Q, double *c, double *lambda);
void lambda_eta(const Problem *PP, double *zeta, double *G, double *gamma, double *dgamma, double *lambda, double *eta, double *t);
void bundle_method(Problem *PP, double *t, int bdl_iter, GlobalVariables *globals);

/* cutting_planec.c */
double evaluateTriangleInequality(double *XX, int N, int type, int ii, int jj, int kk);
double getViolated_TriangleInequalities(double *X, int N, Triangle_Inequality *List, int *ListSize);
double updateTriangleInequalities(Problem *PP, double *y, int *NumAdded, int *NumSubtracted, GlobalVariables *globals);
double getViolated_PentagonalInequalities(double *X, int N, Pentagonal_Inequality *Pent_List, int *ListSize);
double updatePentagonalInequalities(Problem *PP, double *y, int *NumAdded, int *NumSubtracted, int triag, GlobalVariables *globals);
double getViolated_HeptagonalInequalities(double *X, int N, Heptagonal_Inequality *Hepta_List, int *ListSize);
double updateHeptagonalInequalities(Problem *PP, double *y, int *NumAdded, int *NumSubtracted, int hept_index, GlobalVariables *globals);

/* evaluate.c */
double Evaluate(BabNode *node, GlobalVariables *globals, int rank);
void createSubproblem(BabNode *node, Problem *SP, Problem *PP);
double getFixedValue(BabNode *node, Problem *SP);

/* heap.c */
double Bab_LBGet(void);                                    // returns global lower bound
int Bab_numEvalNodes(void);                                // returns number of evaluated nodes
void Bab_incEvalNodes(void);                               // increment the number of evaluated nodes
int isPQEmpty(void);                                       // checks if queue is empty
int Bab_LBUpd(double new_lb, BabSolution *bs);             // checks and updates lower bound if better found, returns 1 if success
BabNode* newNode(BabNode *parentNode);                     // create child node from parent
BabNode* Bab_PQPop(void);                                  // take and remove the node with the highest priority
void Bab_PQInsert(BabNode *node);                          // insert node into priority queue based on intbound and level 
void Bab_LBInit(double lowerBound, BabSolution *bs);       // initialize global lower bound and solution vector
Heap* Init_Heap(int size);                                 // allocates space for heap (array of BabNode*)
void LBUpdate(double new_LB);

/* heuristic.c */
double runHeuristic(Problem *P0, Problem *P, BabNode *node, int *x, double *X, double *Z);
double GW_heuristic(Problem *P0, Problem *P, BabNode *node, int *x, int num, double *Z);
double mc_1opt(int *x, Problem *P0);
int update_best(int *xbest, int *xnew, double *best, Problem *P0);

/* ipm_mc_pk.c */
void ipm_mc_pk(double *L, int n, double *X, double *phi, int print);

/* operators.c */
void diag(const double *X, double *y, int n);
void Diag(double *X, const double *y, int n);
void op_B(const Problem *P, double *y, const double *X,  GlobalVariables *globals);
void op_Bt(const Problem *P, double *X, const double *tt,  GlobalVariables *globals);

/* process_input.c */
void print_symmetric_matrix(double *Mat, int N);
int openOutputFile(char* filename);
void closeOutputFile();
int processCommandLineArguments(int argc, char **argv, int rank);
int readData(const char *instance);
int readParameters(const char *path, int rank);
void setParams(BiqBinParameters params);
void setParams(BiqBinParameters params_in);
void printParameters(BiqBinParameters params_in);
double getDiff();
void setDiff(double diff);


/* qap_simuted_annealing.c */
double qap_simulated_annealing(int *H, int k, double *X, int n, int *pent);

/* biqbin.c */
void init_solver_wrapped(double *L, int number_of_vertices, BiqBinParameters biqbin_parameters);
void evaluate_wrapped(BabNode *node, int rank);
BabNode* init_return_root(double* L, int num_vertices, BiqBinParameters params_in);

#endif /*BIQBIN_H */
