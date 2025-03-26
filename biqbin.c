#include "biqbin.h"
#include <string.h>
#include <sys/time.h>


// For external Init()
extern GlobalVariables *globals;
#define HEAP_SIZE 1000000
extern Heap *heap;
extern BabSolution *BabSol;
extern BiqBinParameters params;

extern int BabPbSize;

// Set all the necessary globals and variables for Evaluate to run
void InitSolverWrapped(double *L, int number_of_vertices, BiqBinParameters biqbin_parameters)
{
    globals = calloc(1, sizeof(GlobalVariables));
    // allocate memory for original problem SP and subproblem PP
    alloc(globals->SP, Problem);
    alloc(globals->PP, Problem);

    globals->SP->n = number_of_vertices;
    globals->PP->n = number_of_vertices;
    // allocate memory for objective matrices for SP and PP
    // globals->SP->L = L;
    alloc_matrix(globals->SP->L, globals->SP->n, double);
    memcpy(globals->SP->L, L, number_of_vertices * number_of_vertices * sizeof(double));
    BabPbSize = number_of_vertices - 1;

    alloc_matrix(globals->PP->L, globals->SP->n, double);
    // Parallel specific
    int N2 = globals->SP->n * globals->SP->n;
    int incx = 1;
    int incy = 1;
    dcopy_(&N2, globals->SP->L, &incx, globals->PP->L, &incy);

    srand(2020);
    setParams(biqbin_parameters);
    
    // Provide B&B with an initial solution
    initializeBabSolution();
    // Allocate the memory
    allocMemory(globals);
    globals->TIME = MPI_Wtime();
}
/* timer */
double time_wall_clock(void)  {
    struct timeval timecheck;
    gettimeofday(&timecheck, NULL);
    return timecheck.tv_sec + timecheck.tv_usec * 1e-6;

}

void EvaluateWrapped(BabNode *node, int rank) {
    Bab_incEvalNodes();
    double upper = Evaluate(node, globals, rank);
    node->upper_bound = upper;
}