#include "biqbin.h"
#include <string.h>

#define HEAP_SIZE 10000000

extern Heap *heap;
extern GlobalVariables *globals;

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
    globals->SP->L = L;
    BabPbSize = number_of_vertices - 1;

    alloc_matrix(globals->PP->L, globals->SP->n, double);
    // Parallel specific
    int N2 = globals->SP->n * globals->SP->n;
    int incx = 1;
    int incy = 1;
    dcopy_(&N2, globals->SP->L, &incx, globals->PP->L, &incy);

    srand(2024);
    setParams(biqbin_parameters);
    
    // Provide B&B with an initial solution
    initializeBabSolution();
    // Allocate the memory
    allocMemory();
    globals->TIME = MPI_Wtime();
}

void EvaluateWrapped(BabNode *node, int rank) {
    Bab_incEvalNodes();
    double upper = Evaluate(node, globals->SP, globals->PP, rank);
    node->upper_bound = upper;
}