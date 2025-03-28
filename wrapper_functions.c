#include <stdio.h>
#include <mpi.h>
#include <stddef.h>
#include <string.h>

#include "biqbin.h"

// heap.c globals
#define HEAP_SIZE 1000000
extern Heap *heap;
extern int BabPbSize;
extern BabSolution *BabSol;

// other external globals
extern BiqBinParameters params;
extern FILE *output;
extern GlobalVariables *globals;

// local variables needed as globals
int over = 0;
extern int num_workers_used; // number of worker processes used by the solver
int numbWorkers;             // MPI comm size
int numbFreeWorkers;
int *busyWorkers;
/***** user defined MPI struct: for sending and receiving *****/
MPI_Datatype BabSolutiontype;
MPI_Datatype BabNodetype;

// Same for all processes, initialize MPI and return rank of process
int initMPI(int argc, char **argv)
{
    // MPI Start: start parallel environment
    MPI_Init(&argc, &argv);

    // get number of proccesses and corresponding ranks
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numbWorkers);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        printf("Number of cores: %d\n", numbWorkers);

    /***** user defined MPI struct: for sending and receiving *****/
    // (1) for BabSolution
    MPI_Datatype type1[1] = {MPI_INT};
    int blocklen1[1] = {NMAX};
    MPI_Aint disp1[1];
    disp1[0] = offsetof(BabSolution, X);
    MPI_Type_create_struct(1, blocklen1, disp1, type1, &BabSolutiontype);
    MPI_Type_commit(&BabSolutiontype);

    // (2) for BabNode
    MPI_Datatype type2[5] = {MPI_INT, BabSolutiontype, MPI_DOUBLE, MPI_INT, MPI_INT};
    int blocklen2[5] = {NMAX, 1, NMAX, 1, 1};
    MPI_Aint disp2[5];
    disp2[0] = offsetof(BabNode, xfixed);
    disp2[1] = offsetof(BabNode, sol);
    disp2[2] = offsetof(BabNode, fracsol);
    disp2[3] = offsetof(BabNode, level);
    disp2[4] = offsetof(BabNode, upper_bound);
    MPI_Type_create_struct(5, blocklen2, disp2, type2, &BabNodetype);
    MPI_Type_commit(&BabNodetype);
    return rank;
}

void finalizeMPI()
{
    MPI_Finalize();
}
// set globals, heap, print initial output, communicate main problem to worker processes, evaluate root node...
int master_init(char *filename, double *L, int num_vertices, int num_edges, BiqBinParameters params_in)
{
    globals = calloc(1, sizeof(GlobalVariables));
    // Start the timer here or in compute?
    globals->TIME = MPI_Wtime();
    /* each process allocates its local priority queue */
    heap = Init_Heap(HEAP_SIZE);

    // Bab_Init(argc, argv, rank) start
    openOutputFile(filename);
    // Write input data to output
    printf("Input file: %s\n", filename);
    fprintf(output, "Input file: %s\n", filename);
    // OUTPUT information on instance
    fprintf(stdout, "\nGraph has %d vertices and %d edges.\n", num_vertices, num_edges);
    fprintf(output, "\nGraph has %d vertices and %d edges.\n", num_vertices, num_edges);
    // READING INSTANCE FILE
    // allocate memory for original problem SP and subproblem PP
    alloc(globals->SP, Problem);
    alloc(globals->PP, Problem);

    globals->SP->n = num_vertices;
    globals->PP->n = num_vertices;
    // allocate memory for objective matrices for SP and PP
    // globals->SP->L = L;
    alloc_matrix(globals->SP->L, globals->SP->n, double);
    memcpy(globals->SP->L, L, num_vertices * num_vertices * sizeof(double));
    BabPbSize = num_vertices - 1;

    alloc_matrix(globals->PP->L, globals->SP->n, double);
    // Parallel specific
    int N2 = globals->SP->n * globals->SP->n;
    int incx = 1;
    int incy = 1;
    dcopy_(&N2, globals->SP->L, &incx, globals->PP->L, &incy);
    // END reading params
    MPI_Bcast(&(globals->SP->n), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(globals->SP->L, globals->SP->n * globals->SP->n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // set global parameters
    setParams(params_in);

    // Seed the random number generator
    srand(2020);

    // Provide B&B with an initial solution
    initializeBabSolution();

    // Allocate the memory
    allocMemory(globals);
    // End of Bab_Init(argc, argv, rank)
    // AFTER INPUT DATA HAS BEEN PROCESSED
    // helper variables
    BabNode *node;
    double g_lowerBound;

    /******************** MASTER PROCESS ********************/
    // only master evaluates the root node
    // and places it in priority queue if not able to prune
    over = Init_PQ();

    printf("Initial lower bound: %.0lf\n", Bab_LBGet());

    // broadcast diff
    printf("diff = %f", globals->diff);
    if (params.use_diff)
        MPI_Bcast(&globals->diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // broadcast lower bound to others or -1 to exit
    MPI_Bcast(&over, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if ((over == -1) || params.root)
    {
        return over;
    }
    else
    {
        g_lowerBound = Bab_LBGet();
        MPI_Bcast(&g_lowerBound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // array of busy workers: 0 = free, 1 = busy
    // only master is busy
    busyWorkers = (int *)malloc(numbWorkers * sizeof(int));
    busyWorkers[0] = 1;
    for (int i = 1; i < numbWorkers; ++i)
        busyWorkers[i] = 0;

    numbFreeWorkers = numbWorkers - 1;
    /***** branch root node and send to workers *****/
    node = Bab_PQPop();

    // Determine the variable x[ic] to branch on
    int ic = getBranchingVariable(node);

    // send two nodes to workers 1 and 2
    BabNode *child_node;
    int worker;

    for (int xic = 0; xic <= 1; ++xic)
    {

        // Create a new child node from the parent node
        child_node = newNode(node);

        // split on node ic
        child_node->xfixed[ic] = 1;
        child_node->sol.X[ic] = xic;

        // increment the number of explored nodes
        Bab_incEvalNodes();

        worker = xic + 1;
        busyWorkers[worker] = 1;
        --numbFreeWorkers;

        MPI_Send(&over, 1, MPI_INT, worker, OVER, MPI_COMM_WORLD);
        MPI_Send(&g_lowerBound, 1, MPI_DOUBLE, worker, LOWER_BOUND, MPI_COMM_WORLD);
        MPI_Send(child_node, 1, BabNodetype, worker, PROBLEM, MPI_COMM_WORLD);

        free(child_node);
    }

    // free parent nodes
    free(node);
    num_workers_used = 2;
    return over;
}
// master_Bab_Main remains unchanged, coordinates communication with workers main loop
int master_main_loop()
{
    MPI_Status status;
    Message message;
    /*** wait for messages: extract source from status ***/
    MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, MESSAGE, MPI_COMM_WORLD, &status);
    int source = status.MPI_SOURCE;

    master_Bab_Main(message, source, busyWorkers, numbWorkers, &numbFreeWorkers, BabSolutiontype);
    over = (numbFreeWorkers == numbWorkers - 1) ? 1 : 0;

    return over; // If it returns 0 end it
}
// Send over signal to all worker processes then print end output and free memory
void master_end()
{
    // send over messages to the workers
    for (int i = 1; i < numbWorkers; ++i)
    {
        MPI_Send(&over, 1, MPI_INT, i, OVER, MPI_COMM_WORLD);
    }

    /* Print results to the standard output and to the output file */
    printFinalOutput(stdout, Bab_numEvalNodes());
    printFinalOutput(output, Bab_numEvalNodes());
    fprintf(output, "Number of cores: %d\n", numbWorkers);
    fprintf(output, "Maximum number of workers used: %d\n", num_workers_used);
    printf("Maximum number of workers used: %d\n", num_workers_used);
    fclose(output);

    /* free memory */
    Bab_End();

    free(busyWorkers);
    free(heap->data);
    free(heap);
}

// Worker receives the SP->L matrix and number of vertices from master process, needs params
int worker_init(BiqBinParameters params_in)
{
    globals = calloc(1, sizeof(GlobalVariables));
    // Start the timer here or in compute?
    globals->TIME = MPI_Wtime();
    /* each process allocates its local priority queue */
    heap = Init_Heap(HEAP_SIZE);

    // Bab_Init - read input file
    // allocate memory for original problem SP and subproblem PP
    alloc(globals->SP, Problem);
    alloc(globals->PP, Problem);

    MPI_Bcast(&(globals->SP->n), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // allocate memory for objective matrices for SP and PP
    alloc_matrix(globals->SP->L, globals->SP->n, double);
    alloc_matrix(globals->PP->L, globals->SP->n, double);

    MPI_Bcast(globals->SP->L, globals->SP->n * globals->SP->n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // IMPORTANT: last node is fixed to 0
    // --> BabPbSize is one less than the size of problem SP
    BabPbSize = globals->SP->n - 1; // num_vertices - 1;
    globals->PP->n = globals->SP->n;

    int N2 = globals->SP->n * globals->SP->n;
    int incx = 1;
    int incy = 1;
    dcopy_(&N2, globals->SP->L, &incx, globals->PP->L, &incy);

    // set global parameters
    setParams(params_in);
    // Seed the random number generator
    srand(2020);

    // Provide B&B with an initial solution
    initializeBabSolution();

    // Allocate the memory
    allocMemory(globals);
    // End Bab_Init

    // helper variables
    double g_lowerBound;
    /******************** WORKER PROCESS ********************/
    // receive diff
    if (params.use_diff)
        MPI_Bcast(&globals->diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // receive over (stop or continue)
    MPI_Bcast(&over, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // receive lower bound
    if (over == -1 || params.root) // root node is pruned
        return over;
    else
        MPI_Bcast(&g_lowerBound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // update lower bound
    BabSolution solx;
    Bab_LBUpd(g_lowerBound, &solx);
    return over;
}

void worker_end()
{
    Bab_End();
    free(heap->data);
    free(heap);
}

/*************************************************************************/
/********************       WORKER MAIN LOOP          ********************/
/*************************************************************************/
// First communication check in worker loop
int worker_check_over()
{
    MPI_Status status;
    MPI_Recv(&over, 1, MPI_INT, MPI_ANY_SOURCE, OVER, MPI_COMM_WORLD, &status);
    return over;
}
// If not over receive subproblem
void worker_receive_problem()
{
    // helper variables
    MPI_Status status;
    double g_lowerBound;
    BabNode *node;
    BabSolution solx;
    alloc(node, BabNode); // get's freed inside worker_Bab_Main

    // receive subproblem from master or other worker
    MPI_Recv(&g_lowerBound, 1, MPI_DOUBLE, MPI_ANY_SOURCE, LOWER_BOUND, MPI_COMM_WORLD, &status);
    MPI_Recv(node, 1, BabNodetype, MPI_ANY_SOURCE, PROBLEM, MPI_COMM_WORLD, &status);

    // update
    Bab_LBUpd(g_lowerBound, &solx);

    // printf("%d - node received: %p\n", rank, (void*) node);
    // start local queue
    Bab_PQInsert(node);
}

int time_limit_reached()
{
    return (params.time_limit > 0 && (MPI_Wtime() - globals->TIME) > params.time_limit) ? 1 : 0;
}

// evaluate with global GlobalVariables struct already set
void evaluate_node_wrapped(BabNode *node, int rank)
{
    /* compute upper bound (SDP bound) and lower bound (via heuristic) for this node */
    node->upper_bound = Evaluate(node, globals, rank);
}

// Check if solution is better, update, communicate with master, send problems to workers etc..
void after_evaluation(BabNode *node, double old_lowerbound)
{
    Message message;
    MPI_Status status;

    // check if better lower bound found --> update info with master
    if (Bab_LBGet() > old_lowerbound)
    {

        message = NEW_VALUE;
        old_lowerbound = Bab_LBGet();

        MPI_Send(&message, 1, MPI_INT, 0, MESSAGE, MPI_COMM_WORLD);
        MPI_Send(&old_lowerbound, 1, MPI_DOUBLE, 0, LOWER_BOUND, MPI_COMM_WORLD);
        MPI_Send(BabSol, 1, BabSolutiontype, 0, SOLUTION, MPI_COMM_WORLD);

        MPI_Recv(&old_lowerbound, 1, MPI_DOUBLE, 0, LOWER_BOUND, MPI_COMM_WORLD, &status);

        // update
        BabSolution solx;
        Bab_LBUpd(old_lowerbound, &solx);
    }
    /* if BabLB + 1.0 < child_node->upper_bound,
     * then we must branch since there could be a better feasible
     * solution in this subproblem
     */
    if (Bab_LBGet() + 1.0 < node->upper_bound)
    {

        /***** branch *****/

        // Determine the variable x[ic] to branch on
        int ic = getBranchingVariable(node);

        BabNode *child_node;

        for (int xic = 0; xic <= 1; ++xic)
        {

            // Create a new child node from the parent node
            child_node = newNode(node);

            // split on node ic
            child_node->xfixed[ic] = 1;
            child_node->sol.X[ic] = xic;

            /* insert node into the priority queue */
            Bab_PQInsert(child_node);
        }

        // free parent node
        free(node);

        /************ distribute subproblems ************/

        // leave 1 problem for this worker and the rest is distributed
        int workers_request = heap->used - 1;
        int num_free_workers;
        double g_lowerBound;
        BabSolution solx;

        // check if other subproblems can be send to free workers --> ask master
        message = SEND_FREEWORKERS;

        MPI_Send(&message, 1, MPI_INT, 0, MESSAGE, MPI_COMM_WORLD);
        MPI_Send(&workers_request, 1, MPI_INT, 0, FREEWORKER, MPI_COMM_WORLD);

        MPI_Recv(&num_free_workers, 1, MPI_INT, 0, NUM_FREE_WORKERS, MPI_COMM_WORLD, &status);

        int free_workers[num_free_workers];

        MPI_Recv(free_workers, num_free_workers, MPI_INT, 0, FREEWORKER, MPI_COMM_WORLD, &status);
        MPI_Recv(&g_lowerBound, 1, MPI_DOUBLE, 0, LOWER_BOUND, MPI_COMM_WORLD, &status);

        Bab_LBUpd(g_lowerBound, &solx);

        // send subproblems to free workers
        if (num_free_workers != 0)
        { // free workers found

            for (int i = 0; i < num_free_workers; ++i)
            {

                // get next subproblem from queue and send it
                node = Bab_PQPop();

                // send subproblem to free worker
                MPI_Send(&over, 1, MPI_INT, free_workers[i], OVER, MPI_COMM_WORLD);
                MPI_Send(&g_lowerBound, 1, MPI_DOUBLE, free_workers[i], LOWER_BOUND, MPI_COMM_WORLD);
                MPI_Send(node, 1, BabNodetype, free_workers[i], PROBLEM, MPI_COMM_WORLD);

                free(node);
            }
        }
    }
    else
    {
        // otherwise, intbound <= BabLB, so we can prune
        free(node);
    }
}

void worker_send_idle()
{
    Message message = IDLE;
    MPI_Send(&message, 1, MPI_INT, 0, MESSAGE, MPI_COMM_WORLD);
}

/******************   UNIT TEST INIT FUNCTIONS   ********************/
// Get the SP main problem pointer
Problem *get_SP(double *L, int num_vertices)
{
    Problem *SP;
    alloc(SP, Problem);
    SP->n = num_vertices;
    // SP->L = L;
    alloc_matrix(SP->L, num_vertices, double);
    memcpy(SP->L, L, num_vertices * num_vertices * sizeof(double));
    return SP;
}

// Get initial PP pointer
Problem *get_PP(Problem *SP)
{
    Problem *PP;
    alloc(PP, Problem);
    PP->n = SP->n;
    alloc_matrix(PP->L, PP->n, double);
    int N2 = SP->n * SP->n;
    int incx = 1;
    int incy = 1;
    dcopy_(&N2, SP->L, &incx, PP->L, &incy);

    return PP;
}
// Get global variables struct pointer, needs global *params to be set
GlobalVariables *get_globals(double *L, int num_vertices)
{
    GlobalVariables *globe = calloc(1, sizeof(GlobalVariables));
    // allocate memory for original problem SP and subproblem PP
    alloc(globe->SP, Problem);
    alloc(globe->PP, Problem);

    globe->SP->n = num_vertices;
    globe->PP->n = num_vertices;
    // allocate memory for objective matrices for SP and PP
    // globals->SP->L = L;
    alloc_matrix(globe->SP->L, globe->SP->n, double);
    memcpy(globe->SP->L, L, num_vertices * num_vertices * sizeof(double));

    alloc_matrix(globe->PP->L, globe->SP->n, double);
    // Parallel specific
    int N2 = globe->SP->n * globe->SP->n;
    int incx = 1;
    int incy = 1;
    dcopy_(&N2, globe->SP->L, &incx, globe->PP->L, &incy);
    // Provide B&B with an initial solution
    initializeBabSolution();
    // Allocate the memory
    allocMemory(globe);
    return globe;
}