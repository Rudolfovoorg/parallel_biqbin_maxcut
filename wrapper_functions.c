#include <stdio.h>
#include <mpi.h>
#include <stddef.h>
#include <string.h>

#include "biqbin.h"
#include "global_var.h"

// heap.c globals
#define HEAP_SIZE 1000000
extern Heap *heap;
extern int main_problem_size;
extern BabSolution *solution;

// other external globals
extern BiqBinParameters params;
extern FILE *output;
extern GlobalVariables globals;

// local variables needed as globals
int num_workers_used; // number of worker processes used by the solver
int numbWorkers;      // MPI comm size
int numbFreeWorkers;
int *busyWorkers;
/***** user defined MPI struct: for sending and receiving *****/
MPI_Datatype BabSolutiontype;
MPI_Datatype BabNodetype;

/// @brief First funtion to be run from python, initializes MPI
/// @param argc number of CLI arguments passed
/// @param argv CLI arguments
/// @return rank of process, 0 if master
int init_mpi_wrapped(int argc, char **argv)
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

/// @brief Sets globals, heap, prints initial output, communicates main problem SP to worker processes
/// @param filename
/// @param L
/// @param num_vertices
/// @param num_edges
/// @param params_in
void master_init(char *filename, double *L, int num_vertices, int num_edges, BiqBinParameters params_in)
{
    // Start the timer here or in compute?
    globals.TIME = MPI_Wtime();
    /* each process allocates its local priority queue */
    heap = init_heap(HEAP_SIZE);

    // Bab_Init(argc, argv, rank) start
    open_output_file(filename);
    // Write input data to output
    printf("Input file: %s\n", filename);
    fprintf(output, "Input file: %s\n", filename);
    // OUTPUT information on instance
    fprintf(stdout, "\nGraph has %d vertices and %d edges.\n", num_vertices, num_edges);
    fprintf(output, "\nGraph has %d vertices and %d edges.\n", num_vertices, num_edges);
    // READING INSTANCE FILE
    // allocate memory for original problem SP and subproblem PP
    alloc(globals.SP, Problem);
    alloc(globals.PP, Problem);

    globals.SP->n = num_vertices;
    globals.PP->n = num_vertices;
    // allocate memory for objective matrices for SP and PP
    // globals.SP->L = L;
    alloc_matrix(globals.SP->L, globals.SP->n, double);
    memcpy(globals.SP->L, L, num_vertices * num_vertices * sizeof(double));
    main_problem_size = num_vertices - 1;

    alloc_matrix(globals.PP->L, globals.SP->n, double);
    // Parallel specific
    int N2 = globals.SP->n * globals.SP->n;
    int incx = 1;
    int incy = 1;
    dcopy_(&N2, globals.SP->L, &incx, globals.PP->L, &incy);
    // END reading params
    MPI_Bcast(&(globals.SP->n), 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(globals.SP->L, globals.SP->n * globals.SP->n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // set global parameters
    set_parameters(params_in);
    if (params.adjust_TriIneq)
    {
        params.TriIneq = (main_problem_size + 1) * 10;
    }

    // Seed the random number generator
    srand(2020);

    // Provide B&B with an initial solution
    initializeBabSolution();

    // Allocate the memory
    allocMemory(&globals);
    // End of Bab_Init(argc, argv, rank)
    // AFTER INPUT DATA HAS BEEN PROCESSED
}

/// @brief Branch or prune from evaluated root node, communicate with child processes
/// @param root_node already evaluated
/// @return 1 if over or 0 if continue evaluating
int master_init_end(BabNode *root_node)
{
    globals.root_bound = root_node->upper_bound;
    printf("Root node bound: %.2f\n", globals.root_bound);

    /* insert node into the priority queue or prune */
    // NOTE: optimal solution has INTEGER value, i.e. add +1 to lower bound
    int over = 0;
    if (get_lower_bound() + 1.0 < root_node->upper_bound)
    {
        pq_push(root_node);
    }
    else
    {
        // otherwise, intbound <= global_lower_bound, so we can prune
        over = -1;
        free(root_node);
    }
    printf("Initial lower bound: %.0lf\n", get_lower_bound());

    // broadcast diff
    printf("diff = %f", globals.diff);
    if (params.use_diff)
        MPI_Bcast(&globals.diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // broadcast lower bound to others or -1 to exit
    MPI_Bcast(&over, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double g_lowerBound;
    if ((over == -1) || params.root)
    {
        return over;
    }
    else
    {
        g_lowerBound = get_lower_bound();
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
    BabNode *node = pq_pop();

    // Determine the variable x[ic] to branch on
    int ic = get_branching_variable(node);

    // send two nodes to workers 1 and 2
    BabNode *child_node;
    int worker;

    for (int xic = 0; xic <= 1; ++xic)
    {

        // Create a new child node from the parent node
        child_node = new_node(node);

        // split on node ic
        child_node->xfixed[ic] = 1;
        child_node->sol.X[ic] = xic;

        // increment the number of explored nodes
        increase_num_eval_nodes();

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

/// @brief Runs until over != 0, master_bab_main remains unchanged, coordinates communication with workers main loop
/// @returns 1 if over else 0
int master_main_loop()
{
    MPI_Status status;
    Message message;
    /*** wait for messages: extract source from status ***/
    MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, MESSAGE, MPI_COMM_WORLD, &status);
    int source = status.MPI_SOURCE;

    master_bab_main(message, source, busyWorkers, numbWorkers, &numbFreeWorkers, BabSolutiontype);
    int over = (numbFreeWorkers == numbWorkers - 1) ? 1 : 0;

    return over; // If it returns 0 end it
}

/// @brief Send over signal to all worker processes then print end output, free memory
void master_end()
{
    // send over messages to the workers
    int over = 1;
    for (int i = 1; i < numbWorkers; ++i)
    {
        MPI_Send(&over, 1, MPI_INT, i, OVER, MPI_COMM_WORLD);
    }

    /* Print results to the standard output and to the output file */
    print_final_output(stdout, num_evaluated_nodes());
    print_final_output(output, num_evaluated_nodes());
    fprintf(output, "Number of cores: %d\n", numbWorkers);
    fprintf(output, "Maximum number of workers used: %d\n", num_workers_used);
    printf("Maximum number of workers used: %d\n", num_workers_used);
    fclose(output);

    /* free memory */
    freeMemory(&globals);   

    free(busyWorkers);
    free(heap->data);
    free(heap);
}

/// @brief Receives the SP->L matrix, number of vertices and lower bound from master process, sets them for this process
/// @param params_in sets parameters in this process
/// @return 0 if not over, non-0 if over
int worker_init(BiqBinParameters params_in)
{
    // Start the timer here or in compute?
    globals.TIME = MPI_Wtime();
    /* each process allocates its local priority queue */
    heap = init_heap(HEAP_SIZE);

    // Bab_Init - read input file
    // allocate memory for original problem SP and subproblem PP
    alloc(globals.SP, Problem);
    alloc(globals.PP, Problem);

    MPI_Bcast(&(globals.SP->n), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // allocate memory for objective matrices for SP and PP
    alloc_matrix(globals.SP->L, globals.SP->n, double);
    alloc_matrix(globals.PP->L, globals.SP->n, double);

    MPI_Bcast(globals.SP->L, globals.SP->n * globals.SP->n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // IMPORTANT: last node is fixed to 0
    // --> main_problem_size is one less than the size of problem SP
    main_problem_size = globals.SP->n - 1; // num_vertices - 1;
    globals.PP->n = globals.SP->n;

    int N2 = globals.SP->n * globals.SP->n;
    int incx = 1;
    int incy = 1;
    dcopy_(&N2, globals.SP->L, &incx, globals.PP->L, &incy);

    // set global parameters
    set_parameters(params_in);
    if (params.adjust_TriIneq)
    {
        params.TriIneq = (main_problem_size + 1) * 10;
    }
    // Seed the random number generator
    srand(2020);

    // Provide B&B with an initial solution
    initializeBabSolution();

    // Allocate the memory
    allocMemory(&globals);
    // End Bab_Init

    // helper variables
    double g_lowerBound;
    /******************** WORKER PROCESS ********************/
    // receive diff
    if (params.use_diff)
        MPI_Bcast(&globals.diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // receive over (stop or continue)
    int over;
    MPI_Bcast(&over, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // receive lower bound
    if (over == -1 || params.root) // root node is pruned
        return over;
    else
        MPI_Bcast(&g_lowerBound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // update lower bound
    BabSolution solx;
    update_lower_bound(g_lowerBound, &solx);
    return over;
}

/// @brief Frees globals struct and heap
void worker_end()
{
    freeMemory(&globals);
    free(heap->data);
    free(heap);
}

/*************************************************************************/
/********************       WORKER MAIN LOOP          ********************/
/*************************************************************************/
///

/// @brief First communication check in worker loop, if over ends loop
/// @return 0 if not over, non-0 if over
int worker_check_over()
{
    MPI_Status status;
    int over;
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
    update_lower_bound(g_lowerBound, &solx);

    // start local queue
    pq_push(node);
}

/// @brief Ends evaluating if passed the time limit
/// @return 1 if limit reached, 0 if not
int time_limit_reached()
{
    return (params.time_limit > 0 && (MPI_Wtime() - globals.TIME) > params.time_limit);
}

/// @brief Update masters lowerbound, prune node or branch and distribute to other workers
/// @param node current node, already evaluated
/// @param old_lowerbound pre-evaluation lower bound of current worker
void after_evaluation(BabNode *node, double old_lowerbound)
{
    Message message;
    MPI_Status status;

    // check if better lower bound found --> update info with master
    if (get_lower_bound() > old_lowerbound)
    {

        message = NEW_VALUE;
        old_lowerbound = get_lower_bound();

        MPI_Send(&message, 1, MPI_INT, 0, MESSAGE, MPI_COMM_WORLD);
        MPI_Send(&old_lowerbound, 1, MPI_DOUBLE, 0, LOWER_BOUND, MPI_COMM_WORLD);
        MPI_Send(solution, 1, BabSolutiontype, 0, SOLUTION, MPI_COMM_WORLD);

        MPI_Recv(&old_lowerbound, 1, MPI_DOUBLE, 0, LOWER_BOUND, MPI_COMM_WORLD, &status);

        // update
        BabSolution solx;
        update_lower_bound(old_lowerbound, &solx);
    }
    /* if global_lower_bound + 1.0 < child_node->upper_bound,
     * then we must branch since there could be a better feasible
     * solution in this subproblem
     */
    if (get_lower_bound() + 1.0 < node->upper_bound)
    {

        /***** branch *****/

        // Determine the variable x[ic] to branch on
        int ic = get_branching_variable(node);

        BabNode *child_node;

        for (int xic = 0; xic <= 1; ++xic)
        {

            // Create a new child node from the parent node
            child_node = new_node(node);

            // split on node ic
            child_node->xfixed[ic] = 1;
            child_node->sol.X[ic] = xic;

            /* insert node into the priority queue */
            pq_push(child_node);
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

        update_lower_bound(g_lowerBound, &solx);

        // send subproblems to free workers
        if (num_free_workers != 0)
        { // free workers found

            for (int i = 0; i < num_free_workers; ++i)
            {

                // get next subproblem from queue and send it
                node = pq_pop();

                // send subproblem to free worker
                int over = 0;
                MPI_Send(&over, 1, MPI_INT, free_workers[i], OVER, MPI_COMM_WORLD);
                MPI_Send(&g_lowerBound, 1, MPI_DOUBLE, free_workers[i], LOWER_BOUND, MPI_COMM_WORLD);
                MPI_Send(node, 1, BabNodetype, free_workers[i], PROBLEM, MPI_COMM_WORLD);

                free(node);
            }
        }
    }
    else
    {
        // otherwise, intbound <= global_lower_bound, so we can prune
        free(node);
    }
}

/// @brief Once pq is empty notify master rank that worker is idle
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
Problem *get_PP(const Problem *SP)
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
GlobalVariables *init_globals(double *L, int num_vertices)
{
    GlobalVariables *globe = calloc(1, sizeof(GlobalVariables));
    if (params.adjust_TriIneq)
    {
        params.TriIneq = num_vertices * 10;
    }
    // allocate memory for original problem SP and subproblem PP
    alloc(globe->SP, Problem);
    alloc(globe->PP, Problem);
    globe->SP->n = num_vertices;
    globe->PP->n = num_vertices;
    // allocate memory for objective matrices for SP and PP
    // globals.SP->L = L;
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

void free_globals(GlobalVariables *globals_in)
{
    freeMemory(globals_in);
    free(globals_in);
}

GlobalVariables *get_globals_pointer()
{
    return &globals;
}