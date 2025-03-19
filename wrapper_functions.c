#include <stdio.h>
#include <mpi.h>
#include <stddef.h>
#include <string.h>

#include "biqbin.h"  

#define HEAP_SIZE 1000000
extern Heap *heap;
extern int BabPbSize;

extern BiqBinParameters params;
extern FILE *output;
extern GlobalVariables *globals;

int over = 0;
extern int num_workers_used; // number of worker processes used by the solver
int numbWorkers; // MPI comm size
int numbFreeWorkers;
int *busyWorkers;
int rank; // rank of each process: from 0 to numWorkers-1
int getRank() {return rank;}
/***** user defined MPI struct: for sending and receiving *****/
MPI_Datatype BabSolutiontype;
MPI_Datatype BabNodetype;

int initMPI(int argc, char** argv) {
    // MPI Start: start parallel environment
    MPI_Init(&argc, &argv);

    // get number of proccesses and corresponding ranks
    MPI_Comm_size(MPI_COMM_WORLD, &numbWorkers);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
	   printf("Number of cores: %d\n", numbWorkers);
	
    /***** user defined MPI struct: for sending and receiving *****/
    // (1) for BabSolution
    MPI_Datatype type1[1] = { MPI_INT };
    int blocklen1[1] = { NMAX };
    MPI_Aint disp1[1];
    disp1[0] = offsetof(BabSolution, X);
    MPI_Type_create_struct(1, blocklen1, disp1, type1, &BabSolutiontype);
    MPI_Type_commit(&BabSolutiontype);

    // (2) for BabNode
    MPI_Datatype type2[5] = { MPI_INT, BabSolutiontype, MPI_DOUBLE, MPI_INT, MPI_INT };
    int blocklen2[5] = { NMAX, 1, NMAX, 1, 1 };
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

void finalizeMPI() {
    MPI_Finalize();
}


// int initSolver(int argc, char** argv) {
//     globals = calloc(1, sizeof(GlobalVariables));
//     // Start the timer here or in compute?
//     globals->TIME = MPI_Wtime();


//     /* each process allocates its local priority queue */
//     heap = Init_Heap(HEAP_SIZE);

//     /* every process reads params and initializes B&B solution,
//         * only master process creates output file, reads input graph
//         * and broadcast it */
//     int read_error = Bab_Init(argc, argv, rank);
//     return read_error;
// }

int master_init(char* filename, double* L, int num_vertices, BiqBinParameters params_in) {
    globals = calloc(1, sizeof(GlobalVariables));
    // Start the timer here or in compute?
    globals->TIME = MPI_Wtime();
    /* each process allocates its local priority queue */
    heap = Init_Heap(HEAP_SIZE);

    // Bab_Init(argc, argv, rank) start
    openOutputFile(filename);
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
    allocMemory();
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

    if ( (over == -1) || params.root) {          
        return over;
    }
    else {
        g_lowerBound = Bab_LBGet();
        MPI_Bcast(&g_lowerBound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }


    // array of busy workers: 0 = free, 1 = busy
    // only master is busy
    busyWorkers = (int*)malloc(numbWorkers * sizeof(int));
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
    
    for (int xic = 0; xic <= 1; ++xic) { 

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

int master_main_loop() {
    MPI_Status status;
    Message message;
    /*** wait for messages: extract source from status ***/
    MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, MESSAGE, MPI_COMM_WORLD, &status);
    int source = status.MPI_SOURCE;

    master_Bab_Main(message, source, busyWorkers, numbWorkers, &numbFreeWorkers, BabSolutiontype);
    over = (numbFreeWorkers == numbWorkers - 1) ? 1 : 0;

    return  over; // If it returns 0 end it
}

void master_end() {
    // send over messages to the workers
    for(int i = 1; i < numbWorkers; ++i) {
        MPI_Send(&over, 1, MPI_INT, i, OVER, MPI_COMM_WORLD);
    }

    /* Print results to the standard output and to the output file */
    printFinalOutput(stdout,Bab_numEvalNodes());
    printFinalOutput(output,Bab_numEvalNodes());
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

// Worker receives the SP->L matrix and number of vertices, need params
int worker_init(BiqBinParameters params_in) {
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
    allocMemory();
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
    if (over == -1 || params.root )   // root node is pruned
        return over;
    else
        MPI_Bcast(&g_lowerBound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // update lower bound
    BabSolution solx;
    Bab_LBUpd(g_lowerBound, &solx);
    return over;
}

int worker_main_loop() {
    // helper variables
    BabNode *node;
    BabSolution solx;
    double g_lowerBound;
    MPI_Status status;
    Message message;
    /************* MAIN LOOP for worker **************/
    // wait for info: stop (from master) or receive new subproblem from other worker
    MPI_Recv(&over, 1, MPI_INT, MPI_ANY_SOURCE, OVER, MPI_COMM_WORLD, &status);
    if (!over) {

        alloc(node, BabNode);

        // receive subproblem from master or other worker
        MPI_Recv(&g_lowerBound, 1, MPI_DOUBLE, MPI_ANY_SOURCE, LOWER_BOUND, MPI_COMM_WORLD, &status);
        MPI_Recv(node, 1, BabNodetype, MPI_ANY_SOURCE, PROBLEM, MPI_COMM_WORLD, &status);

        // update
        Bab_LBUpd(g_lowerBound, &solx);

        // start local queue
        Bab_PQInsert(node);

        while(!isPQEmpty()){

            // check if time limit reached
            if (params.time_limit > 0 && (MPI_Wtime() - globals->TIME) > params.time_limit) {
                break;
            }

            worker_Bab_Main(BabSolutiontype, BabNodetype, rank);
        }

        message = IDLE;
        MPI_Send(&message, 1, MPI_INT, 0, MESSAGE, MPI_COMM_WORLD);
    }
    return over;
}

void worker_end() {
    /* free memory */
    Bab_End();
    free(heap->data);
    free(heap);
}

int worker_compute() {
    MPI_Status status;
    Message message;

    // tag to FINISH set to false
    int over = 0;
    // helper variables
    BabNode *node;
    double g_lowerBound;

     /******************** WORKER PROCESS ********************/
	// receive diff
	if (params.use_diff)
	    MPI_Bcast(&globals->diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);	

	// receive over (stop or continue)
	MPI_Bcast(&over, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // receive lower bound
	if (over == -1 || params.root )   // root node is pruned
	    goto FINISH;
	else
            MPI_Bcast(&g_lowerBound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// update lower bound
	BabSolution solx;
	Bab_LBUpd(g_lowerBound, &solx);
        
    /************* MAIN LOOP for worker **************/
    do {

        // wait for info: stop (from master) or receive new subproblem from other worker
        MPI_Recv(&over, 1, MPI_INT, MPI_ANY_SOURCE, OVER, MPI_COMM_WORLD, &status);

        if (!over) {

            alloc(node, BabNode);

            // receive subproblem from master or other worker
            MPI_Recv(&g_lowerBound, 1, MPI_DOUBLE, MPI_ANY_SOURCE, LOWER_BOUND, MPI_COMM_WORLD, &status);
            MPI_Recv(node, 1, BabNodetype, MPI_ANY_SOURCE, PROBLEM, MPI_COMM_WORLD, &status);
    
            // update
            Bab_LBUpd(g_lowerBound, &solx);

            // start local queue
            Bab_PQInsert(node);

            while(!isPQEmpty()){

                // check if time limit reached
                if (params.time_limit > 0 && (MPI_Wtime() - globals->TIME) > params.time_limit) {
                    break;
                }

                worker_Bab_Main(BabSolutiontype, BabNodetype, rank);
            }

            message = IDLE;
            MPI_Send(&message, 1, MPI_INT, 0, MESSAGE, MPI_COMM_WORLD);
        }
        
    } while (over != 1);

    FINISH:

    /* free memory */
    Bab_End();

    free(heap->data);
    free(heap);

    // MPI finish
    MPI_Finalize();

    return 0;
}


// /***************** Deprecated *******************/
// // Everything should be initialized by this point, rank and the rest are global atm
// int compute() {
//     MPI_Status status;
//     Message message;

//     // tag to FINISH set to false
//     int over = 0;
//     // helper variables
//     BabNode *node;
//     double g_lowerBound;

//     /******************** MASTER PROCESS ********************/
//     if (rank == 0)
//     {

//         // only master evaluates the root node
//         // and places it in priority queue if not able to prune
//         over = Init_PQ();

// 	    printf("Initial lower bound: %.0lf\n", Bab_LBGet());    

//         // broadcast diff
//         printf("diff = %f\n", globals->diff);
//         if (params.use_diff)
//             MPI_Bcast(&globals->diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);	

//             // broadcast lower bound to others or -1 to exit
//         MPI_Bcast(&over, 1, MPI_INT, 0, MPI_COMM_WORLD);

//         if ( (over == -1) || params.root) {          
//             goto FINISH;
//         }
//         else {
//             g_lowerBound = Bab_LBGet();
//             MPI_Bcast(&g_lowerBound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//         }


//         // array of busy workers: 0 = free, 1 = busy
//         // only master is busy
//         int busyWorkers[numbWorkers];
//         busyWorkers[0] = 1;
//         for (int i = 1; i < numbWorkers; ++i)
//             busyWorkers[i] = 0;

//         int numbFreeWorkers = numbWorkers - 1;
//         int source;


//         /***** branch root node and send to workers *****/
//         node = Bab_PQPop();

//         // Determine the variable x[ic] to branch on
//         int ic = getBranchingVariable(node);

//         // send two nodes to workers 1 and 2
//         BabNode *child_node;
//         int worker;
        
//         for (int xic = 0; xic <= 1; ++xic) { 

//             // Create a new child node from the parent node
//             child_node = newNode(node);

//             // split on node ic
//             child_node->xfixed[ic] = 1;
//             child_node->sol.X[ic] = xic;

//             // increment the number of explored nodes
//             Bab_incEvalNodes();

//             worker = xic + 1;
//             busyWorkers[worker] = 1;
//             --numbFreeWorkers;

//             MPI_Send(&over, 1, MPI_INT, worker, OVER, MPI_COMM_WORLD);
// 	        MPI_Send(&g_lowerBound, 1, MPI_DOUBLE, worker, LOWER_BOUND, MPI_COMM_WORLD);
//             MPI_Send(child_node, 1, BabNodetype, worker, PROBLEM, MPI_COMM_WORLD);

//             free(child_node);
//         }

//         // free parent nodes
//         free(node);    

// 	    num_workers_used = 2;

	
//         /************* MAIN LOOP for master **************/
//         do {

//             /*** wait for messages: extract source from status ***/
//             MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, MESSAGE, MPI_COMM_WORLD, &status);
//             source = status.MPI_SOURCE;

//             master_Bab_Main(message, source, busyWorkers, numbWorkers, &numbFreeWorkers, BabSolutiontype);

//         } while ( numbFreeWorkers != numbWorkers - 1 );
//         /*************************************************/

//         // send over messages to the workers
//         over = 1;
//         for(int i = 1; i < numbWorkers; ++i) {
//             MPI_Send(&over, 1, MPI_INT, i, OVER, MPI_COMM_WORLD);
//         }

//     }
//      /******************** WORKER PROCESS ********************/
//     else
//     {
// 	// receive diff
// 	if (params.use_diff)
// 	    MPI_Bcast(&globals->diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);	

// 	// receive over (stop or continue)
// 	MPI_Bcast(&over, 1, MPI_INT, 0, MPI_COMM_WORLD);

//         // receive lower bound
// 	if (over == -1 || params.root )   // root node is pruned
// 	    goto FINISH;
// 	else
//             MPI_Bcast(&g_lowerBound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// 	// update lower bound
// 	BabSolution solx;
// 	Bab_LBUpd(g_lowerBound, &solx);
        
//         /************* MAIN LOOP for worker **************/
//         do {

//             // wait for info: stop (from master) or receive new subproblem from other worker
//             MPI_Recv(&over, 1, MPI_INT, MPI_ANY_SOURCE, OVER, MPI_COMM_WORLD, &status);

//             if (!over) {

// 		        alloc(node, BabNode);

//                 // receive subproblem from master or other worker
// 		        MPI_Recv(&g_lowerBound, 1, MPI_DOUBLE, MPI_ANY_SOURCE, LOWER_BOUND, MPI_COMM_WORLD, &status);
//                 MPI_Recv(node, 1, BabNodetype, MPI_ANY_SOURCE, PROBLEM, MPI_COMM_WORLD, &status);
		
// 		        // update
// 		        Bab_LBUpd(g_lowerBound, &solx);

//                 // start local queue
//                 Bab_PQInsert(node);

//                 while(!isPQEmpty()){

//                     // check if time limit reached
//                     if (params.time_limit > 0 && (MPI_Wtime() - globals->TIME) > params.time_limit) {
//                         break;
//                     }

//                     worker_Bab_Main(BabSolutiontype, BabNodetype, rank);
//                 }

//                 message = IDLE;
//                 MPI_Send(&message, 1, MPI_INT, 0, MESSAGE, MPI_COMM_WORLD);
//             }
            
//         } while (over != 1);

//         //free(node);
//     }

//     FINISH:

//     /* Print results to the standard output and to the output file */
//     if (rank == 0) {
//         printFinalOutput(stdout,Bab_numEvalNodes());
//         printFinalOutput(output,Bab_numEvalNodes());
// 	    fprintf(output, "Number of cores: %d\n", numbWorkers);
// 	    fprintf(output, "Maximum number of workers used: %d\n", num_workers_used);
// 	    printf("Maximum number of workers used: %d\n", num_workers_used);
//         fclose(output);
//     }

//     /* free memory */
//     Bab_End();

//     free(heap->data);
//     free(heap);

//     // MPI finish
//     MPI_Finalize();

//     return 0;
// }

// int wrapped_main(int argc, char** argv) {
//     /*******************************************************
//     *********** BRANCH & BOUND: PARALLEL ALGORITHM ********
//     ******************************************************/
//     // number of processes = master + workers
//     int numbWorkers;

//     // rank of each process: from 0 to numWorkers-1
//     int rank;

//     // MPI Start: start parallel environment
//     MPI_Init(&argc, &argv);

//     // get number of proccesses and corresponding ranks
//     MPI_Comm_size(MPI_COMM_WORLD, &numbWorkers);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//     if (rank == 0)
// 	   printf("Number of cores: %d\n", numbWorkers);
	
//     /***** user defined MPI struct: for sending and receiving *****/
//     // (1) for BabSolution
//     MPI_Datatype BabSolutiontype;
//     MPI_Datatype type1[1] = { MPI_INT };
//     int blocklen1[1] = { NMAX };
//     MPI_Aint disp1[1];
//     disp1[0] = offsetof(BabSolution, X);
//     MPI_Type_create_struct(1, blocklen1, disp1, type1, &BabSolutiontype);
//     MPI_Type_commit(&BabSolutiontype);

//     // (2) for BabNode
//     MPI_Datatype BabNodetype;
//     MPI_Datatype type2[5] = { MPI_INT, BabSolutiontype, MPI_DOUBLE, MPI_INT, MPI_INT };
//     int blocklen2[5] = { NMAX, 1, NMAX, 1, 1 };
//     MPI_Aint disp2[5];
//     disp2[0] = offsetof(BabNode, xfixed);
//     disp2[1] = offsetof(BabNode, sol);
//     disp2[2] = offsetof(BabNode, fracsol);
//     disp2[3] = offsetof(BabNode, level);
//     disp2[4] = offsetof(BabNode, upper_bound);
//     MPI_Type_create_struct(5, blocklen2, disp2, type2, &BabNodetype);
//     MPI_Type_commit(&BabNodetype);
//     /***********************************/

//     // Start the timer
//     globals->TIME = MPI_Wtime();

//     MPI_Status status;
//     // type of message
//     Message message;

//     // tag to FINISH set to false
//     int over = 0;

//     // helper variables
//     BabNode *node;
//     double g_lowerBound;

//     /* each process allocates its local priority queue */
//     heap = Init_Heap(HEAP_SIZE);

//     /* every process reads params and initializes B&B solution,
//      * only master process creates output file, reads input graph
//      * and broadcast it */
//     int read_error = Bab_Init(argc, argv, rank);
	
//     if (read_error)
//         goto FINISH;
        
//     /******************** MASTER PROCESS ********************/
//     if (rank == 0)
//     {

//         // only master evaluates the root node
//         // and places it in priority queue if not able to prune
//         over = Init_PQ();

// 	    printf("Initial lower bound: %.0lf\n", Bab_LBGet());    

//         // broadcast diff
//         printf("diff = %f", globals->diff);
//         if (params.use_diff)
//             MPI_Bcast(&globals->diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);	

//             // broadcast lower bound to others or -1 to exit
//         MPI_Bcast(&over, 1, MPI_INT, 0, MPI_COMM_WORLD);

//         if ( (over == -1) || params.root) {          
//             goto FINISH;
//         }
//         else {
//             g_lowerBound = Bab_LBGet();
//             MPI_Bcast(&g_lowerBound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//         }


//         // array of busy workers: 0 = free, 1 = busy
//         // only master is busy
//         int busyWorkers[numbWorkers];
//         busyWorkers[0] = 1;
//         for (int i = 1; i < numbWorkers; ++i)
//             busyWorkers[i] = 0;

//         int numbFreeWorkers = numbWorkers - 1;
//         int source;


//         /***** branch root node and send to workers *****/
//         node = Bab_PQPop();

//         // Determine the variable x[ic] to branch on
//         int ic = getBranchingVariable(node);

//         // send two nodes to workers 1 and 2
//         BabNode *child_node;
//         int worker;
        
//         for (int xic = 0; xic <= 1; ++xic) { 

//             // Create a new child node from the parent node
//             child_node = newNode(node);

//             // split on node ic
//             child_node->xfixed[ic] = 1;
//             child_node->sol.X[ic] = xic;

//             // increment the number of explored nodes
//             Bab_incEvalNodes();

//             worker = xic + 1;
//             busyWorkers[worker] = 1;
//             --numbFreeWorkers;

//             MPI_Send(&over, 1, MPI_INT, worker, OVER, MPI_COMM_WORLD);
// 	        MPI_Send(&g_lowerBound, 1, MPI_DOUBLE, worker, LOWER_BOUND, MPI_COMM_WORLD);
//             MPI_Send(child_node, 1, BabNodetype, worker, PROBLEM, MPI_COMM_WORLD);

//             free(child_node);
//         }

//         // free parent nodes
//         free(node);    

// 	    num_workers_used = 2;

	
//         /************* MAIN LOOP for master **************/
//         do {

//             /*** wait for messages: extract source from status ***/
//             MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, MESSAGE, MPI_COMM_WORLD, &status);
//             source = status.MPI_SOURCE;

//             master_Bab_Main(message, source, busyWorkers, numbWorkers, &numbFreeWorkers, BabSolutiontype);

//         } while ( numbFreeWorkers != numbWorkers - 1 );
//         /*************************************************/

//         // send over messages to the workers
//         over = 1;
//         for(int i = 1; i < numbWorkers; ++i) {
//             MPI_Send(&over, 1, MPI_INT, i, OVER, MPI_COMM_WORLD);
//         }

//     }
//      /******************** WORKER PROCESS ********************/
//     else
//     {
// 	// receive diff
// 	if (params.use_diff)
// 	    MPI_Bcast(&globals->diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);	

// 	// receive over (stop or continue)
// 	MPI_Bcast(&over, 1, MPI_INT, 0, MPI_COMM_WORLD);

//         // receive lower bound
// 	if (over == -1 || params.root )   // root node is pruned
// 	    goto FINISH;
// 	else
//             MPI_Bcast(&g_lowerBound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// 	// update lower bound
// 	BabSolution solx;
// 	Bab_LBUpd(g_lowerBound, &solx);
        
//         /************* MAIN LOOP for worker **************/
//         do {

//             // wait for info: stop (from master) or receive new subproblem from other worker
//             MPI_Recv(&over, 1, MPI_INT, MPI_ANY_SOURCE, OVER, MPI_COMM_WORLD, &status);

//             if (!over) {

// 		        alloc(node, BabNode);

//                 // receive subproblem from master or other worker
// 		        MPI_Recv(&g_lowerBound, 1, MPI_DOUBLE, MPI_ANY_SOURCE, LOWER_BOUND, MPI_COMM_WORLD, &status);
//                 MPI_Recv(node, 1, BabNodetype, MPI_ANY_SOURCE, PROBLEM, MPI_COMM_WORLD, &status);
		
// 		        // update
// 		        Bab_LBUpd(g_lowerBound, &solx);

//                 // start local queue
//                 Bab_PQInsert(node);

//                 while(!isPQEmpty()){

//                     // check if time limit reached
//                     if (params.time_limit > 0 && (MPI_Wtime() - globals->TIME) > params.time_limit) {
//                         break;
//                     }

//                     worker_Bab_Main(BabSolutiontype, BabNodetype, rank);
//                 }

//                 message = IDLE;
//                 MPI_Send(&message, 1, MPI_INT, 0, MESSAGE, MPI_COMM_WORLD);
//             }
            
//         } while (over != 1);

//         //free(node);
//     }

//     FINISH:

//     /* Print results to the standard output and to the output file */
//     if (rank == 0) {
//         printFinalOutput(stdout,Bab_numEvalNodes());
//         printFinalOutput(output,Bab_numEvalNodes());
// 	    fprintf(output, "Number of cores: %d\n", numbWorkers);
// 	    fprintf(output, "Maximum number of workers used: %d\n", num_workers_used);
// 	    printf("Maximum number of workers used: %d\n", num_workers_used);
//         fclose(output);
//     }

//     /* free memory */
//     Bab_End();

//     free(heap->data);
//     free(heap);

//     // MPI finish
//     MPI_Finalize();

//     return 0;
// }

// int master_compute() {
//     MPI_Status status;
//     Message message;

//     // tag to FINISH set to false
//     int over = 0;
//     // helper variables
//     BabNode *node;
//     double g_lowerBound;

//     /******************** MASTER PROCESS ********************/
//     if (rank == 0)
//     {

//         // only master evaluates the root node
//         // and places it in priority queue if not able to prune
//         over = Init_PQ();

// 	    printf("Initial lower bound: %.0lf\n", Bab_LBGet());    

//         // broadcast diff
//         printf("diff = %f", globals->diff);
//         if (params.use_diff)
//             MPI_Bcast(&globals->diff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);	

//             // broadcast lower bound to others or -1 to exit
//         MPI_Bcast(&over, 1, MPI_INT, 0, MPI_COMM_WORLD);

//         if ( (over == -1) || params.root) {          
//             goto FINISH;
//         }
//         else {
//             g_lowerBound = Bab_LBGet();
//             MPI_Bcast(&g_lowerBound, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//         }


//         // array of busy workers: 0 = free, 1 = busy
//         // only master is busy
//         // int busyWorkers[numbWorkers];
//         busyWorkers = (int*)malloc(numbWorkers * sizeof(int));
//         busyWorkers[0] = 1;
//         for (int i = 1; i < numbWorkers; ++i)
//             busyWorkers[i] = 0;

//         int numbFreeWorkers = numbWorkers - 1;
//         int source;


//         /***** branch root node and send to workers *****/
//         node = Bab_PQPop();

//         // Determine the variable x[ic] to branch on
//         int ic = getBranchingVariable(node);

//         // send two nodes to workers 1 and 2
//         BabNode *child_node;
//         int worker;
        
//         for (int xic = 0; xic <= 1; ++xic) { 

//             // Create a new child node from the parent node
//             child_node = newNode(node);

//             // split on node ic
//             child_node->xfixed[ic] = 1;
//             child_node->sol.X[ic] = xic;

//             // increment the number of explored nodes
//             Bab_incEvalNodes();

//             worker = xic + 1;
//             busyWorkers[worker] = 1;
//             --numbFreeWorkers;

//             MPI_Send(&over, 1, MPI_INT, worker, OVER, MPI_COMM_WORLD);
// 	        MPI_Send(&g_lowerBound, 1, MPI_DOUBLE, worker, LOWER_BOUND, MPI_COMM_WORLD);
//             MPI_Send(child_node, 1, BabNodetype, worker, PROBLEM, MPI_COMM_WORLD);

//             free(child_node);
//         }

//         // free parent nodes
//         free(node);    

// 	    num_workers_used = 2;

	
//         /************* MAIN LOOP for master **************/
//         do {

//             /*** wait for messages: extract source from status ***/
//             MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, MESSAGE, MPI_COMM_WORLD, &status);
//             source = status.MPI_SOURCE;

//             master_Bab_Main(message, source, busyWorkers, numbWorkers, &numbFreeWorkers, BabSolutiontype);

//         } while ( numbFreeWorkers != numbWorkers - 1 );
//         /*************************************************/

//         // send over messages to the workers
//         over = 1;
//         for(int i = 1; i < numbWorkers; ++i) {
//             MPI_Send(&over, 1, MPI_INT, i, OVER, MPI_COMM_WORLD);
//         }

//     }

//     FINISH:

//     /* Print results to the standard output and to the output file */
//     printFinalOutput(stdout,Bab_numEvalNodes());
//     printFinalOutput(output,Bab_numEvalNodes());
//     fprintf(output, "Number of cores: %d\n", numbWorkers);
//     fprintf(output, "Maximum number of workers used: %d\n", num_workers_used);
//     printf("Maximum number of workers used: %d\n", num_workers_used);
//     fclose(output);

//     /* free memory */
//     Bab_End();

//     free(busyWorkers);
//     free(heap->data);
//     free(heap);

//     // MPI finish
//     MPI_Finalize();

//     return 0;
// }