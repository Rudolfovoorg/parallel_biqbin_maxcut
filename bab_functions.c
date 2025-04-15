#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include "biqbin.h"
  
  /* defined in heap.c */  
extern Heap *heap;  
extern int main_problem_size;           
extern BabSolution *solution;
   
extern BiqBinParameters params;
extern GlobalVariables globals;

extern int num_workers_used;

/* initialize global lower bound to 0 and global solution vector to zero */
void initializeBabSolution() { 

    BabSolution bs;

    for (int i = 0; i < main_problem_size; ++i) {
        bs.X[i] = 0;
    }

    init_solution_lb(0, &bs);
}

/* NOTE: int *sol in functions evaluateSolution and updateSolution have length main_problem_size
 * -> to get objecive multiple with Laplacian that is stored in upper left corner of SP->L
 */
double evaluateSolution(const int *sol, const Problem *SP) {

    double val = 0.0;
    int problem_size = SP->n - 1;
    for (int i = 0; i < problem_size; ++i) {
        for (int j = 0; j < problem_size; ++j) {
            val += SP->L[j + i * SP->n] * sol[i] * sol[j];
        }
    }

    return val;
}


/*
 * Only this function can update best solution and value.
 * Returns 1 if success.
 */
int updateSolution(const int *x, const Problem *SP) {
    
    int solutionAdded = 0;
    double sol_value;
    BabSolution solx;

    // Copy x into solx --> because update_lower_bound needs BabSolution and not int*
    for (int i = 0; i < SP->n - 1; ++i) {
      solx.X[i] = x[i];
    }

    sol_value = evaluateSolution(x, SP); // computes objective value of solx

    /* If new solution is better than the global solution,
     * then update and print the new solution. */
    if (update_lower_bound(sol_value, &solx)) {
        solutionAdded = 1;
    }
    
    return solutionAdded;
}


/* MASTER process main routine */
void master_Bab_Main(Message message, int source, int *busyWorkers, int numbWorkers, int *numbFreeWorkers, MPI_Datatype BabSolutiontype) {

    // If the algorithm stops before finding the optimal solution
    if (!globals.stopped && (params.time_limit > 0 && (MPI_Wtime() - globals.TIME) > params.time_limit) ) {
        
        // signal to printFinalOutput that algorihtm stopped early
        globals.stopped = 1;
    }

    MPI_Status status;

    switch(message) {

        case IDLE:

            busyWorkers[source] = 0;
            ++(*numbFreeWorkers);
            break;  

        case NEW_VALUE:
        {
        
            // receive best lower bound and corresponding feasible solution
            double g_lowerBound;
            BabSolution solx;
            
            MPI_Recv(&g_lowerBound, 1, MPI_DOUBLE, source, LOWER_BOUND, MPI_COMM_WORLD, &status);
            MPI_Recv(&solx, 1, BabSolutiontype, source, SOLUTION, MPI_COMM_WORLD, &status);  

            if ( update_lower_bound(g_lowerBound, &solx) ){
                printf("Feasible solution %.0lf\n", get_lower_bound());
            }
            
            // send update information back to worker
            g_lowerBound = get_lower_bound();

            MPI_Send(&g_lowerBound, 1, MPI_DOUBLE, source, LOWER_BOUND, MPI_COMM_WORLD);
            break;       
        }
        case SEND_FREEWORKERS:
        {            
            // get number of requested workers            
            int workers_request;                
            MPI_Recv(&workers_request, 1, MPI_INT, source, FREEWORKER, MPI_COMM_WORLD, &status);
                        
            // compute number of freeworkers
            int num_workers_available = (workers_request < *numbFreeWorkers) ? workers_request : *numbFreeWorkers;
            int available_workers[num_workers_available];

            for(int i = 1, j = 0; (i < numbWorkers) && (j < num_workers_available); ++i)    // master has rank 0 and is not considered
            {
                if(busyWorkers[i] == 0){ // is free
                    available_workers[j] = i;
                    ++j;
                    busyWorkers[i] = 1; // set to busy
                    --(*numbFreeWorkers);
                }
            }
    

            // worker branched subproblem in local queue --> add 2 bab nodes
            increase_num_eval_nodes();
            increase_num_eval_nodes(); 

	        // count current number of busy workers
	        int current_busy = 0;

	        for (int i = 1; i < numbWorkers; ++i) {
		       if (busyWorkers[i] == 1)
		       ++current_busy;
	        } 
	
	        num_workers_used = (current_busy > num_workers_used) ? current_busy : num_workers_used;

	        // send message back
            double g_lowerBound = get_lower_bound();            
            MPI_Send(&num_workers_available, 1, MPI_INT, source, NUM_FREE_WORKERS, MPI_COMM_WORLD);              
            MPI_Send(available_workers, num_workers_available, MPI_INT, source, FREEWORKER, MPI_COMM_WORLD);
            MPI_Send(&g_lowerBound, 1, MPI_DOUBLE, source, LOWER_BOUND, MPI_COMM_WORLD);     
            break;
        }
    }
}

/* print solution 0-1 vector */
void printSolution(FILE *file) {

    fprintf(file, "Solution = ( ");
    for (int i = 0; i < main_problem_size; ++i) {
        if (solution->X[i] == 1) {
            fprintf(file, "%d ", i + 1);
        }
    }
    fprintf(file, ")\n");
}


/* print final output */
void printFinalOutput(FILE *file, int num_nodes) {

    // Best solution found
    double best_sol = get_lower_bound();

    fprintf(file, "\nNodes = %d\n", num_nodes);
    
    // normal termination
    if (!globals.stopped) {
        fprintf(file, "Root node bound = %.2lf\n", globals.root_bound);
        fprintf(file, "Maximum value = %.0lf\n", best_sol);
        
    } else { // B&B stopped early
        fprintf(file, "TIME LIMIT REACHED.\n");
        fprintf(file, "Root node bound = %.2lf\n", globals.root_bound); 
        fprintf(file, "Best value = %.0lf\n", best_sol);
    }

    printSolution(file);
    fprintf(file, "Time = %.2f s\n\n", MPI_Wtime() - globals.TIME);
}


/* Bab function called at the end of the execution.
 * This function frees the memory allocated by the program. */
void Bab_End(void) {
    freeMemory(&globals);   
}


/*
 * getBranchingVariable function used in the Bab_GenChild routine to determine
 * which variable x[ic] to branch on.
 *
 * node: the current node of the branch-and-bound search tree
 */
int getBranchingVariable(BabNode *node) {

    int ic = -1;  // x[ic] is the variable to branch on
    double maxValue, minValue;

    /* 
     * Choose the branching variable x[ic] based on params.branchingStrategy
     */
    if (params.branchingStrategy == LEAST_FRACTIONAL) {
        // Branch on the variable x[ic] that has the least fractional value
        maxValue = -BIG_NUMBER;
        for (int i = 0; i < main_problem_size; ++i) {
            if (!(node->xfixed[i]) && fabs(0.5 - node->fracsol[i]) > maxValue) {
                ic = i;
                maxValue = fabs(0.5 - node->fracsol[ic]);
            }
        }
    }
    else if (params.branchingStrategy == MOST_FRACTIONAL) {
        // Branch on the variable x[ic] that has the most fractional value
        minValue = BIG_NUMBER;
        for (int i = 0; i < main_problem_size; ++i) {
            if (!(node->xfixed[i]) && fabs(0.5 - node->fracsol[i]) < minValue) {
                ic = i;
                minValue = fabs(0.5 - node->fracsol[ic]);
            }
        }
    }
    else {
        fprintf(stderr, "Error: Wrong value for params.branchingStrategy\n");
        MPI_Abort(MPI_COMM_WORLD,10);
    }

    return ic;
}


/* Count the number of fixed variables */
int countFixedVariables(BabNode *node) {
    
    int numFixedVariables = 0;

    for (int i = 0; i < main_problem_size; ++i) {
        if (node->xfixed[i]) {
            ++numFixedVariables;
        }
    }

    return numFixedVariables;
}
