#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "biqbin.h"

/**
 * @brief Performs a simple local search starting from the given feasible solution x.
 * @param x stores feasible solution x that is locally optimal.
 * @param P: subproblem Problem *PP
 * 
 * @note This function is working in {-1,1} model!
 */
static double mc_1opt(int *x, const Problem *P) {

    int N = P->n;

    double *Lx, *d, *delta;
    int *I;
    alloc_vector(Lx, N, double);
    alloc_vector(d, N, double);
    alloc_vector(delta, N, double);
    alloc_vector(I, N, int);


    // Lx = L*x
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            Lx[i] += P->L[j + i*N] * x[j];

    // d = diag(L);
    // cost = x'*Lx
    // delta = d - x.*Lx
    double cost = 0.0;
    
    for (int i = 0; i < N; ++i) {
        d[i] = P->L[i + i * N];
        cost += x[i] * Lx[i];
        delta[i] = d[i] - x[i] * Lx[i];
    }

    // [best, i] = max(delta);
    double best = -BIG_NUMBER;
    int index = 0;

    for (int i = 0; i < N; ++i) {

        if (delta[i] > best) {
            best = delta[i];
            index = i;
        }

    }

    int num_I;      // number of elements in I

    /*** main loop ***/
    while (best > 0.001) {


        // I = find(L(:,index))
        num_I = 0;
        for (int j = 0; j < N; ++j) {
            
            if ( fabs(P->L[index + N * j]) > 0.001 ) { // add to I
                I[num_I] = j;
                ++num_I;
            }
        
        }

        if (x[index] > 0) { // Lx(I) = Lx(I)  - 2 *L(I,index);
            for (int i = 0; i < num_I; ++i) {
                Lx[I[i]] -= 2 * P->L[index + I[i] * N];
            }
        }
        else { // Lx(I) = Lx(I)  + 2 *L(I,index);
            for (int i = 0; i < num_I; ++i) {
                Lx[I[i]] += 2 * P->L[index + I[i] * N];
            }
        }

        // update new cut: x(index) = -x(index) 
        x[index] *= -1;

        // update weight of cut: cost = cost + 4*best
        cost += 4 * best;

        // update new differences: delta = d - x.*Lx
        for (int i = 0; i < N; ++i) {
            delta[i] = d[i] - x[i] * Lx[i];
        }

        // find new champion: [best, i] = max(delta) 
        best = -BIG_NUMBER;
        index = 0;

        for (int i = 0; i < N; ++i) {

            if (delta[i] > best) {
                best = delta[i];
                index = i;
            }

        }

    }

    free(Lx);
    free(d);
    free(delta);
    free(I);

    return cost;
}


/**
 * @brief copies xnew into xbest if xnews solution is better
 * @param xbest current best solution of heuristics
 * @param xnew new solution, gets evaluated inside the function and copied into xbest if it is better
 * @param best pointer to the current best heuristic value (lower bound) of xbest, value gets updated with xnew's heuristic value
 * @param P0: main Problem *SP (declared in global_var.h) only n is read to get the problem size
 */
static int update_best(int *xbest, const int *xnew, double *best, const Problem *P0) {

    int success = 0;
    int N = P0->n - 1; // N = main_problem_size

    double heur_val = evaluate_solution(xnew, P0);

    if ( *best < heur_val ) {
        memcpy(xbest, xnew, sizeof(int) * N);
        *best = heur_val;
        success = 1;
    }

    return success;
}

/**
 * @brief Goemans-Williamson random hyperplane heuristic.
 *
 * @param P0 The original Problem *SP struct, only uses *L (double* L matrix) and n (int number of nodes)
 * @param P  The current subproblem Problem *PP
 * @param node The current BabNode structure, uses only xfixed (int[] array) and sol.X (solution structures int[] X solution nodes).
 * @param x Only value that changes, stores the solution of the heuristic (size is the main problem size: P0->n-1).
 * @param num The number of random hyperplanes to try, set to the number of nodes.
 * @return heurestic value, calculated lower bound of best solution found by GW heurestic.
 * 
 * @note from both P0 and P only int n and double *L values are used
 */
double GW_heuristic(const Problem *P0, const Problem *P, const BabNode *node, int *x, int num, const double *Z) {

    // Problem *P0 ... the original problem - int num vertices
    // Problem *P  ... the current subproblem
    //         num ... number of random hyperplanes

    int index;
    int N = P->n;

    // (local) temporary vector of size X
    int temp_x[N];                    

    // (global) temporary vector of size main_problem_size to store heuristic solutions
    int sol[P0->n - 1];                 

    double sca;                         // dot product of random vector v and col of Z
    double best = -BIG_NUMBER;          // best lower bound found
    double v[N];                        // defines random hyperplane v   


    for (int count = 0; count < num; ++count) {

        // compute random hyperplane v
        for (int i = 0; i < N; ++i) 
            v[i] = ( (double)rand() / (double)(RAND_MAX) ) - 0.5;

        // compute cut temp_x generated by hyperplane v
        index = 0;
        for (int i = 0; i < N; ++i) {

                sca = 0.0;
                for (int j = 0; j < N; ++j)
                    sca += v[j] * Z[j * N + index];

                if (sca < 0) {

                    temp_x[i] = -1;
                    
                }
                else {

                    temp_x[i] = 1;
                    
                }

                ++index;            
        }

        // improve feasible solution through 1-opt
        mc_1opt(temp_x, P);

        // store local cut temp_x into global cut sol
        index = 0;
        for (int i = 0; i < P0->n-1; ++i) {
            if (node->xfixed[i]) 
                sol[i] = node->sol.X[i];
            else {
                sol[i] = (temp_x[index]+1)/2;
                ++index;
            }
        }
        // replace x if a new solution is better
        update_best(x, sol, &best, P0);
    }

    return best;
}