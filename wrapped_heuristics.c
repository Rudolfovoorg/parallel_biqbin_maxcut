#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "biqbin.h"


HeurState* heuristic_init(const Problem *P0, const Problem *P, const BabNode *node, const double *X, double *Z)
{
    HeurState* state = (HeurState*)malloc(sizeof(HeurState));
    state->n = P->n;
    state->N = P0->n - 1;
    state->nn = state->n * state->n;
    state->inc = 1;
    state->UPLO = 'L';

    state->xh = (double *)malloc(state->n * sizeof(double));
    state->temp_x = (int *)malloc(state->N * sizeof(int));

    
    // generate first random cut vector {-1,1}^n
    for (int i = 0; i < state->n; ++i)
        state->xh[i] = 2 * (rand() % 2) - 1; 

    // compute its objective value (store in temp_x and transform to {0,1})
    int index = 0;
    for (int i = 0; i < state->N; ++i) {

        if (node->xfixed[i]) 
            state->temp_x[i] = node->sol.X[i];

        else {
            state->temp_x[i] = (state->xh[index] + 1) / 2.0;
            ++index;
        }
    }    
    state->fh = evaluateSolution(state->temp_x, P0);
    // Z = X
    dcopy_(&state->nn, X, &state->inc, Z, &state->inc);
    return state;
}

int cholesky_factorization(HeurState *state, double *Z)
{
    int info;
    dpotrf_(&state->UPLO, &state->n, Z, &state->n, &info);

    if (info != 0)
    {
        fprintf(stderr, "%s: Cholesky factorization failed (line %d).\n", __func__, __LINE__);
        MPI_Abort(MPI_COMM_WORLD, 10);
    }

    // Zero lower triangle
    for (int i = 0; i < state->n; ++i)
        for (int j = 0; j < i; ++j)
            Z[j + i * state->n] = 0.0;

    return info;
}

int heuristic_postprocess(HeurState *state, const BabNode *node, const int *x, const double *X, double *Z, double heur_val)
{
    int success = 0;
    if (heur_val > state->fh)
    {
        state->fh = heur_val;

        // copy global cut vector x into xh
        // NOTE: skip fixed verticesmake
        int index = 0;
        for (int i = 0; i < state->N; ++i) {

            if (!node->xfixed[i]) {
                state->xh[index] = 2 * x[i] - 1;
                ++index;
            }
        }
        state->xh[state->n-1] = -1.0;                  // last vertex in original is fixed to 0
        success = 1;
    }

    double constant = 0.3 + 0.6 * ((double)rand() / (double)RAND_MAX);
    double alpha = 1.0 - constant;

    dcopy_(&state->nn, X, &state->inc, Z, &state->inc);
    dscal_(&state->nn, &alpha, Z, &state->inc);

    alpha = constant;
    dsyr_(&state->UPLO, &state->n, &alpha, state->xh, &state->inc, Z, &state->n);
    return success;
}

double heuristic_finalize(HeurState *state)
{
    double result = state->fh;
    free(state->xh);
    free(state->temp_x);
    return result;
}
