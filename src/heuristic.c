#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <float.h> // ADDED: DBL_EPSILON

#include "biqbin.h"
#include "wrapper_hooks.h"

extern double *X;
extern double *Z; // stores Cholesky decomposition: X = ZZ^T
extern int BabPbSize;

double runHeuristic(const Problem *P0, Problem *P, BabNode *node)
{
#ifdef PURE_C
    int x[BabPbSize];

    for (int i = 0; i < BabPbSize; ++i)
    {
        x[i] = node->xfixed[i] ? node->sol.X[i] : 0;
    }
    double heur_obj_value = runHeuristic_unpacked(P0->L, P0->n, P->L, P->n, node->xfixed, node->sol.X, x);
    updateSolution(x);
    return heur_obj_value;
#else
    return wrapped_heuristic(P0, P, node);
#endif
}

static void factor_for_gw_inplace(double *Z, int n)
{
    int nn = n * n;
    int inc = 1;
    char UPLO = 'L';
    int info = 0;

    // ADDED:
    // Save a copy because dpotrf overwrites Z even if it fails.
    double Z_backup[nn];
    dcopy_(&nn, Z, &inc, Z_backup, &inc);

    // ORIGINAL BEHAVIOR FIRST:
    // Try Cholesky exactly like before.
    dpotrf_(&UPLO, &n, Z, &n, &info);

    if (info == 0)
    {
        // ORIGINAL BEHAVIOR:
        // Keep only the lower-triangular Cholesky factor.
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < i; ++j)
                Z[j + i * n] = 0.0;

        return;
    }

    // ADDED:
    // Fallback path only when Cholesky fails.
    fprintf(stderr,
            "%s: dpotrf failed: info=%d, n=%d -- falling back to dsyev\n",
            __func__, info, n);

    // ADDED:
    // Restore the original matrix before eigendecomposition.
    dcopy_(&nn, Z_backup, &inc, Z, &inc);

    // ADDED:
    // Compute eigendecomposition Z = V * diag(eig) * V^T
    char JOBZ = 'V';
    int lwork = -1;
    double work_query;
    double eig[n];
    int eig_info = 0;

    // Workspace query
    dsyev_(&JOBZ, &UPLO, &n, Z, &n, eig, &work_query, &lwork, &eig_info);

    if (eig_info != 0)
    {
        fprintf(stderr,
                "%s: dsyev workspace query failed: info=%d, n=%d\n",
                __func__, eig_info, n);
        MPI_Abort(MPI_COMM_WORLD, 10);
    }

    lwork = (int)work_query;
    double work[lwork];

    // Actual eigendecomposition
    dsyev_(&JOBZ, &UPLO, &n, Z, &n, eig, work, &lwork, &eig_info);

    if (eig_info != 0)
    {
        fprintf(stderr,
                "%s: dsyev failed: info=%d, n=%d\n",
                __func__, eig_info, n);
        MPI_Abort(MPI_COMM_WORLD, 10);
    }

    // ADDED:
    // Z currently contains eigenvectors in columns.
    // Turn it into Z = V * sqrt(Lambda), so that Z * Z^T = original matrix.
    double lambda_max = fmax(0.0, eig[n - 1]);
    double eig_tol = n * DBL_EPSILON * fmax(1.0, lambda_max);

    for (int j = 0; j < n; ++j)
    {
        double lambda = eig[j];

        // Accept tiny negative eigenvalues as numerical noise.
        if (lambda < -eig_tol)
        {
            fprintf(stderr,
                    "%s: fallback found non-PSD matrix: eig[%d]=%.17e, tol=%.17e, n=%d\n",
                    __func__, j, lambda, eig_tol, n);
            MPI_Abort(MPI_COMM_WORLD, 10);
        }

        if (lambda <= eig_tol)
            lambda = 0.0;

        double scale = sqrt(lambda);

        for (int i = 0; i < n; ++i)
            Z[i + j * n] *= scale;
    }
}

double runHeuristic_unpacked(const double *P0_L, int P0_N, const double *P_L, int P_N, const int *node_xfixed, const int *node_sol_X, int *x)
{
    // Problem *P0 ... the original problem
    // Problem *P  ... the current subproblem
    // int *x      ... current best feasible solution

    int n = P_N;
    int N = P0_N - 1; // BabPbSize
    int nn = n * n;
    int inc = 1;
    char UPLO = 'L';
    double heur_val;

    double xh[n];  // is used for convex combination with matrix X (n x n)
    int temp_x[N]; // stores xh + some variables are fixed in {0,1} model
    int index;

    // generate first random cut vector {-1,1}^n
    for (int i = 0; i < n; ++i)
        xh[i] = 2 * (rand() % 2) - 1;

    // compute its objective value (store in temp_x and transform to {0,1})
    index = 0;
    for (int i = 0; i < N; ++i)
    {
        if (node_xfixed[i])
            temp_x[i] = node_sol_X[i];
        else
        {
            temp_x[i] = (xh[index] + 1) / 2.0;
            ++index;
        }
    }

    double fh = evaluateSolution(temp_x);

    int done = 0;
    double constant; // scalar in convex combination
    double alpha;

    // Z = X
    dcopy_(&nn, X, &inc, Z, &inc);

    while (done < 2)
    {
        ++done;

        // CHANGED:
        // Old code directly did dpotrf here and aborted on failure.
        // New code:
        //   1) tries the original Cholesky path
        //   2) falls back to dsyev only if Cholesky fails
        factor_for_gw_inplace(Z, n);

        // Goemans-Williamson heuristic
        heur_val = GW_heuristic(P0_L, P0_N, P_L, P_N, node_xfixed, node_sol_X, x, P0_N);

        if (heur_val > fh)
        {
            done = 0;
            fh = heur_val;

            // copy global cut vector x into xh
            // NOTE: skip fixed vertices
            index = 0;
            for (int i = 0; i < N; ++i)
            {
                if (!node_xfixed[i])
                {
                    xh[index] = 2 * x[i] - 1;
                    ++index;
                }
            }

            xh[n - 1] = -1.0; // last vertex in original is fixed to 0
        }

        constant = 0.3 + 0.6 * ((double)rand() / (double)(RAND_MAX));

        // Z = (1-constant)*X + constant* xh * xh'
        alpha = 1.0 - constant;
        dcopy_(&nn, X, &inc, Z, &inc);
        dscal_(&nn, &alpha, Z, &inc);
        alpha = constant;
        dsyr_(&UPLO, &n, &alpha, xh, &inc, Z, &n);
    }

    return heur_val;
}

/* Goemans-Williamson random hyperplane heuristic */
double GW_heuristic(const double *P0_L, int P0_N, const double *P_L, int P_N, const int *node_xfixed, const int *node_sol_X, int *x, int num)
{

    // Problem *P0 ... the original problem
    // Problem *P  ... the current subproblem
    //         num ... number of random hyperplanes

    int index;
    int N = P_N;

    // (local) temporary vector of size X
    int temp_x[N];

    // (global) temporary vector of size BabPbSize to store heuristic solutions
    int sol[P0_N - 1];

    double sca;              // dot product of random vector v and col of Z
    double best = -INFINITY; // best lower bound found
    double v[N];             // defines random hyperplane v

    for (int count = 0; count < num; ++count)
    {

        // compute random hyperplane v
        for (int i = 0; i < N; ++i)
            v[i] = ((double)rand() / (double)(RAND_MAX)) - 0.5;

        // compute cut temp_x generated by hyperplane v
        index = 0;
        for (int i = 0; i < N; ++i)
        {

            sca = 0.0;
            for (int j = 0; j < N; ++j)
                sca += v[j] * Z[j * N + index];

            if (sca < 0)
            {

                temp_x[i] = -1;
            }
            else
            {

                temp_x[i] = 1;
            }

            ++index;
        }

        // improve feasible solution through 1-opt
        mc_1opt(temp_x, P_L, P_N);

        // store local cut temp_x into global cut sol
        index = 0;
        for (int i = 0; i < P0_N - 1; ++i)
        {
            if (node_xfixed[i])
                sol[i] = node_sol_X[i];
            else
            {
                sol[i] = (temp_x[index] + 1) / 2;
                ++index;
            }
        }

        update_best(x, sol, &best, P0_N);
    }

    return best;
}

/*
 * Performs a simple local search starting from the given feasible solution x.
 * Returns a feasible solution x that is locally optimal.
 * The objective value of x is returned.
 */
// NOTE: this function is working in {-1,1} model!
double mc_1opt(int *x, const double *P_L, int P_N)
{

    int N = P_N;

    double *Lx, *d, *delta;
    int *I;
    alloc_vector(Lx, N, double);
    alloc_vector(d, N, double);
    alloc_vector(delta, N, double);
    alloc_vector(I, N, int);

    // Lx = L*x
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            Lx[i] += P_L[j + i * N] * x[j];

    // d = diag(L);
    // cost = x'*Lx
    // delta = d - x.*Lx
    double cost = 0.0;

    for (int i = 0; i < N; ++i)
    {
        d[i] = P_L[i + i * N];
        cost += x[i] * Lx[i];
        delta[i] = d[i] - x[i] * Lx[i];
    }

    // [best, i] = max(delta);
    double best = -INFINITY;
    int index = 0;

    for (int i = 0; i < N; ++i)
    {

        if (delta[i] > best)
        {
            best = delta[i];
            index = i;
        }
    }

    int num_I; // number of elements in I

    /*** main loop ***/
    while (best > 0.001)
    {

        // I = find(L(:,index))
        num_I = 0;
        for (int j = 0; j < N; ++j)
        {

            if (fabs(P_L[index + N * j]) > 0.001)
            { // add to I
                I[num_I] = j;
                ++num_I;
            }
        }

        if (x[index] > 0)
        { // Lx(I) = Lx(I)  - 2 *L(I,index);
            for (int i = 0; i < num_I; ++i)
            {
                Lx[I[i]] -= 2 * P_L[index + I[i] * N];
            }
        }
        else
        { // Lx(I) = Lx(I)  + 2 *L(I,index);
            for (int i = 0; i < num_I; ++i)
            {
                Lx[I[i]] += 2 * P_L[index + I[i] * N];
            }
        }

        // update new cut: x(index) = -x(index)
        x[index] *= -1;

        // update weight of cut: cost = cost + 4*best
        cost += 4 * best;

        // update new differences: delta = d - x.*Lx
        for (int i = 0; i < N; ++i)
        {
            delta[i] = d[i] - x[i] * Lx[i];
        }

        // find new champion: [best, i] = max(delta)
        best = -INFINITY;
        index = 0;

        for (int i = 0; i < N; ++i)
        {

            if (delta[i] > best)
            {
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

/// @brief Update best solution if a better one was found.
/// @param xbest current best solution, gets updated if xnew is better
/// @param xnew  new solution vector found
/// @param best  current best objective value of xbest, gets updated if xnew is better
/// @param P0_N  size of the main problem SP
/// @return      1 if solution was updated, 0 if not
int update_best(int *xbest, const int *xnew, double *best, int P0_N)
{

    int success = 0;
    int N = P0_N - 1; // N = BabPbSize

    double heur_val = evaluateSolution(xnew);

    if (*best < heur_val)
    {
        memcpy(xbest, xnew, sizeof(int) * N);
        *best = heur_val;
        success = 1;
    }

    return success;
}