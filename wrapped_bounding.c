#include "biqbin.h"

extern BiqBinParameters params;
extern int heptafail;
/****** SDP bound variables needed for reuse *****/
/****** Considering using a struct for them *****/
int count; // number of iterations (adding and purging of cutting planes)

int inc;
int inc_e;
double e; // for vector of all ones
int mk;   // (PP->NIneq + PP->NPentIneq + PP->NHeptaIneq) * PP->bundle

// triangle inequalities
int Tri_NumAdded;
int Tri_NumSubtracted;

// pentagonal inequalities
int Pent_NumAdded;
int Pent_NumSubtracted;

// heptagonal inequalities
int Hepta_NumAdded;
int Hepta_NumSubtracted;

/// helper variables
double fixedvalue; // fixed value contributes to the objective value
double oldf;       // SDP loop uses oldf
int bdl_iter;
double t;

// upper bound solutions
double basic_bound;
double bound;

// Check if done
int giveup;
int prune;
int done;

// This one can be a static function, definition is at the end of the file
static double getFixedValue(const BabNode *node, const Problem *SP);

/// @brief Initialize SDP bound. Builds the temp solution x to be used in heuristics
/// @param node current node
/// @param x stores best solution for the current subproblem globals->PP and node throughout evaluation
/// @param globals_in global variables
void init_sdp(BabNode *node, int *x, GlobalVariables *globals_in)
{
    int problem_size = globals_in->SP->n - 1;

    count = 0; // number of iterations (adding and purging of cutting planes)

    inc = 1;
    inc_e = 0;
    e = 1.0; // for vector of all ones

    bdl_iter = params.init_bundle_iter;

    fixedvalue = getFixedValue(node, globals_in->SP);
    /*** start with no cuts ***/
    // triangle inequalities
    globals_in->PP->NIneq = 0;
    Tri_NumAdded = 0;
    Tri_NumSubtracted = 0;

    // pentagonal inequalities
    globals_in->PP->NPentIneq = 0;
    Pent_NumAdded = 0;
    Pent_NumSubtracted = 0;

    // heptagonal inequalities
    globals_in->PP->NHeptaIneq = 0;
    Hepta_NumAdded = 0;
    Hepta_NumSubtracted = 0;

    /* solve basic SDP relaxation with interior-point method */
    ipm_mc_pk(globals_in->PP->L, globals_in->PP->n, globals_in->X, &globals_in->f, 0);

    basic_bound = globals_in->f + fixedvalue;
    int index = 0;
    for (int i = 0; i < problem_size; ++i)
    {
        if (node->xfixed[i])
        {
            node->fracsol[i] = (double)node->sol.X[i];
        }
        else
        {
            // convert x (last column X) from {-1,1} to {0,1}
            node->fracsol[i] = 0.5 * (globals_in->X[(globals_in->PP->n - 1) + index * globals_in->PP->n] + 1.0);
            ++index;
        }
    }

    /* run heuristic */
    for (int i = 0; i < problem_size; ++i)
    {
        if (node->xfixed[i])
        {
            x[i] = node->sol.X[i];
        }
        else
        {
            x[i] = 0;
        }
    }
}

/// @brief Preproces before the main sdp loop starts
/// @param globals_in pointer to global variables
/// @param is_root root node is treated differently
/// @return 1 if over, 0 if continue with the main sdp loop
int init_main_sdp_loop(GlobalVariables *globals_in, int is_root)
{ // upper bound
    // fixed value contributes to the objective value
    bound = globals_in->f + fixedvalue;
    // Check if done
    giveup = 0;
    prune = 0;
    // check pruning condition
    if (bound < get_lower_bound() + 1.0)
    {
        prune = 1;
        done = 1;
        return done;
    }

    // check if cutting planes need to be added
    if (params.use_diff && (is_root == 0) && (bound > get_lower_bound() + globals_in->diff + 1.0))
    {
        giveup = 1;
        done = 1;
        return done;
    }

    /* separate first triangle inequality */
    double viol3 = updateTriangleInequalities(globals_in->PP, globals_in->dual_gamma, &Tri_NumAdded, &Tri_NumSubtracted, globals_in);

    /***************
     * Bundle init *
     ***************/

    // set dual_gamma = 0
    for (int i = 0; i < globals_in->PP->NIneq; ++i)
    {
        globals_in->dual_gamma[i] = globals_in->Cuts[i].y;
    }

    // t = 0.5 * (f - fh) / (PP->NIneq * viol3^2)
    t = 0.5 * (bound - get_lower_bound()) / (globals_in->PP->NIneq * viol3 * viol3);

    // first evaluation at dual_gamma: f = fct_eval(PP, dual_gamma, X, g)
    // since dual_gamma = 0, this is just basic SDP relaxation
    // --> only need to compute subgradient
    dcopy_(&globals_in->PP->NIneq, &e, &inc_e, globals_in->g, &inc);
    op_B(globals_in->PP, globals_in->g, globals_in->X, globals_in);

    /* setup for bundle */
    // F[0] = <L,X>
    globals_in->F[0] = 0.0;
    for (int i = 0; i < globals_in->PP->n; ++i)
    {
        for (int j = i; j < globals_in->PP->n; ++j)
        {
            if (i == j)
            {
                globals_in->F[0] += globals_in->PP->L[i + i * globals_in->PP->n] * globals_in->X[i + i * globals_in->PP->n];
            }
            else
            {
                globals_in->F[0] += 2 * globals_in->PP->L[j + i * globals_in->PP->n] * globals_in->X[j + i * globals_in->PP->n];
            }
        }
    }

    // G = g
    dcopy_(&globals_in->PP->NIneq, globals_in->g, &inc, globals_in->G, &inc);

    // include X in X_bundle
    int nn = globals_in->PP->n * globals_in->PP->n;
    dcopy_(&nn, globals_in->X, &inc, globals_in->X_bundle, &inc);

    // initialize the bundle counter
    globals_in->PP->bundle = 1;

    done = 0;
    return done;
}

/// @brief Use bundle_method, calculate upper bound
/// @param globals_in
/// @return 1 if prune node, 0 if not
int main_sdp_loop_start(GlobalVariables *globals_in)
{
    // Update iteration counter
    ++count;
    oldf = globals_in->f;
    // Call bundle method
    bundle_method(globals_in->PP, &t, bdl_iter, globals_in);

    // upper bound
    bound = globals_in->f + fixedvalue;
    // prune test
    prune = (bound < get_lower_bound() + 1.0) ? 1 : 0;

    return prune;
}

/// @brief Ran after heuristics in SDP_bound main loop
/// @param node reads node->xfixed; sets node->fracsol
/// @param globals_in
/// @return 1 if calculating upper bound is done
int main_sdp_loop_end(BabNode *node, GlobalVariables *globals_in)
{
    int problem_size = globals_in->SP->n - 1;
    int prune = (bound < get_lower_bound() + 1.0) ? 1 : 0;
    // compute
    double gap = bound - get_lower_bound();
    // printf("Gap: %f\n", gap);

    /* check if we will not be able to prune the node */
    if (count == params.triag_iter + params.pent_iter + params.hept_iter)
    {
        if ((gap - 1.0 > (oldf - globals_in->f) * (params.max_outer_iter - count)))
            giveup = 1;
    }

    /* check if extra iterations can close the gap */
    if (count == params.max_outer_iter)
    {
        if (gap - 1.0 > (oldf - globals_in->f) * params.extra_iter)
            giveup = 1;
    }

    /* max number of iterations reached */
    if (count == params.max_outer_iter + params.extra_iter)
        giveup = 1;

    // purge inactive cutting planes, add new inequalities
    if (!prune && !giveup)
    {

        int triag = globals_in->PP->NIneq;     // save number of triangle and pentagonal inequalities before purging
        int penta = globals_in->PP->NPentIneq; // --> to know with which index in dual vector dual_gamma, pentagonal
                                               // and heptagonal inequalities start!

        double viol3 = updateTriangleInequalities(globals_in->PP, globals_in->dual_gamma, &Tri_NumAdded, &Tri_NumSubtracted, globals_in);

        double viol5 = 0.0;
        double viol7 = 0.0;

        /* include pentagonal and heptagonal inequalities */
        if (params.include_Pent && (count > params.triag_iter || viol3 < 0.2))
        {
            viol5 = updatePentagonalInequalities(globals_in->PP, globals_in->dual_gamma, &Pent_NumAdded, &Pent_NumSubtracted, triag, globals_in);
        }
        if (params.include_Hepta && ((count > params.triag_iter + params.pent_iter) || (viol3 < 0.2 && (1 - viol5 < 0.4))))
        {
            if (!heptafail) {
                viol7 = updateHeptagonalInequalities(globals_in->PP, globals_in->dual_gamma, &Hepta_NumAdded, &Hepta_NumSubtracted, triag + penta, globals_in);
            }
        }
    }
    else
    {
        Tri_NumAdded = 0;
        Tri_NumSubtracted = 0;
        Pent_NumAdded = 0;
        Pent_NumSubtracted = 0;
        Hepta_NumAdded = 0;
        Hepta_NumSubtracted = 0;
        heptafail = 0;
    }

    // Test stopping conditions
    done =
        prune || // can prune the B&B tree
        giveup;  // upper bound to far away from lower bound

    // Store the fractional solution in the node
    int index = 0;
    for (int i = 0; i < problem_size; ++i)
    {
        if (node->xfixed[i])
        {
            node->fracsol[i] = (double)node->sol.X[i];
        }
        else
        {
            // convert x (last column X) from {-1,1} to {0,1}
            node->fracsol[i] = 0.5 * (globals_in->X[(globals_in->PP->n - 1) + index * globals_in->PP->n] + 1.0);
            ++index;
        }
    }
    // END NOTE

    /*** bundle update: due to separation of new cutting planes ***/
    if (!done)
    {

        // adjust size of dual_gamma
        for (int i = 0; i < globals_in->PP->NIneq; ++i)
            globals_in->dual_gamma[i] = globals_in->Cuts[i].y;

        for (int i = 0; i < globals_in->PP->NPentIneq; ++i)
            globals_in->dual_gamma[i + globals_in->PP->NIneq] = globals_in->Pent_Cuts[i].y;

        for (int i = 0; i < globals_in->PP->NHeptaIneq; ++i)
            globals_in->dual_gamma[i + globals_in->PP->NIneq + globals_in->PP->NPentIneq] = globals_in->Hepta_Cuts[i].y;

        fct_eval(globals_in->PP, globals_in->dual_gamma, globals_in->X_test, globals_in->g, globals_in);

        // G
        /* for i = 1:k
         *      G(:,i) = b - A*X(:,i);
         * end
         */
        mk = (globals_in->PP->NIneq + globals_in->PP->NPentIneq + globals_in->PP->NHeptaIneq) * globals_in->PP->bundle;
        int nn = globals_in->PP->n * globals_in->PP->n;
        dcopy_(&mk, &e, &inc_e, globals_in->G, &inc); // fill G with 1
        for (int i = 0; i < globals_in->PP->bundle; ++i)
        {
            op_B(globals_in->PP, globals_in->G + i * (globals_in->PP->NIneq + globals_in->PP->NPentIneq + globals_in->PP->NHeptaIneq), globals_in->X_bundle + i * nn, globals_in);
        }

        // add g to G
        int ineq = globals_in->PP->NIneq + globals_in->PP->NPentIneq + globals_in->PP->NHeptaIneq;
        dcopy_(&ineq, globals_in->g, &inc, globals_in->G + globals_in->PP->bundle * (globals_in->PP->NIneq + globals_in->PP->NPentIneq + globals_in->PP->NHeptaIneq), &inc);

        // add <L, X> to F
        globals_in->F[globals_in->PP->bundle] = 0.0;
        for (int i = 0; i < globals_in->PP->n; ++i)
        {
            for (int j = i; j < globals_in->PP->n; ++j)
            {
                if (i == j)
                {
                    globals_in->F[globals_in->PP->bundle] += globals_in->PP->L[i + i * globals_in->PP->n] * globals_in->X_test[i + i * globals_in->PP->n];
                }
                else
                {
                    globals_in->F[globals_in->PP->bundle] += 2 * globals_in->PP->L[j + i * globals_in->PP->n] * globals_in->X_test[j + i * globals_in->PP->n];
                }
            }
        }

        // add X to X_bundle
        dcopy_(&nn, globals_in->X_test, &inc, globals_in->X_bundle + globals_in->PP->bundle * nn, &inc);

        // Check bundle size for overflow (can not append more)
        if (globals_in->PP->bundle == MaxBundle)
        {
            fprintf(stderr, "\nError: Bundle size too large! Adjust MaxBundle in biqbin.h.\n");
            MPI_Abort(MPI_COMM_WORLD, 10);
        }

        // increase bundle
        ++(globals_in->PP->bundle);

        // new estimate for t
        t *= 1.05;
    }

    /* increase number of bundle iterations */
    bdl_iter += count % 2;
    bdl_iter = (bdl_iter < params.max_bundle_iter) ? bdl_iter : params.max_bundle_iter;
    return done;
} // end while loop

/// @brief Sets final upper bound of node to globals->f + fixedvalue;
/// @param node current node
/// @param globals_in uses only globals->f
/// @return upper bound
double get_upper_bound(BabNode *node, const GlobalVariables *globals_in)
{
    bound = globals_in->f + fixedvalue;
    node->upper_bound = bound;
    return bound;
}
/// @brief Root node needs to set the globals->diff if params.diff == 1
void set_globals_diff(GlobalVariables *globals_in)
{
    globals_in->diff = basic_bound - bound;
}

/// @brief Updates lower bound and solution nodes based on the temp solution x
/// @param node saves the x solution in the node->sol.X if it is better
/// @param x solution which is being evaluated
/// @param SP main problem SP
void update_solution_wrapped(BabNode *node, const int *x, const Problem *SP)
{
    if (update_solution(x, SP)) // updates lower bound if the current x solution is best
    {
        int problem_size = SP->n - 1;
        for (int i = 0; i < problem_size; ++i)
        {
            node->sol.X[i] = x[i];
        }
    }
}

/**********************************************************************/
/*******************  Copied from evaluate.c  ************************/
/********************************************************************/
/// @brief The fixed value is contribution of the fixed variables to the objective value.
/// @param node
/// @param SP  is the original problem
/// @return the fixed value of tohe node.
static double getFixedValue(const BabNode *node, const Problem *SP)
{

    int N = SP->n;
    int problem_size = SP->n - 1;
    double fixedvalue = 0.0;

    for (int i = 0; i < problem_size; ++i)
    {
        for (int j = 0; j < problem_size; ++j)
        {
            if (node->xfixed[i] && node->xfixed[j])
            {
                fixedvalue += SP->L[j + i * N] * node->sol.X[i] * node->sol.X[j];
            }
        }
    }

    return fixedvalue;
}

/// @brief Writes subproblem to PP. Computes the subproblem removing the rows and the columns of the fixed variables upper left corner of SP->L
/// @param node is the current node
/// @param SP is the original problem
/// @param PP is the subproblem (some variables are fixed)
/// @note Function prepares objective matrix L for model in -1,1 variables: max x'LX, s.t. x in {-1,1}^(PP->n)
void create_subproblem(BabNode *node, Problem *SP, Problem *PP)
{

    // Subproblem size is the number of non-fixed variables in the node
    PP->n = SP->n - count_fixed_variable(node);

    /* build objective:
     * Laplacian;
     * z'*L*z = sum_{i != fixed, j != fixed} L_ij*xi*xj (smaller matrix L_bar for subproblem)
              + sum_{i = fixed, j = fixed} L_ij*x_i*xj (getFixedValue)
              + sum_{rows of fixed vertices without fixed entries}  (linear part, that is twice added to diagonal of L_bar)
     */

    /* Laplacian is created by deleting appropriate rows and cols
     * of upper left corner of SP->L
     */
    int index = 0;
    int N = SP->n;
    double row_sum = 0.0;

    // rows which are deleted due to fixed variable
    // later add to diagonal
    double fixedRow[PP->n - 1];
    for (int i = 0; i < PP->n - 1; ++i)
        fixedRow[i] = 0.0;

    // counter for fixedRow
    int fixed = 0;

    // last element (lower right corner) is sum
    double sum = 0.0;

    int problem_size = SP->n - 1;
    for (int i = 0; i < problem_size; ++i)
    {
        for (int j = 0; j < problem_size; ++j)
        {
            if (!node->xfixed[i] && !node->xfixed[j])
            { // delete rows and cols of SP->L
                PP->L[index] = SP->L[j + i * N];
                row_sum += PP->L[index];
                ++index;
            }
            else if ((node->xfixed[i] && node->sol.X[i] == 1) && !node->xfixed[j])
            { // save fixed rows to add to diagonal
                fixedRow[fixed] += SP->L[j + i * N];
                ++fixed;
            }
        }

        if (!node->xfixed[i])
        {
            PP->L[index] = row_sum; // vector part of PP->L (last column)
            ++index;
        }

        // row scaned, set to 0
        row_sum = 0.0;
        fixed = 0;
    }

    // add last row (copy from last column)
    for (int i = 0; i < PP->n - 1; ++i)
        PP->L[i + (PP->n - 1) * PP->n] = PP->L[PP->n - 1 + i * PP->n];

    /* LINEAR PART OF PP->L:   add 2x vector fixedRow to diagonal, last col and last row */
    for (int i = 0; i < PP->n - 1; ++i)
    {
        PP->L[i + i * PP->n] += 2 * fixedRow[i];
        PP->L[PP->n - 1 + i * PP->n] += 2 * fixedRow[i];
        PP->L[i + (PP->n - 1) * PP->n] += 2 * fixedRow[i];

        sum += PP->L[i + (PP->n - 1) * PP->n];
    }

    /* CONSTANT PART OF PP->L:  element (PP->n - 1, PP->n - 1) */
    PP->L[PP->n - 1 + (PP->n - 1) * PP->n] = sum;

    /* multiple by 1/4 the whole matrix L */
    double alpha = 0.25;
    int inc = 1;
    int nn = (PP->n) * (PP->n);
    dscal_(&nn, &alpha, PP->L, &inc);
}
