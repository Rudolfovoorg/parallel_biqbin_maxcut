#include "biqbin.h"

extern BiqBinParameters params;

/****** SDP bound variables needed for reuse *****/
double viol3; // maximum violation of triangle inequalities
double viol5; // maximum violation of pentagonal inequalities
double viol7; // maximum violation of heptagonal inequalities
int count;    // number of iterations (adding and purging of cutting planes)

int triag; // starting index for pentagonal inequalities in vector dual_gamma
int penta; // starting index for heptagonal inequalities in vector dual_gamma

int inc;
int inc_e;
double e; // for vector of all ones
int nn;
int mk; // (PP->NIneq + PP->NPentIneq + PP->NHeptaIneq) * PP->bundle

// triangle inequalities
int Tri_NumAdded;
int Tri_NumSubtracted;

// pentagonal inequalities
int Pent_NumAdded;
int Pent_NumSubtracted;

// heptagonal inequalities
int Hepta_NumAdded;
int Hepta_NumSubtracted;

// upper bound
// fixed value contributes to the objective value
double fixedvalue;
double oldf;
int bdl_iter;
double t;

double basic_bound;
double bound;

// helpers
double gap;

// Check if done
int giveup;
int prune;
int done;

double* get_subproblem_laplacean(GlobalVariables *globals_in) {
    return globals_in->PP->L;
}

// builds the temp solution x to be used in heuristics
void init_sdp(BabNode *node, int *x, GlobalVariables *globals_in)
{
    int problem_size = globals_in->SP->n - 1;

    viol5 = 0.0; // maximum violation of pentagonal inequalities
    viol7 = 0.0; // maximum violation of heptagonal inequalities
    count = 0;   // number of iterations (adding and purging of cutting planes)
    
    inc = 1;
    inc_e = 0;
    e = 1.0; // for vector of all ones
    nn = globals_in->PP->n * globals_in->PP->n;

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

double heuristics_wrapped(BabNode *node, int *x, GlobalVariables *globals_in)
{
    // Updates x with the current best solution, returns the lower bound
    return runHeuristic(globals_in->SP, globals_in->PP, node, x, globals_in->X, globals_in->Z);
}

void update_solution_wrapped(BabNode *node, int *x, GlobalVariables *globals_in)
{
    if (updateSolution(x, globals_in->SP)) // updates lower bound if the current x solution is best
    {
        int problem_size = globals_in->SP->n - 1;
        for (int i = 0; i < problem_size; ++i)
        {
            node->sol.X[i] = x[i];
        }
    }
}

int init_main_sdp_loop(GlobalVariables *globals_in, int is_root)
{    // upper bound
    // fixed value contributes to the objective value
    bound = globals_in->f + fixedvalue;
    // Check if done
    giveup = 0;
    prune = 0;
    // check pruning condition
    if (bound < Bab_LBGet() + 1.0)
    {
        prune = 1;
        done = 1;
        return done;
    }

    // check if cutting planes need to be added
    if (params.use_diff && (is_root == 0) && (bound > Bab_LBGet() + globals_in->diff + 1.0))
    {
        giveup = 1;
        done = 1;
        return done;
    }

    /* separate first triangle inequality */
    viol3 = updateTriangleInequalities(globals_in->PP, globals_in->dual_gamma, &Tri_NumAdded, &Tri_NumSubtracted, globals_in);

    /***************
     * Bundle init *
     ***************/

    // set dual_gamma = 0
    for (int i = 0; i < globals_in->PP->NIneq; ++i)
    {
        globals_in->dual_gamma[i] = globals_in->Cuts[i].y;
    }

    // t = 0.5 * (f - fh) / (PP->NIneq * viol3^2)
    t = 0.5 * (bound - Bab_LBGet()) / (globals_in->PP->NIneq * viol3 * viol3);

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
    dcopy_(&nn, globals_in->X, &inc, globals_in->X_bundle, &inc);

    // initialize the bundle counter
    globals_in->PP->bundle = 1;

    done = 0;
    return done;
}

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
    prune = (bound < Bab_LBGet() + 1.0) ? 1 : 0;
    return prune;
}

void update_partial_solution(BabNode *node, int *x, int problem_size) {
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


// run heuristics -> update_solution


int main_sdp_loop_end(BabNode *node, GlobalVariables *globals_in) {
    int problem_size = globals_in->SP->n - 1;
    int prune = (bound < Bab_LBGet() + 1.0) ? 1 : 0;
    // compute
    gap = bound - Bab_LBGet();
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

        triag = globals_in->PP->NIneq;     // save number of triangle and pentagonal inequalities before purging
        penta = globals_in->PP->NPentIneq; // --> to know with which index in dual vector dual_gamma, pentagonal
                               // and heptagonal inequalities start!

        viol3 = updateTriangleInequalities(globals_in->PP, globals_in->dual_gamma, &Tri_NumAdded, &Tri_NumSubtracted, globals_in);

        /* include pentagonal and heptagonal inequalities */
        if (params.include_Pent && (count > params.triag_iter || viol3 < 0.2))
            viol5 = updatePentagonalInequalities(globals_in->PP, globals_in->dual_gamma, &Pent_NumAdded, &Pent_NumSubtracted, triag, globals_in);

        if (params.include_Hepta && ((count > params.triag_iter + params.pent_iter) || (viol3 < 0.2 && (1 - viol5 < 0.4))))
            viol7 = updateHeptagonalInequalities(globals_in->PP, globals_in->dual_gamma, &Hepta_NumAdded, &Hepta_NumSubtracted, triag + penta, globals_in);
    }
    else
    {
        Tri_NumAdded = 0;
        Tri_NumSubtracted = 0;
        Pent_NumAdded = 0;
        Pent_NumSubtracted = 0;
        Hepta_NumAdded = 0;
        Hepta_NumSubtracted = 0;
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

double get_upper_bound(BabNode *node, GlobalVariables *globals_in)
{
    bound = globals_in->f + fixedvalue;
    node->upper_bound = bound;
    return bound;
}
// need to set the globals->diff on the root node
void set_globals_diff(GlobalVariables *globals_in)
{
    globals_in->diff = basic_bound - bound;
}