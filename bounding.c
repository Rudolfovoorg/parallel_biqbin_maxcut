#include "biqbin.h"

extern BiqBinParameters params;
extern FILE *output;

// extern GlobalVariables *globals;

// extern double TIME;                 
// extern Triangle_Inequality *Cuts;            // vector of triangle inequality constraints
// extern Pentagonal_Inequality *Pent_Cuts;     // vector of pentagonal inequality constraints
// extern Heptagonal_Inequality *Hepta_Cuts;    // vector of heptagonal inequality constraints

// extern double f;                             // function value of relaxation
// extern double *X;                            // current X
// extern double *X_bundle;                     // current X
// extern double *F;                            // bundle of function values
// extern double *G;                            // bundle of subgradients
// extern double *g;                            // subgradient
// extern double *dual_gamma;                   // dual multiplers for triangle inequalities
// extern double *X_test;

// extern double diff;		                     // difference between basic SDP relaxation and bound with added cutting planes

/******** main bounding routine calling bundle method ********/
double SDPbound(BabNode *node, Problem *PP, int rank, GlobalVariables *globals_in) {
    int problem_size = globals_in->SP->n - 1;
    int index;                      // helps to store the fractional solution in the node
    double bound;                   // f + fixedvalue
    double gap;                     // difference between best lower bound and upper bound
    double oldf;                    // stores f from previous iteration 
    int x[problem_size];            // vector for heuristic
    double viol3;                   // maximum violation of triangle inequalities
    double viol5 = 0.0;             // maximum violation of pentagonal inequalities
    double viol7 = 0.0;             // maximum violation of heptagonal inequalities
    int count = 0;                  // number of iterations (adding and purging of cutting planes)

    int triag;                      // starting index for pentagonal inequalities in vector dual_gamma
    int penta;                      // starting index for heptagonal inequalities in vector dual_gamma

    int inc = 1;
    int inc_e = 0;
    double e = 1.0;                 // for vector of all ones
    int nn = PP->n * PP->n;
    int mk;                         // (PP->NIneq + PP->NPentIneq + PP->NHeptaIneq) * PP->bundle
    
    /* stopping conditions */
    // int done = 0;                   
    // int giveup = 0;                                   
    // int prune = 0;

    // number of initial iterations of bundle method
    int bdl_iter = params.init_bundle_iter;

    // fixed value contributes to the objective value
    double fixedvalue = getFixedValue(node, globals_in->SP);

    /*** start with no cuts ***/
    // triangle inequalities
    PP->NIneq = 0; 
    int Tri_NumAdded = 0;
    int Tri_NumSubtracted = 0;

    // pentagonal inequalities
    PP->NPentIneq = 0;
    int Pent_NumAdded = 0;
    int Pent_NumSubtracted = 0;

    // heptagonal inequalities
    PP->NHeptaIneq = 0;
    int Hepta_NumAdded = 0;
    int Hepta_NumSubtracted = 0;                         

    /* solve basic SDP relaxation with interior-point method */
    ipm_mc_pk(PP->L, PP->n, globals_in->X, &globals_in->f, 0);

    // store basic SDP bound to compute diff in the root node
    double basic_bound = globals_in->f + fixedvalue;

    // Store the fractional solution in the node


    index = 0;
    for (int i = 0; i < problem_size; ++i) {
        if (node->xfixed[i]) {
            node->fracsol[i] = (double) node->sol.X[i];
        }
        else {
            // convert x (last column X) from {-1,1} to {0,1}
            node->fracsol[i] = 0.5*(globals_in->X[(PP->n - 1) + index*PP->n] + 1.0); 
            ++index;
        }
    }
    /* run heuristic */
    for (int i = 0; i < problem_size; ++i) {
        if (node->xfixed[i]) {
            x[i] = node->sol.X[i];
        }
        else {
            x[i] = 0;
        }
    }
    // Updates x with the current best solution
    runHeuristic(globals_in->SP, PP, node, x, globals_in->X, globals_in->Z);
    //
    if (updateSolution(x, globals_in->SP)) // updates lower bound if the current x solution is best
    {
        // Beno's note - I need to store the solution into current nodes BabSolution struct to read it in python
        for (int i = 0; i < problem_size; ++i) {
            node->sol.X[i] = x[i];
        }
    }
    
    // upper bound
    bound = globals_in->f + fixedvalue;

    // Check if done
    int giveup = 0;
    int prune = 0;
    // check pruning condition
    if ( bound < get_lower_bound() + 1.0 ) {
        prune = 1;
        goto END;
    }


    // check if cutting planes need to be added
    if (params.use_diff && (rank != 0) && (bound > get_lower_bound() + globals_in->diff + 1.0)) {
        giveup = 1;
        goto END;
    }

    /* separate first triangle inequality */
    viol3 = updateTriangleInequalities(PP, globals_in->dual_gamma, &Tri_NumAdded, &Tri_NumSubtracted, globals_in);

    /***************
     * Bundle init *
     ***************/

    // set dual_gamma = 0
    for (int i = 0; i < PP->NIneq; ++i) {
        globals_in->dual_gamma[i] = globals_in->Cuts[i].y;
    }

    // t = 0.5 * (f - fh) / (PP->NIneq * viol3^2)
    double t = 0.5 * (bound - get_lower_bound()) / (PP->NIneq * viol3 * viol3);

    // first evaluation at dual_gamma: f = fct_eval(PP, dual_gamma, X, g)
    // since dual_gamma = 0, this is just basic SDP relaxation
    // --> only need to compute subgradient
    dcopy_(&PP->NIneq, &e, &inc_e, globals_in->g, &inc);
    op_B(PP, globals_in->g, globals_in->X,  globals_in);

    /* setup for bundle */
    // F[0] = <L,X>
    globals_in->F[0] = 0.0;
    for (int i = 0; i < PP->n; ++i) {
        for (int j = i; j < PP->n; ++j) {
            if (i == j) {
                globals_in->F[0] += PP->L[i + i*PP->n] * globals_in->X[i + i*PP->n];
            }
            else {
                globals_in->F[0] += 2 * PP->L[j + i*PP->n] * globals_in->X[j + i*PP->n];
            }
        }
    }

    // G = g
    dcopy_(&PP->NIneq, globals_in->g, &inc, globals_in->G, &inc);

    // include X in X_bundle
    dcopy_(&nn, globals_in->X, &inc, globals_in->X_bundle, &inc);

    // initialize the bundle counter
    PP->bundle = 1;

    int done = 0;

    
    /*** Main loop ***/
    while (!done) {

        // Update iteration counter
        ++count;
        oldf = globals_in->f;

        // Call bundle method
        bundle_method(PP, &t, bdl_iter, globals_in);  

        // upper bound
        bound = globals_in->f + fixedvalue;
        // prune test
        prune = ( bound < get_lower_bound() + 1.0 ) ? 1 : 0;

        /******** heuristic ********/
        if (!prune) {

            for (int i = 0; i < problem_size; ++i) {
                if (node->xfixed[i]) {
                    x[i] = node->sol.X[i];
                }
                else {
                    x[i] = 0;
                }
            }

            runHeuristic(globals_in->SP, PP, node, x, globals_in->X, globals_in->Z);
            
            if (updateSolution(x, globals_in->SP)) // updates lower bound
            {
                // Beno's note - I need to store the solution into current nodes BabSolution struct to read it in python
                for (int i = 0; i < problem_size; ++i) {
                    node->sol.X[i] = x[i];
                }
            }
            prune = ( bound < get_lower_bound() + 1.0 ) ? 1 : 0;
        }
        /***************************/

        // compute 
        gap = bound - get_lower_bound();
        // printf("Gap: %f\n", gap);

        /* check if we will not be able to prune the node */
        if (count == params.triag_iter + params.pent_iter + params.hept_iter) {
            if ( (gap - 1.0 > (oldf - globals_in->f)*(params.max_outer_iter - count)))
                giveup = 1;
        }

        /* check if extra iterations can close the gap */
        if (count == params.max_outer_iter) {
            if ( gap - 1.0 > (oldf - globals_in->f)*params.extra_iter )
                giveup = 1;
        }
        
        /* max number of iterations reached */
        if (count == params.max_outer_iter + params.extra_iter)
            giveup = 1; 


        // purge inactive cutting planes, add new inequalities
        if (!prune && !giveup) {
            
            triag = PP->NIneq;          // save number of triangle and pentagonal inequalities before purging
            penta = PP->NPentIneq;      // --> to know with which index in dual vector dual_gamma, pentagonal
                                        // and heptagonal inequalities start!

            viol3 = updateTriangleInequalities(PP, globals_in->dual_gamma, &Tri_NumAdded, &Tri_NumSubtracted, globals_in);
                      
            /* include pentagonal and heptagonal inequalities */          
            if ( params.include_Pent && (count > params.triag_iter || viol3 < 0.2) )
                viol5 = updatePentagonalInequalities(PP, globals_in->dual_gamma, &Pent_NumAdded, &Pent_NumSubtracted, triag, globals_in);  

            if ( params.include_Hepta && ( (count > params.triag_iter + params.pent_iter) || (viol3 < 0.2 && (1 - viol5 < 0.4)) ) )
                viol7 = updateHeptagonalInequalities(PP, globals_in->dual_gamma, &Hepta_NumAdded, &Hepta_NumSubtracted, triag + penta, globals_in);      
        }
        else {               
            Tri_NumAdded = 0;
            Tri_NumSubtracted = 0;
            Pent_NumAdded = 0;
            Pent_NumSubtracted = 0;
            Hepta_NumAdded = 0;
            Hepta_NumSubtracted = 0;
        }

        // Test stopping conditions
        done = 
            prune ||                       // can prune the B&B tree 
            giveup;                        // upper bound to far away from lower bound

        // Store the fractional solution in the node    
        index = 0;
        for (int i = 0; i < problem_size; ++i) {
            if (node->xfixed[i]) {
                node->fracsol[i] = (double) node->sol.X[i];
            }
            else {
                // convert x (last column X) from {-1,1} to {0,1}
                node->fracsol[i] = 0.5*(globals_in->X[(PP->n - 1) + index*PP->n] + 1.0); 
                ++index;
            }
        }

        /*** bundle update: due to separation of new cutting planes ***/
        if (!done) {
            // adjust size of dual_gamma
            for (int i = 0; i < PP->NIneq; ++i)
                globals_in->dual_gamma[i] = globals_in->Cuts[i].y;
            
            for (int i = 0; i < PP->NPentIneq; ++i)
                globals_in->dual_gamma[i + PP->NIneq] = globals_in->Pent_Cuts[i].y;

            for (int i = 0; i < PP->NHeptaIneq; ++i)
                globals_in->dual_gamma[i + PP->NIneq + PP->NPentIneq] = globals_in->Hepta_Cuts[i].y;


            fct_eval(PP, globals_in->dual_gamma, globals_in->X_test, globals_in->g, globals_in);

            // G
            /* for i = 1:k
             *      G(:,i) = b - A*X(:,i);
             * end
             */ 
            mk = (PP->NIneq + PP->NPentIneq + PP->NHeptaIneq) * PP->bundle;
            dcopy_(&mk, &e, &inc_e, globals_in->G, &inc); // fill G with 1
            for (int i = 0; i < PP->bundle; ++i) {
                op_B(PP, globals_in->G + i*(PP->NIneq + PP->NPentIneq + PP->NHeptaIneq), globals_in->X_bundle + i * nn, globals_in);
            }

            // add g to G
            int ineq = PP->NIneq + PP->NPentIneq + PP->NHeptaIneq;
            dcopy_(&ineq, globals_in->g, &inc, globals_in->G + PP->bundle * (PP->NIneq + PP->NPentIneq + PP->NHeptaIneq), &inc);

            // add <L, X> to F
            globals_in->F[PP->bundle] = 0.0;
            for (int i = 0; i < PP->n; ++i) {
                for (int j = i; j < PP->n; ++j) {
                    if (i == j) {
                        globals_in->F[PP->bundle] += PP->L[i + i*PP->n] * globals_in->X_test[i + i*PP->n];
                    }
                    else {
                        globals_in->F[PP->bundle] += 2 * PP->L[j + i*PP->n] * globals_in->X_test[j + i*PP->n];
                    }
                }
            }

            // add X to X_bundle
            dcopy_(&nn, globals_in->X_test, &inc, globals_in->X_bundle + PP->bundle * nn, &inc);

            // Check bundle size for overflow (can not append more)
            if (PP->bundle == MaxBundle) {
                fprintf(stderr, "\nError: Bundle size too large! Adjust MaxBundle in biqbin.h.\n");
                MPI_Abort(MPI_COMM_WORLD,10);
            }

            // increase bundle
            ++(PP->bundle);

            // new estimate for t
            t *= 1.05;

        }

        /* increase number of bundle iterations */
        bdl_iter += count % 2;
        bdl_iter = (bdl_iter  < params.max_bundle_iter) ? bdl_iter  : params.max_bundle_iter;

 
    } // end while loop

    bound = globals_in->f + fixedvalue;

    // compute difference between basic SDP relaxation and bound with added cutting planes
    if (rank == 0)
    {
        globals_in->diff = basic_bound - bound;
    }
    END:   

    return bound;
}

