// NOTE: this file is only included in allocate_free.c !
#include "biqbin.h"
/********************************************************/
/************ List of all global variables **************/
/********************************************************/
BiqBinParameters params;            // BiqBin parameters
FILE *output;                       // output file
Problem *SP;                        // original problem instance
Problem *PP;                        // subproblem instance
int stopped = 0;                    // true if the algorithm stopped at root node or after a time limit
double root_bound;                  // SDP upper bound at root node
double TIME;                        // CPU time
double diff;			            // difference between basic SDP relaxation and bound with added cutting planes  	
/********************************************************/


/********************************************************/
/*************** Specific to node ***********************/
/********************************************************/
/* PRIMAL variables */
double *X;                          // Stores current (psd) X (primal solution). Violated inequalities are computed from X.
double *Z;                          // Cholesky factorization: X = ZZ^T (used for heuristic)
double *X_bundle;                   // containts bundle matrices as columns
double *X_test;                     // matching pair X for gamma_test

/* DUAL variables */
double *dual_gamma;                      // (nonnegative) dual multiplier to cutting planes
double *dgamma;                     // step direction vector
double *gamma_test;
double *lambda;                     // vector containing scalars of convex combinations of bundle matrices X_i
double *eta;                        // dual multiplier to dual_gamma >= 0 constraint
double *F;                          // vector of values <L,X_i>
double *g;                          // subgradient
double *G;                          // matrix of subgradients

double f;                           // objective value of relaxation                      

/* Triangle Inequalities variables */
Triangle_Inequality *Cuts;          // vector (MaxTriIneqAdded) of current triangle inequality constraints
Triangle_Inequality *List;          // vector (params.TriIneq) of new violated triangle inequalities

/* Pentagonal Inequalities variables */
Pentagonal_Inequality *Pent_Cuts;   // vector (MaxPentIneqAdded) of current pentagonal inequality constraints
Pentagonal_Inequality *Pent_List;   // vector (params.PentIneq) of new violated pentagonal inequalities

/* Heptagonal Inequalities variables */
Heptagonal_Inequality *Hepta_Cuts;   // vector (MaxHeptaIneqAdded) of current heptagonal inequality constraints
Heptagonal_Inequality *Hepta_List;   // vector (params.HeptaIneq) of new violated heptagonal inequalities
