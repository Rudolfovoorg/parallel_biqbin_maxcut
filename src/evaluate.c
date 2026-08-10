#include "biqbin.h"
#include "wrapper_hooks.h"

extern BiqBinParameters params;
extern int BabPbSize;
extern double *X;

/**
 * Evaluate a B&B node and return the full upper bound in the original
 * objective scale.
 *
 * SDPbound(...) returns only the reduced-subproblem relaxation value f.
 * Therefore this function adds the fixed-variable contribution:
 *
 *     full_UB(node) = SDPbound(node, SP, PP) + getFixedValue(node, SP).

 * Side effects:
 *   - createSubproblem(...) rebuilds PP from SP and node;
 *   - SDPbound(...) may update the global lower bound through the heuristic;
 *   - update_fractional_solution(...) extracts branching scores from X.
 *
 * @param node Current B&B node
 * @param SP   Original problem
 * @param PP   Subproblem for the current B&B node
 */
double Evaluate(BabNode *node, const Problem *SP, Problem *PP)
{
    // create subproblem PP
    createSubproblem(node, SP, PP);

    // compute the SDP relaxation value and run heuristic
    double bound;
#ifdef PURE_C
    bound = SDPbound(node, SP, PP);
#else
    bound = wrapped_sdp_bound(node, SP, PP);
#endif

    // Fractional solution is updated after B&B node is evaluated
    // This is needed for branching, fracsol is set to the last column of X
    // in the {0, 1} .. X is in {-1, 1}
    update_fractional_solution(node, PP, X);

    // Fixed value (objective contribution of the fixed part) is added after
    // node evaluation as well, as this will simplify the custom sdp bound
    // implementations significantly
    return bound + getFixedValue(node, SP);
}

/**
 * @brief Construct a subproblem (PP) from the original problem (SP) at a given B&B node.
 *
 * Removes rows and columns of variables fixed at the current node (assumed to occupy
 * the upper-left block of SP->L), producing a reduced objective matrix PP->L.
 * The resulting subproblem is formulated for variables in {-1, 1}:
 *
 *      max x' L x,  s.t. x ∈ {-1,1}^(PP->n)
 *
 * @param node Current branch-and-bound node defining fixed variables
 * @param SP   Original problem
 * @param PP   Output subproblem with reduced dimension and updated matrix
 */
void createSubproblem(const BabNode *node, const Problem *SP, Problem *PP)
{
    // Subproblem size is the number of non-fixed variables in the node
    PP->n = BabPbSize + 1 - countFixedVariables(node);

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

    for (int i = 0; i < BabPbSize; ++i)
    {
        for (int j = 0; j < BabPbSize; ++j)
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

/*
 * Return the fixed value of the node.
 * The fixed value is contribution of the fixed variables to
 * the objective value.
 */
double getFixedValue(const BabNode *node, const Problem *SP)
{

    int N = SP->n;
    double fixedvalue = 0.0;

    for (int i = 0; i < BabPbSize; ++i)
    {
        for (int j = 0; j < BabPbSize; ++j)
        {
            if (node->xfixed[i] && node->xfixed[j])
            {
                fixedvalue += SP->L[j + i * N] * node->sol.X[i] * node->sol.X[j];
            }
        }
    }

    return fixedvalue;
}

/// @brief Stores fractional solution in node->fracsol, used for branching
/// @param node current B&B node
/// @param PP subproblem for the current node
/// @param X primal SDP solution
void update_fractional_solution(BabNode *node, Problem *PP, double *X)
{
    // BZ: We updated node->fracsol in SDPBound multiple times, in the end it was
    // was always set to the last value of X, so it makes more sense to update it
    // after nodes evaluation.

    // Store the fractional solution in the node for branching
    int index = 0;
    for (int i = 0; i < BabPbSize; ++i)
    {
        if (node->xfixed[i])
        {
            node->fracsol[i] = (double)node->sol.X[i]; // Fixed are the same as sol.X
        }
        else
        {
            // convert x (last column of primal solution X) from {-1,1} to {0,1}
            node->fracsol[i] = 0.5 * (X[(PP->n - 1) + index * PP->n] + 1.0);
            ++index;
        }
    }
}