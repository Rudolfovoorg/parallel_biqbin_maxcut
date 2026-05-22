#include "wrapper_hooks.h"
#include "wrapper_utils.h"

py::object python_heuristic_override;
py::object python_node_evaluation_override;


/// @brief set heuristic function to a Python function
void set_heuristic_override(py::object func) { python_heuristic_override = func; }

/// @brief set the SDPBound function (computes both bounds) to a Python function
void set_node_evaluation_override(py::object func) { python_node_evaluation_override = func; }

/// @brief Called in runHeuristic in heuristic.c
/// @param P0 is the original Problem *SP in global_var.h
/// @param P  current subproblem Problem *PP in global_var.h
/// @param node current branch and bound node
/// @return best lower bound of the current subproblem found by the heuristic used
double wrapped_heuristic(const Problem *P0, const Problem *P, const BabNode *node)
{
    // Call Python override
    return python_heuristic_override(
                                py::cast(node, py::return_value_policy::reference),
                                py::cast(P0, py::return_value_policy::reference),
                                py::cast(P, py::return_value_policy::reference))
                                .cast<double>();
}

/// @brief SDPBound in bounding.c originally, called in Evaluate in evaluate.c it internally calls wrapped_heuristic many times
/// @param node current branch and bound node
/// @param P0 is the original Problem *SP in global_var.h
/// @param P  current subproblem Problem *PP in global_var.h
/// @param rank MPI rank of the process
/// @return best upper bound of the current subproblem found by the heuristic used
double wrapped_sdp_bound(BabNode *node, const Problem *P0, Problem *P)
{
    return python_node_evaluation_override(
               py::cast(node, py::return_value_policy::reference),
               py::cast(P0, py::return_value_policy::reference),
               py::cast(P, py::return_value_policy::reference))
        .cast<double>();
}
