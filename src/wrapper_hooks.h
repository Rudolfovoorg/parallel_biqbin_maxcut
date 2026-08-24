#ifndef WRAPPER_HOOKS_H
#define WRAPPER_HOOKS_H

#ifndef PURE_C

#include "biqbin_cpp_api.h"

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

EXTERN_C double wrapped_heuristic(const Problem *P0, const Problem *P, const BabNode *node);
EXTERN_C double wrapped_sdp_bound(BabNode *node, const Problem *P0, Problem *P);

/* C++-only: Pybind11 overrides and RAII guard */
#ifdef __cplusplus
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

extern py::object python_heuristic_override;
extern py::object python_node_evaluation_override;

void set_heuristic_override(py::object func);
void set_node_evaluation_override(py::object func);

struct PythonReferencesGuard
{
    ~PythonReferencesGuard()
    {
        python_heuristic_override = py::object();
        python_node_evaluation_override = py::object();
    }
};

#endif /* __cplusplus */
#endif /* PURE_C */
#endif /* WRAPPER_HOOKS_H */