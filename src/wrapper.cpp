
#include <iostream>

/* Boost library */
// https://github.com/avast/boost-python-examples/blob/master/boost-python-examples/boost-python-examples.cpp
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/scoped_array.hpp>
#include <boost/python/wrapper.hpp>
#include <boost/python/module.hpp>
#include <boost/python/call.hpp>

#include "biqbin_cpp_api.h"
#include "blas_laplack.h"

#include "wrapper.h"

namespace np = boost::python::numpy;
namespace p = boost::python;

/* biqbin's global variables from global_var.h */
extern Problem *SP;
extern Problem *PP;
extern BabSolution *BabSol;
extern int BabPbSize;

/* final solution */
std::vector<int> selected_nodes;
double running_time;
int rank;

// Global Python override function (if any)
p::object python_heuristic_override;
p::object py_read_data_override;

/// @brief set heuristic function from Python
/// @param func
void set_heuristic_override(p::object func)
{
    python_heuristic_override = func;
}

/// @brief set instance reading function from Python
/// @param func
void set_read_data_override(p::object func)
{
    py_read_data_override = func;
}

void set_rank(int r)
{
    rank = r;
}

int get_rank()
{
    return rank;
}

/// @brief TODO: find a better fix for conflicts with MPI
void clean_python_references(void)
{
    py_read_data_override = p::object(); // Clear the callback
    python_heuristic_override = p::object();
}

/// @brief Helper functions for better error messages
/// @tparam T int or double
/// @return string of type T
template <typename T>
const char *type_name();
template <>
const char *type_name<double>() { return "float64"; }
template <>
const char *type_name<int>() { return "int32"; }

/// @brief Checks whether c++ is getting the correct format numpy array from Python, throws error
/// @tparam T either a double or int
/// @param np_in numpy array passed in
/// @param dimensions checks the shape of the np array
template <typename T>
void check_np_array_validity(const np::ndarray &np_in, int dimensions, const std::string &np_array_name = "")
{
    // Check dtype
    if (np_in.get_dtype() != np::dtype::get_builtin<T>())
    {
        std::string msg = np_array_name + " - Incorrect array data type: expected " + std::string(type_name<T>());
        PyErr_SetString(PyExc_TypeError, msg.c_str());
        p::throw_error_already_set();
    }
    // Check number of dimensions
    if (np_in.get_nd() != dimensions)
    {
        std::string msg = np_array_name + " - Incorrect number of dimensions: expected " +
                          std::to_string(dimensions) + ", got " + std::to_string(np_in.get_nd());
        PyErr_SetString(PyExc_TypeError, msg.c_str());
        p::throw_error_already_set();
    }

    // If 2D, check for square shape
    if (dimensions == 2 && np_in.shape(0) != np_in.shape(1))
    {
        std::string msg = np_array_name + " - Incorrect shape: expected a square (n x n) array, got a (" +
                          std::to_string(np_in.shape(0)) + " x " + std::to_string(np_in.shape(1)) + " )";
        PyErr_SetString(PyExc_ValueError, msg.c_str());
        p::throw_error_already_set();
    }

    // Check row-major contiguous
    if (!(np_in.get_flags() & np::ndarray::C_CONTIGUOUS))
    {
        std::string msg = np_array_name + " - Array must be row-major contiguous";
        PyErr_SetString(PyExc_TypeError, msg.c_str());
        p::throw_error_already_set();
    }
}

/// @brief Creates a numpy array of the solution, returned after biqbin is done solving
/// @return np.ndarray(dtype = np.int32) of the final solution (node names in a np list)
np::ndarray get_selected_nodes_np_array()
{
    // Create NumPy array to return to user
    np::dtype dtype = np::dtype::get_builtin<int>();
    p::tuple shape = p::make_tuple(selected_nodes.size());

    // Allocate NumPy array
    np::ndarray result = np::zeros(shape, dtype);
    for (int i = 0; i < static_cast<int>(selected_nodes.size()); ++i)
    {
        result[i] = selected_nodes[i];
    }
    return result;
}

/// @brief
/// @param py_args argv in a Python string list ["biqbin", "graph_instance_path", "parameters_path"]
/// @return dictionary of "max_val" value of maximum cut and "solution" vertices

/// @brief Run the solver, retrieve the solution
/// @param prog_name argv[0] "biqbin_*.py"
/// @param problem_instance_name argv[1] "problem_path_to_file"
/// @param params_file_name argv[2] "path_to_params_file"
/// @return biqbin maxcut result
p::dict run_py(char *prog_name, char *problem_instance_name, char *params_file_name)
{
    char *argv[3] = {prog_name, problem_instance_name, params_file_name};

    wrapped_main(3, argv);
    clean_python_references(); // TODO: handle python references better

    // Build result dictionary
    p::dict result_dict, nested;
    result_dict["time"] = running_time;
    result_dict["maxcut"] = nested;
    result_dict["maxcut"]["computed_val"] = Bab_LBGet();
    result_dict["maxcut"]["solution"] = get_selected_nodes_np_array();
    return result_dict;
}

double run_heuristic_python(
    const np::ndarray &P0_L_array,
    const np::ndarray &P_L_array,
    const np::ndarray &xfixed_array,
    const np::ndarray &node_sol_X_array,
    const np::ndarray &x_array)
{
    // Check if input is valid
    check_np_array_validity<double>(P0_L_array, 2, "P0_L");
    check_np_array_validity<double>(P_L_array, 2, "P_L");
    check_np_array_validity<int>(xfixed_array, 1, "xfixed");
    check_np_array_validity<int>(node_sol_X_array, 1, "node_sol_x");
    check_np_array_validity<int>(x_array, 1, "x");

    double *P0_L = reinterpret_cast<double *>(P0_L_array.get_data());
    double *P_L = reinterpret_cast<double *>(P_L_array.get_data());
    int *xfixed = reinterpret_cast<int *>(xfixed_array.get_data());
    int *node_sol_X = reinterpret_cast<int *>(node_sol_X_array.get_data());
    int *x = reinterpret_cast<int *>(x_array.get_data());

    return runHeuristic_unpacked(P0_L, P0_L_array.shape(0), P_L, P_L_array.shape(0), xfixed, node_sol_X, x);
}

/// @brief Called in runHeuristic in heuristic.c
/// @param P0 is the original Problem *SP in global_var.h
/// @param P  current subproblem Problem *PP in global_var.h
/// @param node current branch and bound node
/// @param x stores the best solution nodes found the by the heuristic function
/// @return best lower bound of the current subproblem found by the heuristic used
double wrapped_heuristic(Problem *P0, Problem *P, BabNode *node, int *x)
{
    np::ndarray P0_L_array = np::from_data(P0->L, np::dtype::get_builtin<double>(),
                                           p::make_tuple(P0->n, P0->n),
                                           p::make_tuple(sizeof(double) * P0->n, sizeof(double)),
                                           p::object());

    np::ndarray P_L_array = np::from_data(P->L, np::dtype::get_builtin<double>(),
                                          p::make_tuple(P->n, P->n),
                                          p::make_tuple(sizeof(double) * P->n, sizeof(double)),
                                          p::object());

    np::ndarray xfixed_array = np::from_data(node->xfixed, np::dtype::get_builtin<int>(),
                                             p::make_tuple(P0->n - 1),
                                             p::make_tuple(sizeof(int)),
                                             p::object());

    np::ndarray sol_X_array = np::from_data(node->sol.X, np::dtype::get_builtin<int>(),
                                            p::make_tuple(P0->n - 1),
                                            p::make_tuple(sizeof(int)),
                                            p::object());

    np::ndarray x_array = np::from_data(x, np::dtype::get_builtin<int>(),
                                        p::make_tuple(BabPbSize),
                                        p::make_tuple(sizeof(int)),
                                        p::object());

    // RK https://wiki.python.org/moin/boost.python/extract
    return p::extract<double>(python_heuristic_override(P0_L_array, P_L_array, xfixed_array, sol_X_array, x_array));
}

/// @brief Get an adjacency matrix from Python and set Problem *SP->L and *PP global variables
int wrapped_read_data()
{
    np::ndarray np_adj = p::extract<np::ndarray>(py_read_data_override());

    check_np_array_validity<double>(np_adj, 2, "adj");

    return process_adj_matrix(reinterpret_cast<double *>(np_adj.get_data()),
                              np_adj.shape(0));
}

// Python module exposure
BOOST_PYTHON_MODULE(biqbin)
{
    np::initialize();

    p::def("set_heuristic", &set_heuristic_override);
    p::def("set_read_data", &set_read_data_override);
    p::def("read_bqp_data", &read_data_BQP);
    p::def("run", &run_py);
    p::def("default_heuristic", &run_heuristic_python);
    p::def("get_rank", &get_rank);
}

/// @brief Copy the solution before memory is freed, so it can be retrieved in Python // RK ???
void copy_solution()
{
    for (int i = 0; i < BabPbSize; ++i) // RK I need to you Beno to explain me this !!!
    {
        if (BabSol->X[i] == 1)
        {
            selected_nodes.push_back(i + 1); // 1-based indexing
        }
    }
}

/// @brief record time at the end
/// @param time
void record_time(double time)
{
    running_time = time;
}