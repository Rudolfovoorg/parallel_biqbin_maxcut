
#include <iostream>

#include <mpi.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // for std::vector

#include "biqbin_cpp_api.h"
#include "blas_laplack.h"

#include "wrapper.h"
#include "wrapper_utils.h"
#include "wrapper_hooks.h"

namespace py = pybind11;

/* biqbin's global variables from global_var.h */
extern Problem *SP;
extern Problem *PP;
extern BabSolution *BabSol;
extern int BabPbSize;

/* final solution */
std::vector<int> selected_nodes;
std::vector<int> solution_x;

/* meta_data */
extern int num_workers_used;
extern int time_limit_reached;
extern int heuristic_counter;
extern int heuristic_sum;
/* root meta_data */
extern double root_upper_bound;
extern double root_lower_bound;
extern double root_eval_time;
std::vector<int> root_sol_x;

/* MPI data */
extern int rank;
extern int num_workers;

static double running_time;
static int time_limit;

// Python received problem, memory is owned by Python
static double *adj_matrix;
static int adj_matrix_size;

int get_rank()
{
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "MPI is not initialized. Call biqbin.init() before using any solver.");
        return -1;
    }
    return rank;
}
int get_time_limit() { return time_limit; }

/// @brief Run the solver, retrieve the solution
/// @param prog_name argv[0] "biqbin_*.py"
/// @param problem_instance_name argv[1] "problem_path_to_file"
/// @param params_file_name argv[2] "path_to_params_file"
/// @return biqbin maxcut result
py::dict run_py(char *prog_name, char *problem_instance_name, py::array_t<double> &adj_matrix_in, char *params_file_name, int time_limit_in)
{
    // heuristic guard to clean Python references on exit
    PythonReferencesGuard heuristic_guard;

    // Check if MPI was initialized before running
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "MPI is not initialized. Call biqbin.init() before using any solver.");
        return py::dict();
    }

    time_limit = time_limit_in;
    heuristic_counter = 0;

    if (rank == 0)
    {
        // One last safety check
        check_np_array_validity<double>(adj_matrix_in, 2, true, "maxcut_adjacency_matrix");
        // Set the problem data, memory owned by Python
        adj_matrix_size = adj_matrix_in.shape(0);
        adj_matrix = static_cast<double *>(adj_matrix_in.mutable_data());
    }

    char *argv[3] = {prog_name, problem_instance_name, params_file_name};
    wrapped_main(3, argv);

    // Save results
    py::dict result_dict;
    py::dict solution_info;
    py::dict meta_data;
    py::dict root_node_dict;

    meta_data["time"] = running_time;
    meta_data["time_limit_reached"] = (time_limit_reached) ? true : false;
    meta_data["eval_bab_nodes"] = Bab_numEvalNodes();
    meta_data["heuristic_run_count"] = heuristic_sum;
    meta_data["num_workers_used"] = num_workers_used;

    root_node_dict["time"] = root_eval_time;
    root_node_dict["heuristic_value"] = root_lower_bound;
    root_node_dict["heuristic_run_count"] = heuristic_counter;
    root_node_dict["sdp_value"] = root_upper_bound;
    root_node_dict["root_solution"] = py::cast(root_sol_x);
    meta_data["root_node"] = root_node_dict;

    solution_info["computed_val"] = Bab_LBGet();
    solution_info["solution"] = py::cast(selected_nodes); // we converted it from numpy to regular list immidiately in python so might as well do it here.
    solution_info["x"] = py::cast(solution_x);
    result_dict["meta_data"] = meta_data;
    result_dict["maxcut"] = solution_info;

    return result_dict;
}

/// @brief Default GW heuristic
/// @param P0_L_array       Main Problem L: SP->L
/// @param P_L_array        Subproblem L: PP->L
/// @param xfixed_array     Fixed variables in solution x
/// @param node_sol_X_array Solution stored in current babnode
/// @param x_array          Heuristic solution x this
/// @return                 Lower bound of heuristic solution
double run_heuristic_python(
    py::array_t<double> P0_L_array,
    py::array_t<double> P_L_array,
    py::array_t<int> xfixed_array,
    py::array_t<int> node_sol_X_array,
    py::array_t<int> x_array)
{
    // Check if input is valid
    check_np_array_validity<double>(P0_L_array, 2, false, "P0_L");
    check_np_array_validity<double>(P_L_array, 2, false, "P_L");
    check_np_array_validity<int>(xfixed_array, 1, false, "xfixed");
    check_np_array_validity<int>(node_sol_X_array, 1, false, "node_sol_x");
    check_np_array_validity<int>(x_array, 1, true, "x");

    const auto P0_L = P0_L_array.data();
    const auto P_L = P_L_array.data();
    const auto xfixed = xfixed_array.data();
    const auto node_sol_X = node_sol_X_array.data();
    const auto x = x_array.mutable_data(); // only x is modified

    return runHeuristic_unpacked(P0_L, P0_L_array.shape(0), P_L, P_L_array.shape(0), xfixed, node_sol_X, x);
}

/// @brief Get an adjacency matrix from Python and set Problem *SP->L and *PP global variables
int wrapped_read_data()
{
    return process_adj_matrix(adj_matrix, adj_matrix_size);
}

bool update_solution_python(const py::array_t<int> potential_solution)
{
    check_np_array_validity(potential_solution, 1, true, "new solution x");
    int is_updated = updateSolution(potential_solution.data());
    return is_updated != 0;
}

/// @brief Copy the solution before memory is freed, so it can be retrieved in Python
void copy_solution()
{
    for (int i = 0; i < BabPbSize; ++i)
    {
        solution_x.push_back(BabSol->X[i]); // binary solution vec x
        if (BabSol->X[i] == 1)
        {
            selected_nodes.push_back(i + 1); // 1-based indexing
        }
    }
    solution_x.push_back(0); // .. solution is one more than BabPbSize
}

/// @brief Copy the solution before memory is freed, so it can be retrieved in Python
void copy_root_solution()
{
    for (int i = 0; i < BabPbSize; ++i)
    {
        root_sol_x.push_back(BabSol->X[i]); // binary solution vec x
    }
    root_sol_x.push_back(0); // .. solution is one more than BabPbSize
}

/// @brief record time at the end
/// @param time
void record_time(double time) { running_time = time; }

py::tuple init_mpi_python()
{
    // Initialize MPI without propagating CLI arguments
    MPI_Init(NULL, NULL);
    // get number of proccesses and corresponding ranks
    MPI_Comm_size(MPI_COMM_WORLD, &num_workers);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return py::make_tuple(num_workers, rank);
}

PYBIND11_MODULE(biqbin_module, m, "Biqbin solver")
{
    m.def("init_mpi", &init_mpi_python, "Initialize MPI protocol");
    m.def("set_heuristic", &set_heuristic_override, "Override the heuristic function");
    m.def("run", &run_py, "Run the solver");
    m.def("goemans_williamson_heuristic", &run_heuristic_python, "Default C-implemented GW heuristic");
    m.def("update_mc_lower_bound_solution", &update_solution_python, "updates the MaxCut lower-bound solution if it is better than the current one");
    m.def("get_rank", &get_rank, "Get the mpi rank");

    m.def("set_node_evaluation", &set_node_evaluation_override, "Override the SDP bound function");
    m.def("sdp_bound", &SDPbound, "Default C-implemented SDPbound");

    py::class_<BabSolution>(m, "BabSolution")
        .def(py::init<>())
        .def_property_readonly("X", &detail::babsolution_get_X);

    py::class_<BabNode>(m, "BabNode")
        .def(py::init<>())
        .def_property_readonly("xfixed", &detail::babnode_get_xfixed)
        .def_property_readonly("sol", &detail::babnode_get_sol)
        .def_readonly("level", &BabNode::level)
        .def_readonly("upper_bound", &BabNode::upper_bound)
        .def("fracsol", &detail::babnode_get_fracsol);

    py::class_<Problem>(m, "Problem")
        .def(py::init<>())
        .def_property_readonly("L", &detail::problem_get_L)
        .def_readonly("n", &Problem::n)
        .def_readonly("NIneq", &Problem::NIneq)
        .def_readonly("NPentIneq", &Problem::NPentIneq)
        .def_readonly("NHeptaIneq", &Problem::NHeptaIneq)
        .def_readonly("bundle", &Problem::bundle);
}