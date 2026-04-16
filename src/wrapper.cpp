
#include <iostream>

#include <mpi.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // for std::vector

#include "biqbin_cpp_api.h"
#include "blas_laplack.h"
#include "parameters.h"
#include "wrapper_utils.h"

#include "wrapper.h"

namespace py = pybind11;

/* biqbin's global variables from global_var.h */
extern BiqBinParameters params;
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

double running_time;

// Python override functions
py::object python_heuristic_override;

/// @brief RAII guard for the python_heuristic_override function, cleans up Python references after the solver is run.
struct HeuristicGuard
{
    ~HeuristicGuard() { python_heuristic_override = py::object(); }
};

// Python received problem, memory is owned by Python
const double *adj_matrix;
int adj_matrix_size;

/// @brief set heuristic function from python
/// @param func
void set_heuristic_override(py::object func) { python_heuristic_override = func; }

int get_rank() { return rank; }

/// @brief Run the solver, retrieve the solution
/// @param prog_name argv[0] "biqbin_*.py"
/// @param problem_instance_name argv[1] "problem_path_to_file"
/// @param params_in BiqbinParameters instanced initialized in Python
/// @return biqbin maxcut result
py::dict run_py(char *prog_name, char *problem_instance_name, py::array_t<double> &adj_matrix_in, BiqBinParameters &params_in)
{
    // heuristic guard to clean Python references on exit
    HeuristicGuard heuristic_guard;

    params = params_in;
    if (rank == 0)
    {
        // One last safety check
        check_np_array_validity<double>(adj_matrix_in, 2, true, "maxcut_adjacency_matrix");
        // Set the problem data, memory owned by Python
        adj_matrix_size = adj_matrix_in.shape(0);
        adj_matrix = static_cast<const double *>(adj_matrix_in.data());
    }

    // wrapped main expects 3 arguments
    char *argv[3] = {prog_name, problem_instance_name, "params"};
    wrapped_main(3, argv);

    // If not master rank return empty dict
    if (rank != 0)
    {
        return py::dict();
    }

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
    solution_info["solution"] = py::cast(selected_nodes);
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

    const auto P0_L = P0_L_array.data();             // const
    const auto P_L = P_L_array.data();               // const
    const auto xfixed = xfixed_array.data();         // const
    const auto node_sol_X = node_sol_X_array.data(); // const
    const auto x = x_array.mutable_data();           // mutable

    return runHeuristic_unpacked(P0_L, P0_L_array.shape(0), P_L, P_L_array.shape(0), xfixed, node_sol_X, x);
}

/// @brief Called in runHeuristic in heuristic.c
/// @param P0 is the original Problem *SP in global_var.h
/// @param P  current subproblem Problem *PP in global_var.h
/// @param node current branch and bound node
/// @param x stores the best solution nodes found the by the heuristic function
/// @return best lower bound of the current subproblem found by the heuristic used
double wrapped_heuristic(const Problem *P0, const Problem *P, const BabNode *node, int *x)
{
    heuristic_counter++;
    // Wrap matrices
    py::array_t<double> P0_L_array = wrapped_matrix(P0->L, P0->n, P0->n, false);
    py::array_t<double> P_L_array = wrapped_matrix(P->L, P->n, P->n, false);

    // Wrap vectors
    py::array_t<int> xfixed_array = wrapped_array(node->xfixed, P0->n - 1, false);
    py::array_t<int> sol_X_array = wrapped_array(node->sol.X, P0->n - 1, false);
    py::array_t<int> x_array = wrapped_array(x, BabPbSize, true);

    // Call Python override
    return python_heuristic_override(
               P0_L_array,
               P_L_array,
               xfixed_array,
               sol_X_array,
               x_array)
        .cast<double>();
}

/// @brief Get an adjacency matrix from Python and set Problem *SP->L and *PP global variables
int wrapped_read_data()
{
    return process_adj_matrix(adj_matrix, adj_matrix_size);
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
        root_sol_x.push_back(BabSol->X[i]); // binary solution vector x
    }
    root_sol_x.push_back(0); // NOTE: Solution is one more than BabPbSize
}

/// @brief Record solving time at the end
/// @param time
void record_time(double time) { running_time = time; }

/// @brief Expose MPI_Init to Python
/// @return (size, rank) tuple
py::tuple init_mpi_python()
{
    // Initialize MPI without propagating CLI arguments
    MPI_Init(NULL, NULL);
    // get number of proccesses and corresponding ranks
    MPI_Comm_size(MPI_COMM_WORLD, &num_workers);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return py::make_tuple(num_workers, rank);
}

PYBIND11_MODULE(biqbin_module, m)
{
    m.doc() = "Biqbin solver";
    m.def("init_mpi", &init_mpi_python, "Initialize MPI protocol");
    m.def("set_heuristic", &set_heuristic_override, "Override the heuristic function");
    m.def("run", &run_py, "Run the solver");
    m.def("goemans_williamson_heuristic", &run_heuristic_python, "Default C-implemented GW heuristic");
    m.def("get_rank", &get_rank, "Get the mpi rank");

    py::class_<BiqBinParameters>(m, "_Parameters")
        .def(py::init([]()
                      {
        BiqBinParameters p;
#define P(type, name, format, def_value) p.name = def_value;
        PARAM_FIELDS
#undef P
        return p; }))
#define P(type, name, format, def_value) \
    .def_readwrite(#name, &BiqBinParameters::name)
            PARAM_FIELDS
#undef P
        ;
}