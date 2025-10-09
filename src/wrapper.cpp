
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // for std::vector

#include "biqbin_cpp_api.h"
#include "blas_laplack.h"

#include "wrapper.h"

namespace py = pybind11;

/* biqbin's global variables from global_var.h */
extern Problem *SP;
extern Problem *PP;
extern BabSolution *BabSol;
extern int BabPbSize;

/* final solution */
std::vector<int> selected_nodes;
std::vector<int> solution_x;


/* bqp funtions */
extern py::dict read_solution_bqp();
extern double* read_data_bqp(const char *instance, int *adj_N);
extern double* post_process_BQP_input(InputData input_data, int* adj_N);

/* meta_data */
extern int num_workers_used;
extern int time_limit_reached;
extern int rank;
extern int heuristic_counter;
extern int heuristic_sum;

double running_time;
int time_limit;

// Python override functions
py::object python_heuristic_override;
py::object py_read_data_override;

/// @brief set heuristic function from python
/// @param func
void set_heuristic_override(py::object func) { python_heuristic_override = func; }

/// @brief set problem instance reading function from Python
/// @param func
void set_read_data_override(py::object func) { py_read_data_override = func; }

int get_rank() { return rank; }
int get_time_limit() { return time_limit; }

/// @brief TODO: find a better fix for conflicts with MPI
void clean_python_references(void)
{
    py_read_data_override = py::object();
    python_heuristic_override = py::object();
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
void check_np_array_validity(const py::array_t<T> &np_in, int expected_ndim, const std::string &np_array_name)
{
    // Ensure the array has the correct dtype
    if (!py::isinstance<py::array_t<T>>(np_in))
    {
        throw py::type_error(np_array_name + " must have dtype " + type_name<T>());
    }

    // Check number of dimensions
    if (np_in.ndim() != expected_ndim)
    {
        throw py::type_error(np_array_name + " must have " + std::to_string(expected_ndim) +
                             " dimensions, got " + std::to_string(np_in.ndim()));
    }

    // If 2D, check if square
    if (expected_ndim == 2) {
        if (np_in.shape(0) != np_in.shape(1)) {
            throw py::type_error(np_array_name + " must be square (shape[0] == shape[1]), got shape (" +
                                 std::to_string(np_in.shape(0)) + ", " + std::to_string(np_in.shape(1)) + ")");
        }
    }

    // Ensure the array is row-major (C-contiguous)
    if (!(np_in.flags() & py::array::c_style))
    {
        throw py::type_error(np_array_name + " must be row-major (C-contiguous).");
    }

    // Ensure the array is writable
    if (!np_in.writeable())
    {
        throw py::type_error(np_array_name + " must be writable.");
    }
}

/// @brief Creates a numpy array of the solution, returned after biqbin is done solving
/// @return np.ndarray(dtype = np.int32) of the final solution (node names in a np list)
template <typename T>
py::array_t<T> get_numpy_array_from_vec(std::vector<T> &in_vec)
{
    auto result = py::array_t<T>(in_vec.size());
    auto buf = result.template mutable_unchecked<1>();
    for (size_t i = 0; i < in_vec.size(); ++i)
    {
        buf(i) = in_vec[i];
    }
    return result;
}

/// @brief Run the solver, retrieve the solution
/// @param prog_name argv[0] "biqbin_*.py"
/// @param problem_instance_name argv[1] "problem_path_to_file"
/// @param params_file_name argv[2] "path_to_params_file"
/// @return biqbin maxcut result
py::dict run_py(char* prog_name, char* problem_instance_name, char* params_file_name, int time_limit_in)
{
    time_limit = time_limit_in;
    heuristic_counter = 0;

    char *argv[3] = {prog_name, problem_instance_name, params_file_name};
    wrapped_main(3, argv);
    clean_python_references();

    // Save results
    py::dict result_dict;
    py::dict solution_info;
    py::dict meta_data;

    meta_data["time"] = running_time;
    meta_data["time_limit_reached"] = (time_limit_reached) ? true : false;
    meta_data["eval_bab_nodes"] = Bab_numEvalNodes();
    meta_data["heuristic_run_count"] = heuristic_sum;
    meta_data["num_workers_used"] = num_workers_used;
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
/// @param x_array          Heuristic solution x
/// @return                 Lower bound of heuristic solution
double run_heuristic_python(
    py::array_t<double> P0_L_array,
    py::array_t<double> P_L_array,
    py::array_t<int> xfixed_array,
    py::array_t<int> node_sol_X_array,
    py::array_t<int> x_array)
{
    // Check if input is valid
    check_np_array_validity<double>(P0_L_array, 2, "P0_L");
    check_np_array_validity<double>(P_L_array, 2, "P_L");
    check_np_array_validity<int>(xfixed_array, 1, "xfixed");
    check_np_array_validity<int>(node_sol_X_array, 1, "node_sol_x");
    check_np_array_validity<int>(x_array, 1, "x");

    const auto P0_L = P0_L_array.mutable_data();
    const auto P_L = P_L_array.mutable_data();
    const auto xfixed = xfixed_array.mutable_data();
    const auto node_sol_X = node_sol_X_array.mutable_data();
    const auto x = x_array.mutable_data(); // only x is modified

    return runHeuristic_unpacked(P0_L, P0_L_array.shape(0), P_L, P_L_array.shape(0), xfixed, node_sol_X, x);
}

// Helper to wrap C++ arrays without letting Python own them
template <typename T>
py::array_t<T> wrapped_array(T *data, ssize_t size)
{
    return py::array_t<T>(
        {size},           // shape
        {sizeof(T)},      // stride
        data,             // pointer to memory
        py::cast(nullptr) // noop deleter, Python won't free memory
    );
}

template <typename T>
py::array_t<T> wrapped_matrix(T *data, ssize_t rows, ssize_t cols)
{
    return py::array_t<T>(
        {rows, cols},                  // shape
        {sizeof(T) * cols, sizeof(T)}, // row-major strides
        data,
        py::cast(nullptr)); // noop deleter, Python won't free memory
}

/// @brief Called in runHeuristic in heuristic.c
/// @param P0 is the original Problem *SP in global_var.h
/// @param P  current subproblem Problem *PP in global_var.h
/// @param node current branch and bound node
/// @param x stores the best solution nodes found the by the heuristic function
/// @return best lower bound of the current subproblem found by the heuristic used
double wrapped_heuristic(Problem *P0, Problem *P, BabNode *node, int *x)
{
    heuristic_counter++;
    // Wrap matrices
    py::array_t<double> P0_L_array = wrapped_matrix(P0->L, P0->n, P0->n);
    py::array_t<double> P_L_array = wrapped_matrix(P->L, P->n, P->n);

    // Wrap vectors
    py::array_t<int> xfixed_array = wrapped_array(node->xfixed, P0->n - 1);
    py::array_t<int> sol_X_array = wrapped_array(node->sol.X, P0->n - 1);
    py::array_t<int> x_array = wrapped_array(x, BabPbSize);

    // Call Python override
    return python_heuristic_override(
               P0_L_array, P_L_array, xfixed_array, sol_X_array, x_array)
        .cast<double>();
}

/// @brief Read the instance problem file return the adjacency matrix
/// @param instance path to instance file
/// @return adjacency matrix
py::array_t<double> read_data_python(const std::string &instance)
{
    double *adj;
    int adj_N;
    adj = readData(instance.c_str(), &adj_N);

    return py::array_t<double>({adj_N, adj_N}, adj);
}

/// @brief Read the bqp problem file return the adjacency matrix
/// @param instance path to instance file
/// @return adjacency matrix
py::array_t<double> read_data_bqp_python(const std::string &instance)
{
    double *adj;
    int adj_N;
    adj = read_data_bqp(instance.c_str(), &adj_N);

    return py::array_t<double>({adj_N, adj_N}, adj);
}

/// @brief Read and parse bqp json data
/// @param instance Built in DataGetterBQPJson
/// @return adjacency matrix
py::array_t<double> read_data_bqp_json_python(py::dict &instance) 
{
    double *adj;
    int adj_N;
    InputData input_data;
    input_data.n = instance["number_of_variables"].cast<int>();
    input_data.m = instance["number_of_constraints"].cast<int>();

    // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html - under "vectorizing functions"
    // grab the underlying numpy array
    py::array_t<double> A_array = instance["Am"].cast<py::array_t<double>>();
    py::array_t<double> F_array = instance["Fm"].cast<py::array_t<double>>();
    py::array_t<double> c_array = instance["cm"].cast<py::array_t<double>>();
    py::array_t<double> b_array = instance["bm"].cast<py::array_t<double>>();

    // grab the buffer info
    auto A_buf = A_array.request();
    auto F_buf = F_array.request();
    auto c_buf = c_array.request();
    auto b_buf = b_array.request();

    // cast buffer info into a double* raw pointer, it is owned by Python
    input_data.A = static_cast<double*>(A_buf.ptr);
    input_data.F = static_cast<double*>(F_buf.ptr);
    input_data.c = static_cast<double*>(c_buf.ptr);
    input_data.b = static_cast<double*>(b_buf.ptr);

    // Original function
    adj = post_process_BQP_input(input_data, &adj_N);

    return py::array_t<double>({adj_N, adj_N}, adj);
}

/// @brief Get an adjacency matrix from Python and set Problem *SP->L and *PP global variables
int wrapped_read_data()
{
    py::array np_adj;
    try
    {
        np_adj = py_read_data_override().cast<py::array>();
    }
    catch (const py::error_already_set &e)
    {
        std::cerr << "Python error: " << e.what() << std::endl;
        std::exit(1);
    }
    check_np_array_validity<double>(np_adj, 2, "adj");
    return process_adj_matrix(static_cast<double *>(np_adj.mutable_data()), np_adj.shape(0));
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

/// @brief record time at the end
/// @param time
void record_time(double time) { running_time = time; }

PYBIND11_MODULE(biqbin, m)
{
    m.def("set_heuristic", &set_heuristic_override);
    m.def("set_read_data", &set_read_data_override);
    m.def("run", &run_py);
    m.def("default_heuristic", &run_heuristic_python);
    m.def("default_read_data", &read_data_python);
    m.def("read_data_bqp", &read_data_bqp_python);
    m.def("read_data_bqp_json", &read_data_bqp_json_python);
    m.def("read_solution_bqp", &read_solution_bqp);
    m.def("get_rank", &get_rank);
}