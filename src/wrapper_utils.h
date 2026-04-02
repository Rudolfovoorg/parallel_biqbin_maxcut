#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// From NumPy C API — stable ABI value, never changes
static constexpr int NPY_WRITEABLE_FLAG = 0x0400;

template <typename T>
void check_np_array_validity(const py::array_t<T> &np_in, int expected_ndim, bool expected_writable, const std::string &np_array_name)
{
    // Check number of dimensions
    if (np_in.ndim() != expected_ndim)
    {
        throw py::type_error(np_array_name + " must have " + std::to_string(expected_ndim) +
                             " dimensions, got " + std::to_string(np_in.ndim()));
    }
    // If 2D, check if square
    if (expected_ndim == 2)
    {
        if (np_in.shape(0) != np_in.shape(1))
        {
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
    if (np_in.writeable() != expected_writable)
    {
        throw py::type_error(np_array_name + " has wrong writeable flag.");
    }
}

template <typename T>
py::array_t<T> wrapped_array(T *data, ssize_t size, bool writable)
{
    py::array_t<T> arr(
        {size},      // shape
        {sizeof(T)}, // stride
        data,        // pointer to data memory
        py::none()   // noop deleter, Python won't free memory
    );
    if (!writable)
    {
        py::detail::array_proxy(arr.ptr())->flags &= ~NPY_WRITEABLE_FLAG;
    }
    return arr;
}

template <typename T>
py::array_t<T> wrapped_matrix(T *data, ssize_t rows, ssize_t cols, bool writable)
{
    py::array_t<T> arr(
        {rows, cols},                  // shape
        {sizeof(T) * cols, sizeof(T)}, // row-major strides
        data,                          // pointer to data memory
        py::none()                     // noop deleter, Python won't free memory
    );
    if (!writable)
    {
        py::detail::array_proxy(arr.ptr())->flags &= ~NPY_WRITEABLE_FLAG;
    }

    return arr;
}

template <typename T>
py::array_t<T> get_numpy_array_from_vec(const std::vector<T> &vec)
{
    return py::array_t<T>(vec.size(), vec.data());
}