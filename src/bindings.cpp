#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .export_values();

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int>&, Device>(), 
             py::arg("shape"), py::arg("device") = Device::CPU)
        .def(py::init<const std::vector<float>&, const std::vector<int>&, Device>(), 
             py::arg("data"), py::arg("shape"), py::arg("device") = Device::CPU)
        .def_property_readonly("shape", &Tensor::get_shape)
        .def_property_readonly("device", &Tensor::get_device)
        .def("size", &Tensor::size)
        .def("to", &Tensor::to)
        .def("to_list", &Tensor::to_vector)
        .def("add", &Tensor::add)
        .def("sub", &Tensor::sub)
        .def("mul", &Tensor::mul)
        .def("matmul", &Tensor::matmul)
        .def("transpose", &Tensor::transpose)
        .def("fill", &Tensor::fill);
}
