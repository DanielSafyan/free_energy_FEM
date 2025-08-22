#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "TetrahedralMesh.hpp"
#include "NPENSimulation.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fem_core_py, m) {
    m.doc() = "FEM Core Implementation for NPEN Simulation";
    
    // Bind ElementData struct
    py::class_<ElementData>(m, "ElementData")
        .def(py::init<>())
        .def_readonly("volume", &ElementData::volume)
        .def_readonly("grads", &ElementData::grads);
    
    // Bind TetrahedralMesh class
    py::class_<TetrahedralMesh, std::shared_ptr<TetrahedralMesh>>(m, "TetrahedralMesh")
        .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXi&>())
        .def("numNodes", &TetrahedralMesh::numNodes)
        .def("numElements", &TetrahedralMesh::numElements)
        .def("getNodes", &TetrahedralMesh::getNodes)
        .def("getElements", &TetrahedralMesh::getElements)
        .def("getElementData", &TetrahedralMesh::getElementData, 
             py::return_value_policy::reference_internal);
    
    // Bind NPENSimulation class
    py::class_<NPENSimulation, std::shared_ptr<NPENSimulation>>(m, "NPENSimulation")
        .def(py::init<std::shared_ptr<TetrahedralMesh>, double, double, double, double, 
                      int, int, double, double, double, double, double>())
        .def("step", [](NPENSimulation& self, 
                         const Eigen::VectorXd& c_prev, 
                         const Eigen::VectorXd& c3_prev, 
                         const Eigen::VectorXd& phi_prev) {
            Eigen::VectorXd c_next(c_prev.size()), c3_next(c3_prev.size()), phi_next(phi_prev.size());
            self.step(c_prev, c3_prev, phi_prev, c_next, c3_next, phi_next);
            return std::make_tuple(c_next, c3_next, phi_next);
        })
        .def("step2", [](NPENSimulation& self,
                          const Eigen::VectorXd& c_prev,
                          const Eigen::VectorXd& c3_prev,
                          const Eigen::VectorXd& phi_prev,
                          const Eigen::VectorXi& electrode_indices,
                          const Eigen::VectorXd& applied_voltages,
                          double rtol = 1e-3, double atol = 1e-14, int max_iter = 50) {
            Eigen::VectorXd c_next(c_prev.size()), c3_next(c3_prev.size()), phi_next(phi_prev.size());
            self.step2(c_prev, c3_prev, phi_prev, electrode_indices, applied_voltages,
                       c_next, c3_next, phi_next, rtol, atol, max_iter);
            return std::make_tuple(c_next, c3_next, phi_next);
        }, py::arg("c_prev"), py::arg("c3_prev"), py::arg("phi_prev"),
           py::arg("electrode_indices"), py::arg("applied_voltages"),
           py::arg("rtol") = 1e-3, py::arg("atol") = 1e-14, py::arg("max_iter") = 50)
        .def("getPhiC", &NPENSimulation::getPhiC)
        .def("getC0", &NPENSimulation::getC0);
}
