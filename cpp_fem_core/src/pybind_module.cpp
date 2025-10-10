#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "TetrahedralMesh.hpp"
#include "NPENSimulation.hpp"
#include "FluxCalculator.hpp"
#include "CurrentCalculator.hpp"

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
        .def("getBoundaryFaces", &TetrahedralMesh::getBoundaryFaces)
        .def("getElementData", &TetrahedralMesh::getElementData, 
             py::return_value_policy::reference_internal);
    
    // Bind NPENSimulation class
    py::class_<NPENSimulation, std::shared_ptr<NPENSimulation>>(m, "NPENSimulation")
        .def(py::init<std::shared_ptr<TetrahedralMesh>, double,
                      double, double, double, double, double,
                      int, int, double, double, double, double, double>())
        .def("setAdvectionScheme", &NPENSimulation::setAdvectionScheme, py::arg("scheme"))
        .def("setElectrodeFaces", &NPENSimulation::setElectrodeFaces,
             py::arg("face_sets"), py::arg("voltages"), py::arg("k_reaction"))
        .def("step", [](NPENSimulation& self, 
                         const Eigen::VectorXd& c_prev, 
                         const Eigen::VectorXd& phi_prev) {
            Eigen::VectorXd c_next(c_prev.size()), phi_next(phi_prev.size());
            self.step(c_prev, phi_prev, c_next, phi_next);
            return std::make_tuple(c_next, phi_next);
        })
        .def("step2", [](NPENSimulation& self,
                          const Eigen::VectorXd& c_prev,
                          const Eigen::VectorXd& phi_prev,
                          const Eigen::VectorXi& electrode_indices,
                          const Eigen::VectorXd& applied_voltages,
                          double rtol = 1e-3, double atol = 1e-14, int max_iter = 50, double k_reaction = 0.5) {
            Eigen::VectorXd c_next(c_prev.size()), phi_next(phi_prev.size());
            self.step2(c_prev, phi_prev, electrode_indices, applied_voltages,
                       c_next, phi_next, rtol, atol, max_iter, k_reaction);
            return std::make_tuple(c_next, phi_next);
        }, py::arg("c_prev"), py::arg("phi_prev"),
           py::arg("electrode_indices"), py::arg("applied_voltages"),
           py::arg("rtol") = 1e-3, py::arg("atol") = 1e-14, py::arg("max_iter") = 50,
           py::arg("k_reaction") = 0.5)
        .def("step2_many", [](NPENSimulation& self,
                               const Eigen::VectorXd& c0,
                               const Eigen::VectorXd& phi0,
                               const Eigen::VectorXi& electrode_indices,
                               const Eigen::VectorXd& applied_voltages,
                               int steps,
                               double rtol = 1e-3, double atol = 1e-14, int max_iter = 50, double k_reaction = 0.5) {
            Eigen::MatrixXd c_hist, phi_hist;
            self.step2_many(c0, phi0, electrode_indices, applied_voltages,
                            steps, c_hist, phi_hist, rtol, atol, max_iter, k_reaction);
            return std::make_tuple(c_hist, phi_hist);
        }, py::arg("c0"), py::arg("phi0"), py::arg("electrode_indices"), py::arg("applied_voltages"),
           py::arg("steps"), py::arg("rtol") = 1e-3, py::arg("atol") = 1e-14, py::arg("max_iter") = 50,
           py::arg("k_reaction") = 0.5)
        .def("getPhiC", &NPENSimulation::getPhiC)
        .def("getC0", &NPENSimulation::getC0);
    
    // Bind FluxCalculator class
    py::class_<FluxCalculator, std::shared_ptr<FluxCalculator>>(m, "FluxCalculator")
        .def(py::init<std::shared_ptr<TetrahedralMesh>, double, double>())
        .def("computeFlux", [](FluxCalculator& self, 
                               const Eigen::VectorXd& c, 
                               const Eigen::VectorXd& phi) {
            Eigen::MatrixXd flux_vectors;
            self.computeFlux(c, phi, flux_vectors);
            return flux_vectors;
        })
        .def("computeFluxHistory", [](FluxCalculator& self,
                                      const Eigen::MatrixXd& c_history,
                                      const Eigen::MatrixXd& phi_history) {
            std::vector<Eigen::MatrixXd> flux_tensor;
            self.computeFluxHistory(c_history, phi_history, flux_tensor);
            return flux_tensor;
        })
        .def("computeGradient", [](FluxCalculator& self,
                                   const Eigen::VectorXd& field) {
            Eigen::MatrixXd gradient;
            self.computeGradient(field, gradient);
            return gradient;
        })
        .def("getD1", &FluxCalculator::getD1)
        .def("getZ1", &FluxCalculator::getZ1)
        .def("getMesh", &FluxCalculator::getMesh, 
             py::return_value_policy::reference_internal);
    
    // Bind CurrentCalculator class
    py::class_<CurrentCalculator, std::shared_ptr<CurrentCalculator>>(m, "CurrentCalculator")
        .def(py::init<std::shared_ptr<TetrahedralMesh>, double, double, double,
                      double, double, double, double, double, double, double, double>(),
             py::arg("mesh"), py::arg("R"), py::arg("T"), py::arg("F"),
             py::arg("D_diff1"), py::arg("D_mig1"), py::arg("D_diff2"), py::arg("D_mig2"),
             py::arg("z1"), py::arg("z2"), py::arg("c0"), py::arg("phi_c"))
        .def("computeCurrentScalar", [](CurrentCalculator& self,
                                         const Eigen::VectorXd& c,
                                         const Eigen::VectorXd& phi) {
            Eigen::VectorXd current;
            self.computeCurrentScalar(c, phi, current);
            return current;
        })
        .def("computeCurrentHistory", [](CurrentCalculator& self,
                                          const Eigen::MatrixXd& c_hist,
                                          const Eigen::MatrixXd& phi_hist) {
            Eigen::MatrixXd out;
            self.computeCurrentHistory(c_hist, phi_hist, out);
            return out;
        })
        .def("writeCurrentToH5", &CurrentCalculator::writeCurrentToH5,
             py::arg("h5Filename"), py::arg("current"),
             py::arg("datasetPath") = std::string("states/current"),
             py::arg("overwrite") = false)
        .def("getR", &CurrentCalculator::getR)
        .def("getT", &CurrentCalculator::getT)
        .def("getF", &CurrentCalculator::getF)
        .def("getD1", &CurrentCalculator::getD1)
        .def("getD2", &CurrentCalculator::getD2)
        .def("getZ1", &CurrentCalculator::getZ1)
        .def("getZ2", &CurrentCalculator::getZ2)
        .def("getC0", &CurrentCalculator::getC0)
        .def("getPhiC", &CurrentCalculator::getPhiC)
        .def("getMesh", &CurrentCalculator::getMesh, 
             py::return_value_policy::reference_internal);
}
