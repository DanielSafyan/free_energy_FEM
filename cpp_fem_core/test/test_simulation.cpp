#include "TetrahedralMesh.hpp"
#include "NPENSimulation.hpp"
#include <iostream>
#include <Eigen/Dense>

int main() {
    std::cout << "Testing C++ FEM Core Implementation" << std::endl;
    
    // Create a simple test mesh (unit tetrahedron)
    Eigen::MatrixXd nodes(4, 3);
    nodes << 0, 0, 0,
             1, 0, 0,
             0, 1, 0,
             0, 0, 1;
    
    Eigen::MatrixXi elements(1, 4);
    elements << 0, 1, 2, 3;
    
    // Create mesh
    TetrahedralMesh mesh(nodes, elements);
    
    // Print mesh info
    std::cout << "Created mesh with " << mesh.numNodes() << " nodes and " 
              << mesh.numElements() << " elements" << std::endl;
    
    // Test element data computation
    const ElementData& elemData = mesh.getElementData(0);
    std::cout << "Element volume: " << elemData.volume << std::endl;
    std::cout << "Element gradients:\n" << elemData.grads << std::endl;
    
    // Create simulation
    double dt = 0.01;
    double D1 = 1e-9, D2 = 1e-9, D3 = 1e-9;
    int z1 = 1, z2 = -1;
    double epsilon = 8.854e-12;
    double R = 8.314, T = 298;
    double L_c = 1e-3, c0 = 1.0;
    
    auto meshPtr = std::make_shared<TetrahedralMesh>(mesh);
    // New constructor expects split coefficients: (D_diff1, D_mig1, D_diff2, D_mig2)
    // Use D1 for both diffusion and migration of species 1, and D2 for species 2 in this smoke test
    NPENSimulation simulation(meshPtr, dt, /*D_diff1*/ D1, /*D_mig1*/ D1,
                              /*D_diff2*/ D2, /*D_mig2*/ D2,
                              D3, z1, z2, epsilon, R, T, L_c, c0);
    
    // Create initial conditions
    Eigen::VectorXd c_prev = Eigen::VectorXd::Constant(4, 1.0);
    Eigen::VectorXd phi_prev = Eigen::VectorXd::Constant(4, 0.0);
    
    Eigen::VectorXd c_next(4), phi_next(4);
    
    // Perform one step
    simulation.step(c_prev, phi_prev, c_next, phi_next);
    
    // Print results
    std::cout << "After one step:" << std::endl;
    std::cout << "c: " << c_next.transpose() << std::endl;
    // No c3 in reduced NPEN model
    std::cout << "phi: " << phi_next.transpose() << std::endl;
    
    std::cout << "Test completed successfully!" << std::endl;
    
    return 0;
}
