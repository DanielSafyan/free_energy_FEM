#ifndef NPEN_SIMULATION_HPP
#define NPEN_SIMULATION_HPP

#include "TetrahedralMesh.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

class NPENSimulation {
public:
    // Constructor
    NPENSimulation(std::shared_ptr<TetrahedralMesh> mesh, double dt, 
                   double D1, double D2, double D3, 
                   int z1, int z2, 
                   double epsilon, double R, double T, 
                   double L_c, double c0);
    
    // Perform one time step
    void step(const Eigen::VectorXd& c_prev, const Eigen::VectorXd& c3_prev, 
              const Eigen::VectorXd& phi_prev,
              Eigen::VectorXd& c_next, Eigen::VectorXd& c3_next, 
              Eigen::VectorXd& phi_next);
    
    // Perform one time step with electrode boundary conditions
    void step2(const Eigen::VectorXd& c_prev, const Eigen::VectorXd& c3_prev, 
               const Eigen::VectorXd& phi_prev,
               const Eigen::VectorXi& electrode_indices, 
               const Eigen::VectorXd& applied_voltages,
               Eigen::VectorXd& c_next, Eigen::VectorXd& c3_next, 
               Eigen::VectorXd& phi_next,
               double rtol = 1e-3, double atol = 1e-14, int max_iter = 50, double k_reaction = 0.5);
    
    // Getters for physical constants
    double getPhiC() const { return m_phi_c; }
    double getC0() const { return m_c0; }
    
private:
    // Mesh
    std::shared_ptr<TetrahedralMesh> m_mesh;
    
    // Physical constants
    double m_dt;        // Time step
    double m_D1, m_D2, m_D3;  // Diffusion coefficients
    int m_z1, m_z2;     // Ion charges
    double m_epsilon;   // Permittivity
    double m_R, m_T;    // Gas constant, temperature
    double m_F;         // Faraday constant
    double m_L_c;       // Characteristic length
    double m_c0;        // Reference concentration
    double m_phi_c;     // Thermal voltage (R*T/F)
    
    // Precomputed matrices
    Eigen::SparseMatrix<double> m_M_c;     // Mass matrix for c
    Eigen::SparseMatrix<double> m_M_c3;    // Mass matrix for c3
    Eigen::SparseMatrix<double> m_M_phi;   // Mass matrix for phi
    Eigen::SparseMatrix<double> m_K_c;     // Stiffness matrix for c
    Eigen::SparseMatrix<double> m_K_c3;    // Stiffness matrix for c3
    Eigen::SparseMatrix<double> m_K_phi;   // Stiffness matrix for phi
    
    // Precomputed solvers
    Eigen::SparseLU<Eigen::SparseMatrix<double>> m_solver_c;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> m_solver_c3;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> m_solver_phi;
    
    // Initialize matrices and solvers
    void initializeMatrices();
    
    // Assembly methods
    void assembleResidualAndJacobian(
        const Eigen::VectorXd& c, const Eigen::VectorXd& c3, const Eigen::VectorXd& phi,
        const Eigen::VectorXd& c_prev, const Eigen::VectorXd& c3_prev,
        Eigen::VectorXd& residual_c, Eigen::VectorXd& residual_c3, Eigen::VectorXd& residual_phi);

    // Helpers to assemble variable-coefficient matrices for Jacobian construction
    // Weighted stiffness: builds matrix for \int w * (\nabla u \cdot \nabla v) d\Omega
    Eigen::SparseMatrix<double> assembleWeightedStiffness(const Eigen::VectorXd& weight) const;

    // Convection-like matrix for drift linearization:
    // builds matrix with entries A_ij = prefactor * (\nabla phi \cdot \nabla N_i) * \int N_j d\Omega
    // for linear tetrahedra, \int N_j d\Omega = volume / 4
    Eigen::SparseMatrix<double> assembleConvectionMatrix(const Eigen::VectorXd& phi, double prefactor) const;
};

#endif // NPEN_SIMULATION_HPP
