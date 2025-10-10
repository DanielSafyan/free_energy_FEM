#ifndef NPEN_SIMULATION_HPP
#define NPEN_SIMULATION_HPP

#include "TetrahedralMesh.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
 #include <string>
 #include <vector>

class NPENSimulation {
public:
    // Constructor
    NPENSimulation(std::shared_ptr<TetrahedralMesh> mesh, double dt, 
                   double D_diff1, double D_mig1, double D_diff2, double D_mig2, double D3, 
                   int z1, int z2, 
                   double epsilon, double R, double T, 
                   double L_c, double c0);
    
    // Perform one time step
    void step(const Eigen::VectorXd& c_prev,
              const Eigen::VectorXd& phi_prev,
              Eigen::VectorXd& c_next,
              Eigen::VectorXd& phi_next);
    
    // Perform one time step with electrode boundary conditions
    void step2(const Eigen::VectorXd& c_prev,
               const Eigen::VectorXd& phi_prev,
               const Eigen::VectorXi& electrode_indices, 
               const Eigen::VectorXd& applied_voltages,
               Eigen::VectorXd& c_next,
               Eigen::VectorXd& phi_next,
               double rtol = 1e-3, double atol = 1e-14, int max_iter = 50, double k_reaction = 0.5);

    // Perform multiple time steps with fixed electrode voltages.
    // Outputs histories with shape (N_nodes, steps), each column is the state after that step.
    void step2_many(const Eigen::VectorXd& c0,
                    const Eigen::VectorXd& phi0,
                    const Eigen::VectorXi& electrode_indices,
                    const Eigen::VectorXd& applied_voltages,
                    int steps,
                    Eigen::MatrixXd& c_hist,
                    Eigen::MatrixXd& phi_hist,
                    double rtol = 1e-3, double atol = 1e-14, int max_iter = 50, double k_reaction = 0.5);
    
    // Getters for physical constants
    double getPhiC() const { return m_phi_c; }
    double getC0() const { return m_c0; }

    // Configuration
    void setAdvectionScheme(const std::string& scheme) { m_useSG = (scheme == "sg" || scheme == "SG" || scheme == "eafe"); }
    // Set electrode surfaces via boundary face index sets, with matching voltages (Volts) and reaction rates (1/s)
    void setElectrodeFaces(const std::vector<std::vector<int>>& face_sets,
                           const std::vector<double>& voltages,
                           const std::vector<double>& k_reaction);
    
private:
    // Mesh
    std::shared_ptr<TetrahedralMesh> m_mesh;
    
    // Physical constants
    double m_dt;        // Time step
    // Transport coefficients (split into diffusion and migration parts)
    double m_D1_diff, m_D1_mig, m_D2_diff, m_D2_mig, m_D3;  
    int m_z1, m_z2;     // Ion charges
    double m_epsilon;   // Permittivity
    double m_R, m_T;    // Gas constant, temperature
    double m_F;         // Faraday constant
    double m_L_c;       // Characteristic length
    double m_c0;        // Reference concentration
    double m_phi_c;     // Thermal voltage (R*T/F)
    
    // Precomputed matrices
    Eigen::SparseMatrix<double> m_M_c;     // Mass matrix for c
    Eigen::SparseMatrix<double> m_M_phi;   // Mass matrix for phi
    Eigen::SparseMatrix<double> m_K_c;     // Stiffness matrix for c
    Eigen::SparseMatrix<double> m_K_phi;   // Stiffness matrix for phi
    
    // Precomputed solvers
    Eigen::SparseLU<Eigen::SparseMatrix<double>> m_solver_c;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> m_solver_phi;
    
    // Initialize matrices and solvers
    void initializeMatrices();

    // Helpers to assemble variable-coefficient matrices for Jacobian construction
    // Weighted stiffness: builds matrix for \int w * (\nabla u \cdot \nabla v) d\Omega
    Eigen::SparseMatrix<double> assembleWeightedStiffness(const Eigen::VectorXd& weight) const;

    // Convection-like matrix for drift linearization:
    // builds matrix with entries A_ij = prefactor * (\nabla phi \cdot \nabla N_i) * \int N_j d\Omega
    // for linear tetrahedra, \int N_j d\Omega = volume / 4
    Eigen::SparseMatrix<double> assembleConvectionMatrix(const Eigen::VectorXd& phi, double prefactor) const;

    // Scharfetter–Gummel/EAFE-like operator for c-equation drift–diffusion (optional)
    Eigen::SparseMatrix<double> assembleSGOperator(const Eigen::VectorXd& phi, double D1_diff_dim, double D1_mig_dim, int z1) const;

    // Electrode surfaces (boundary faces) and boundary mass matrices
    bool m_useSG = true;
    std::vector<std::vector<int>> m_electrodeFaceSets;             // indices into mesh->getBoundaryFaces()
    std::vector<double>          m_electrodeVoltages;              // Volts
    std::vector<double>          m_electrodeK;                     // 1/s
    std::vector<Eigen::SparseMatrix<double>> m_electrodeMboundary; // per-surface boundary mass
};

#endif // NPEN_SIMULATION_HPP
