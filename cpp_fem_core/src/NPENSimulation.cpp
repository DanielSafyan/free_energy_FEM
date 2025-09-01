#include "NPENSimulation.hpp"
#include <iostream>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <unsupported/Eigen/SparseExtra>

NPENSimulation::NPENSimulation(std::shared_ptr<TetrahedralMesh> mesh, double dt, 
                               double D1, double D2, double D3, 
                               int z1, int z2, 
                               double epsilon, double R, double T, 
                               double L_c, double c0)
    : m_mesh(mesh), m_dt(dt), m_D1(D1), m_D2(D2), m_D3(D3), 
      m_z1(z1), m_z2(z2), m_epsilon(epsilon), m_R(R), m_T(T), 
      m_L_c(L_c), m_c0(c0) {
    // Compute derived constants
    m_F = 96485.33212;  // Faraday constant C/mol
    m_phi_c = m_R * m_T / m_F;  // Thermal voltage
    
    // Initialize matrices
    initializeMatrices();
}

Eigen::SparseMatrix<double> NPENSimulation::assembleWeightedStiffness(const Eigen::VectorXd& weight) const {
    const size_t numNodes = m_mesh->numNodes();
    const size_t numElements = m_mesh->numElements();
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(numElements * 16);

    const Eigen::MatrixXi& elems = m_mesh->getElements();
    for (size_t e = 0; e < numElements; ++e) {
        const ElementData& ed = m_mesh->getElementData(e);
        const double vol = ed.volume;

        int n0 = elems(e, 0), n1 = elems(e, 1), n2 = elems(e, 2), n3 = elems(e, 3);
        double w_avg = 0.25 * (weight(n0) + weight(n1) + weight(n2) + weight(n3));

        // Local 4x4 contribution: w_avg * (gradNi · gradNj) * vol
        for (int i = 0; i < 4; ++i) {
            int ni = (i == 0 ? n0 : i == 1 ? n1 : i == 2 ? n2 : n3);
            const Eigen::RowVector3d gi = ed.grads.row(i);
            for (int j = 0; j < 4; ++j) {
                int nj = (j == 0 ? n0 : j == 1 ? n1 : j == 2 ? n2 : n3);
                const Eigen::RowVector3d gj = ed.grads.row(j);
                double val = w_avg * gi.dot(gj) * vol;
                trips.emplace_back(ni, nj, val);
            }
        }
    }

    Eigen::SparseMatrix<double> K(numNodes, numNodes);
    K.setFromTriplets(trips.begin(), trips.end());
    K.makeCompressed();
    return K;
}

Eigen::SparseMatrix<double> NPENSimulation::assembleConvectionMatrix(const Eigen::VectorXd& phi, double prefactor) const {
    const size_t numNodes = m_mesh->numNodes();
    const size_t numElements = m_mesh->numElements();
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(numElements * 16);

    const Eigen::MatrixXi& elems = m_mesh->getElements();
    for (size_t e = 0; e < numElements; ++e) {
        const ElementData& ed = m_mesh->getElementData(e);
        const double vol = ed.volume;
        int n0 = elems(e, 0), n1 = elems(e, 1), n2 = elems(e, 2), n3 = elems(e, 3);

        // phi values on element
        Eigen::Vector4d phi_loc;
        phi_loc << phi(n0), phi(n1), phi(n2), phi(n3);
        // grad(phi) is constant per linear tetra
        Eigen::Vector3d grad_phi = ed.grads.transpose() * phi_loc;

        // Entry A_ij = prefactor * (grad_phi · grad N_i) * ∫ N_j dΩ, with ∫ N_j dΩ = vol/4
        const double intNj = vol * 0.25;
        for (int i = 0; i < 4; ++i) {
            int ni = (i == 0 ? n0 : i == 1 ? n1 : i == 2 ? n2 : n3);
            double rowFactor = prefactor * ed.grads.row(i).dot(grad_phi) * intNj;
            // same value distributes to all j for this element
            trips.emplace_back(ni, n0, rowFactor);
            trips.emplace_back(ni, n1, rowFactor);
            trips.emplace_back(ni, n2, rowFactor);
            trips.emplace_back(ni, n3, rowFactor);
        }
    }

    Eigen::SparseMatrix<double> C(numNodes, numNodes);
    C.setFromTriplets(trips.begin(), trips.end());
    C.makeCompressed();
    return C;
}

void NPENSimulation::initializeMatrices() {
    size_t numNodes = m_mesh->numNodes();
    
    // Initialize sparse matrices
    m_M_c.resize(numNodes, numNodes);
    m_M_c3.resize(numNodes, numNodes);
    m_M_phi.resize(numNodes, numNodes);
    m_K_c.resize(numNodes, numNodes);
    m_K_c3.resize(numNodes, numNodes);
    m_K_phi.resize(numNodes, numNodes);
    
    // Assemble mass matrices
    m_mesh->assembleMassMatrix(m_M_c, 1.0);
    m_mesh->assembleMassMatrix(m_M_c3, 1.0);
    m_mesh->assembleMassMatrix(m_M_phi, 1.0);
    
    // Assemble stiffness matrices
    m_mesh->assembleStiffnessMatrix(m_K_c, 1.0);
    m_mesh->assembleStiffnessMatrix(m_K_c3, 1.0);
    m_mesh->assembleStiffnessMatrix(m_K_phi, 1.0);  // Use unit coefficient for phi stiffness
    
    // For NPEN, we don't precompute solvers since the system is coupled
    // Solvers will be computed on-demand in the step functions
}

void NPENSimulation::step(const Eigen::VectorXd& c_prev, const Eigen::VectorXd& c3_prev,
                          const Eigen::VectorXd& phi_prev,
                          Eigen::VectorXd& c_next, Eigen::VectorXd& c3_next,
                          Eigen::VectorXd& phi_next) {
    const int N = static_cast<int>(m_mesh->numNodes());
    // Start from previous state
    Eigen::VectorXd c = c_prev;
    Eigen::VectorXd c3 = c3_prev;
    Eigen::VectorXd phi = phi_prev;

    // Dimensionless parameters
    const double D_c = std::max({m_D1, m_D2, m_D3});
    const double D1_dim = m_D1 / D_c;
    const double D2_dim = m_D2 / D_c;
    const double D3_dim = m_D3 / D_c;
    const double dt_dim = m_dt * D_c / (m_L_c * m_L_c);

    const int max_iter = 25;
    const double rtol = 1e-6, atol = 1e-12;
    double initial_residual_norm = -1.0;

    for (int it = 0; it < max_iter; ++it) {
        // Build Jacobian blocks
        Eigen::SparseMatrix<double> J_cc_drift = assembleConvectionMatrix(phi, D1_dim * m_z1);
        // Weighted stiffness blocks treating c as coefficient
        Eigen::VectorXd w_cphi = (D1_dim * m_z1) * c;
        Eigen::SparseMatrix<double> K_c_phi = assembleWeightedStiffness(w_cphi);
        Eigen::VectorXd w_phiphi = (D1_dim + D2_dim) * c;
        Eigen::SparseMatrix<double> K_phi_phi = assembleWeightedStiffness(w_phiphi);

        Eigen::SparseMatrix<double> J11 = (1.0 / dt_dim) * m_M_c + D1_dim * m_K_c + J_cc_drift;
        Eigen::SparseMatrix<double> J13 = K_c_phi;
        Eigen::SparseMatrix<double> J22 = (1.0 / dt_dim) * m_M_c + D3_dim * m_K_c;
        Eigen::SparseMatrix<double> J31 = -(D1_dim - D2_dim) * m_K_c;
        Eigen::SparseMatrix<double> J33 = K_phi_phi;

        // Residuals
        Eigen::VectorXd R_c   = (1.0 / dt_dim) * (m_M_c * (c  - c_prev)) + D1_dim * (m_K_c * c)  + K_c_phi   * phi;
        Eigen::VectorXd R_c3  = (1.0 / dt_dim) * (m_M_c * (c3 - c3_prev)) + D3_dim * (m_K_c * c3);
        Eigen::VectorXd R_phi = K_phi_phi * phi + (-(D1_dim - D2_dim)) * (m_K_c * c);

        // Assemble big Jacobian via triplets
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(J11.nonZeros() + J13.nonZeros() + J22.nonZeros() + J31.nonZeros() + J33.nonZeros());
        auto appendBlock = [&](const Eigen::SparseMatrix<double>& A, int r0, int c0) {
            for (int k = 0; k < A.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
                    trips.emplace_back(r0 + it.row(), c0 + it.col(), it.value());
                }
            }
        };
        appendBlock(J11, 0, 0);
        appendBlock(J13, 0, N);
        appendBlock(J22, N, N);
        appendBlock(J31, 2 * N, 0);
        appendBlock(J33, 2 * N, 2 * N);

        // Stack residuals
        Eigen::VectorXd R(3 * N);
        R.segment(0, N) = R_c;
        R.segment(N, N) = R_c3;
        R.segment(2 * N, N) = R_phi;

        // Apply a Dirichlet anchor for phi at node 0: phi(0) = 0 (to fix gauge)
        const int anchorRow = 2 * N + 0;
        std::vector<Eigen::Triplet<double>> tripsFiltered;
        tripsFiltered.reserve(trips.size());
        for (const auto& t : trips) {
            if (t.row() != anchorRow) tripsFiltered.push_back(t);
        }
        tripsFiltered.emplace_back(anchorRow, anchorRow, 1.0);
        R(anchorRow) = phi(0) - 0.0;

        Eigen::SparseMatrix<double> J(3 * N, 3 * N);
        J.setFromTriplets(tripsFiltered.begin(), tripsFiltered.end());
        J.makeCompressed();

        // Solve J * delta = -R
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.analyzePattern(J);
        solver.factorize(J);
        if (solver.info() != Eigen::Success) {
            // Fallback: leave state unchanged
            c_next = c_prev; c3_next = c3_prev; phi_next = phi_prev;
            return;
        }
        Eigen::VectorXd delta = solver.solve(-R);

        // Update (simple under-relaxation for phi if desired)
        const double alpha = 1.0;
        const double alpha_phi = 1.0;
        c   += alpha     * delta.segment(0, N);
        c3  += alpha     * delta.segment(N, N);
        phi += alpha_phi * delta.segment(2 * N, N);

        // Convergence check on residual norm
        double nrm = R.norm();
        if (it == 0) initial_residual_norm = (nrm > 0 ? nrm : 1.0);
        if (nrm < (initial_residual_norm * rtol) + atol) break;
    }

    c_next = c;
    c3_next = c3;
    phi_next = phi;
}

void NPENSimulation::step2(const Eigen::VectorXd& c_prev, const Eigen::VectorXd& c3_prev, 
                            const Eigen::VectorXd& phi_prev,
                            const Eigen::VectorXi& electrode_indices, 
                            const Eigen::VectorXd& applied_voltages,
                           Eigen::VectorXd& c_next, Eigen::VectorXd& c3_next, 
                           Eigen::VectorXd& phi_next,
                           double rtol, double atol, int max_iter) {
    size_t numNodes = m_mesh->numNodes();
    
    // Initialize solution vectors
    c_next = c_prev;
    c3_next = c3_prev;
    phi_next = phi_prev;
    
    // Reaction rate constant (match Python default)
    double k_reaction = 0.5;
    
    double initial_residual_norm = -1.0;
    
    for (int it = 0; it < max_iter; ++it) {
        // Assemble residuals for each variable
        Eigen::VectorXd residual_c = Eigen::VectorXd::Zero(numNodes);
        Eigen::VectorXd residual_c3 = Eigen::VectorXd::Zero(numNodes);
        Eigen::VectorXd residual_phi = Eigen::VectorXd::Zero(numNodes);
        
        assembleResidualAndJacobian(c_next, c3_next, phi_next, c_prev, c3_prev,
                                    residual_c, residual_c3, residual_phi);
        
        // Apply Dirichlet BCs and first-order reaction terms at specified electrode nodes
        for (int i = 0; i < electrode_indices.size(); ++i) {
            int node_idx = electrode_indices(i);
            if (node_idx < 0 || node_idx >= (int)numNodes) continue;  // Skip invalid indices
            
            // Check if we have a corresponding voltage value
            if (i >= applied_voltages.size()) continue;
            
            // Skip if voltage is NaN
            if (std::isnan(applied_voltages(i))) continue;
            
            // Apply Dirichlet BC: phi = applied_voltage
            // Convert applied voltage to dimensionless form
            double voltage_dim = applied_voltages(i) / m_phi_c;
            
            // Set residual for phi at this node to enforce BC
            residual_phi(node_idx) = phi_next(node_idx) - voltage_dim;
            
            // Apply first-order reaction boundary condition on c at the same node
            // J_reaction = -k * c, added to the residual
            double k_reaction_dimless = -k_reaction * m_L_c / std::max({m_D1, m_D2, m_D3});
            residual_c(node_idx) += k_reaction_dimless * c_next(node_idx);
        }
        
        // Combine residuals into single vector
        Eigen::VectorXd residual = Eigen::VectorXd::Zero(3 * numNodes);
        residual.segment(0, numNodes) = residual_c;
        residual.segment(numNodes, numNodes) = residual_c3;
        residual.segment(2 * numNodes, numNodes) = residual_phi;
        
        // Compute residual norm
        double nrm = residual.norm();
        if (it == 0) {
            initial_residual_norm = (nrm > 0) ? nrm : 1.0;
        }
        
        // Check for convergence
        if (nrm < (initial_residual_norm * rtol) + atol) {
            break;
        }
        
        // For now, we'll use a simplified update approach
        // In a full implementation, we would solve the coupled system
        // But we'll improve the decoupled approach with better scaling
        
        // Non-dimensional parameters for proper scaling
        double D_c = std::max({m_D1, m_D2, m_D3});
        double dt_dim = m_dt * D_c / (m_L_c * m_L_c);
        
        // Improved decoupled updates with proper scaling
        for (size_t i = 0; i < numNodes; ++i) {
            // Update c with under-relaxation and proper scaling
            if (std::abs(residual_c(i)) < 1e10) {  // Avoid updating if residual is too large
                c_next(i) -= 0.5 * residual_c(i) * dt_dim;
            }
            
            // Update c3 with under-relaxation and proper scaling
            if (std::abs(residual_c3(i)) < 1e10) {  // Avoid updating if residual is too large
                c3_next(i) -= 0.5 * residual_c3(i) * dt_dim;
            }
            
            // Update phi with under-relaxation
            if (std::abs(residual_phi(i)) < 1e10) {  // Avoid updating if residual is too large
                phi_next(i) -= 0.5 * residual_phi(i);
            }
        }
        
        // Apply hard limits to prevent divergence
        for (size_t i = 0; i < numNodes; ++i) {
            // Limit c to reasonable range
            if (c_next(i) < 0.0) c_next(i) = 0.0;
            if (c_next(i) > 2.0) c_next(i) = 2.0;
            
            // Limit c3 to reasonable range
            if (c3_next(i) < 0.0) c3_next(i) = 0.0;
            if (c3_next(i) > 2.0) c3_next(i) = 2.0;
            
            // Limit phi to reasonable range
            if (phi_next(i) < -10.0) phi_next(i) = -10.0;
            if (phi_next(i) > 10.0) phi_next(i) = 10.0;
        }
    }
}

void NPENSimulation::assembleResidualAndJacobian(
    const Eigen::VectorXd& c, const Eigen::VectorXd& c3, const Eigen::VectorXd& phi,
    const Eigen::VectorXd& c_prev, const Eigen::VectorXd& c3_prev,
    Eigen::VectorXd& residual_c, Eigen::VectorXd& residual_c3, Eigen::VectorXd& residual_phi) {
    
    size_t numNodes = m_mesh->numNodes();
    size_t numElements = m_mesh->numElements();
    
    // Initialize residuals
    residual_c.setZero();
    residual_c3.setZero();
    residual_phi.setZero();
    
    // Non-dimensional parameters (match Python implementation)
    double D1_dim = m_D1 / std::max({m_D1, m_D2, m_D3});
    double D2_dim = m_D2 / std::max({m_D1, m_D2, m_D3});
    double D3_dim = m_D3 / std::max({m_D1, m_D2, m_D3});
    double dt_dim = m_dt * std::max({m_D1, m_D2, m_D3}) / (m_L_c * m_L_c);
    
    for (size_t i = 0; i < numElements; ++i) {
        const ElementData& elemData = m_mesh->getElementData(i);
        double volume = elemData.volume;
        
        // Get node indices for this element
        Eigen::Vector4i elemNodes;
        elemNodes << m_mesh->getElements()(i, 0), 
                     m_mesh->getElements()(i, 1), 
                     m_mesh->getElements()(i, 2), 
                     m_mesh->getElements()(i, 3);
        
        // Get values at nodes
        Eigen::Vector4d c_local, c3_local, phi_local;
        Eigen::Vector4d c_prev_local, c3_prev_local;
        
        for (int j = 0; j < 4; ++j) {
            int nodeIdx = elemNodes(j);
            c_local(j) = c(nodeIdx);
            c3_local(j) = c3(nodeIdx);
            phi_local(j) = phi(nodeIdx);
            c_prev_local(j) = c_prev(nodeIdx);
            c3_prev_local(j) = c3_prev(nodeIdx);
        }
        
        // Compute gradients
        const Eigen::Matrix<double, 4, 3>& grads = elemData.grads;
        Eigen::Vector3d grad_c = grads.transpose() * c_local;
        Eigen::Vector3d grad_c3 = grads.transpose() * c3_local;
        Eigen::Vector3d grad_phi = grads.transpose() * phi_local;
        
        // Compute average values
        double c_avg = c_local.mean();
        
        // Compute residual contributions for NPEN equations
        for (int j = 0; j < 4; ++j) {
            int nodeIdx = elemNodes(j);
            
            // c-equation residual: (c - c_prev)/dt + D1 * Δc + D1 * z1 * c * Δphi = 0
            // Mass term
            residual_c(nodeIdx) += (c_local(j) - c_prev_local(j)) * volume / dt_dim;
            // Diffusion term
            residual_c(nodeIdx) += D1_dim * grad_c.dot(grads.row(j)) * volume;
            // Drift term
            residual_c(nodeIdx) += D1_dim * m_z1 * c_local(j) * grad_phi.dot(grads.row(j)) * volume;
            
            // c3-equation residual: (c3 - c3_prev)/dt + D3 * Δc3 = 0
            // Mass term
            residual_c3(nodeIdx) += (c3_local(j) - c3_prev_local(j)) * volume / dt_dim;
            // Diffusion term
            residual_c3(nodeIdx) += D3_dim * grad_c3.dot(grads.row(j)) * volume;
            
            // phi-equation residual from current conservation:
            // ∇·((D1-D2)∇c + (D1+D2)c∇phi) = 0
            // First term: -(D1-D2) * Δc
            residual_phi(nodeIdx) += -(D1_dim - D2_dim) * grad_c.dot(grads.row(j)) * volume;
            // Second term: (D1+D2) * c * Δphi
            residual_phi(nodeIdx) += (D1_dim + D2_dim) * c_avg * grad_phi.dot(grads.row(j)) * volume;
        }
    }
}
