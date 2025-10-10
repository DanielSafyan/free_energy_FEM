#include "NPENSimulation.hpp"
#include <iostream>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <unsupported/Eigen/SparseExtra>
#include <cmath>

NPENSimulation::NPENSimulation(std::shared_ptr<TetrahedralMesh> mesh, double dt, 
                               double D_diff1, double D_mig1, double D_diff2, double D_mig2, double D3, 
                               int z1, int z2, 
                               double epsilon, double R, double T, 
                               double L_c, double c0)
    : m_mesh(mesh), m_dt(dt), 
      m_D1_diff(D_diff1), m_D1_mig(D_mig1), m_D2_diff(D_diff2), m_D2_mig(D_mig2), m_D3(D3), 
      m_z1(z1), m_z2(z2), m_epsilon(epsilon), m_R(R), m_T(T), 
      m_L_c(L_c), m_c0(c0) {
    // Compute derived constants
    m_F = 96485.33212;  // Faraday constant C/mol
    m_phi_c = m_R * m_T / m_F;  // Thermal voltage
    
    // Initialize matrices
    initializeMatrices();
}

// Store electrode boundary faces and their parameters; preserve cached boundary mass matrices
// unless the face sets actually change. This avoids expensive reassembly every step.
void NPENSimulation::setElectrodeFaces(const std::vector<std::vector<int>>& face_sets,
                                       const std::vector<double>& voltages,
                                       const std::vector<double>& k_reaction) {
    bool changed = (m_electrodeFaceSets.size() != face_sets.size());
    if (!changed) {
        for (size_t i = 0; i < face_sets.size(); ++i) {
            if (m_electrodeFaceSets[i].size() != face_sets[i].size()) { changed = true; break; }
            for (size_t j = 0; j < face_sets[i].size(); ++j) {
                if (m_electrodeFaceSets[i][j] != face_sets[i][j]) { changed = true; break; }
            }
            if (changed) break;
        }
    }
    m_electrodeFaceSets = face_sets;
    m_electrodeVoltages = voltages;
    m_electrodeK = k_reaction;
    if (changed) {
        m_electrodeMboundary.clear();
    }
}

// Bernoulli function used in SG discretization
static inline double bernoulli(double x) {
    const double tol = 1e-8;
    if (std::abs(x) < tol) return 1.0 - 0.5 * x;
    double ex = std::exp(x);
    return x / (ex - 1.0);
}

Eigen::SparseMatrix<double> NPENSimulation::assembleSGOperator(const Eigen::VectorXd& phi, double D1_diff_dim, double D1_mig_dim, int z1) const {
    // Edge-based exponential fitting over linear tetrahedra.
    // For each element, for each of its 6 edges (i,j):
    //   tau_ij = -\int_K grad N_i · grad N_j dK  (positive)
    //   Pe_ij  = (D1_mig_dim/D1_diff_dim) * z1 * (phi_j - phi_i)
    //   Contrib:
    //     A_ii += D1_diff_dim * tau_ij * B(Pe_ij)
    //     A_ij -= D1_diff_dim * tau_ij * B(Pe_ij)
    //     A_jj += D1_diff_dim * tau_ij * B(-Pe_ij)
    //     A_ji -= D1_diff_dim * tau_ij * B(-Pe_ij)
    const size_t N = m_mesh->numNodes();
    const size_t E = m_mesh->numElements();
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(E * 6 * 8); // 6 edges, 8 triplets (ii,ij,jj,ji)

    const Eigen::MatrixXi& elems = m_mesh->getElements();
    for (size_t e = 0; e < E; ++e) {
        const ElementData& ed = m_mesh->getElementData(e);
        // Local stiffness (unit coefficient): Kloc(i,j) = vol * gradNi·gradNj
        Eigen::Matrix4d Kloc = Eigen::Matrix4d::Zero();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                Kloc(i,j) = ed.volume * ed.grads.row(i).dot(ed.grads.row(j));
            }
        }
        int n[4] = { elems(e,0), elems(e,1), elems(e,2), elems(e,3) };
        // All 6 edges of the tetrahedron
        const int edgePairs[6][2] = { {0,1},{0,2},{0,3},{1,2},{1,3},{2,3} };
        for (int ee = 0; ee < 6; ++ee) {
            int li = edgePairs[ee][0];
            int lj = edgePairs[ee][1];
            int gi = n[li];
            int gj = n[lj];
            double tau_ij = -Kloc(li, lj); // positive
            if (tau_ij <= 0) continue;
            double dphi = phi(gj) - phi(gi);
            double Pe = (D1_mig_dim / std::max(D1_diff_dim, 1e-15)) * static_cast<double>(z1) * dphi;
            double Bij = bernoulli(Pe);
            double Bji = bernoulli(-Pe);
            double w = D1_diff_dim * tau_ij;
            // ii, ij
            trips.emplace_back(gi, gi, w * Bij);
            trips.emplace_back(gi, gj, -w * Bij);
            // jj, ji
            trips.emplace_back(gj, gj, w * Bji);
            trips.emplace_back(gj, gi, -w * Bji);
        }
    }

    Eigen::SparseMatrix<double> A(N, N);
    A.setFromTriplets(trips.begin(), trips.end());
    A.makeCompressed();
    return A;
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
    m_M_phi.resize(numNodes, numNodes);
    m_K_c.resize(numNodes, numNodes);
    m_K_phi.resize(numNodes, numNodes);
    
    // Assemble mass matrices
    m_mesh->assembleMassMatrix(m_M_c, 1.0);
    m_mesh->assembleMassMatrix(m_M_phi, 1.0);
    
    // Assemble stiffness matrices
    m_mesh->assembleStiffnessMatrix(m_K_c, 1.0);
    m_mesh->assembleStiffnessMatrix(m_K_phi, 1.0);  // Use unit coefficient for phi stiffness
    
    // For NPEN, we don't precompute solvers since the system is coupled
    // Solvers will be computed on-demand in the step functions
}

void NPENSimulation::step(const Eigen::VectorXd& c_prev,
                          const Eigen::VectorXd& phi_prev,
                          Eigen::VectorXd& c_next,
                          Eigen::VectorXd& phi_next) {
    const int N = static_cast<int>(m_mesh->numNodes());
    // Start from previous state
    Eigen::VectorXd c = c_prev;
    // no neutral species in reduced NPEN
    Eigen::VectorXd phi = phi_prev;

    // Dimensionless parameters
    const double D_c = std::max({m_D1_diff, m_D2_diff, m_D1_mig, m_D2_mig, m_D3});
    const double D1_diff_dim = m_D1_diff / D_c;
    const double D1_mig_dim  = m_D1_mig  / D_c;
    const double D2_diff_dim = m_D2_diff / D_c;
    const double D2_mig_dim  = m_D2_mig  / D_c;
    const double D3_dim = m_D3 / D_c; // kept for completeness but unused
    const double dt_dim = m_dt * D_c / (m_L_c * m_L_c);

    const int max_iter = 25;
    const double rtol = 1e-6, atol = 1e-12;
    double initial_residual_norm = -1.0;

    for (int it = 0; it < max_iter; ++it) {
        // Build Jacobian blocks
        // c-equation operator: SG if enabled, else Galerkin diffusion + linearized drift
        Eigen::SparseMatrix<double> A_c;
        if (m_useSG) {
            A_c = assembleSGOperator(phi, D1_diff_dim, D1_mig_dim, m_z1);
        } else {
            Eigen::SparseMatrix<double> J_cc_drift = assembleConvectionMatrix(phi, D1_mig_dim * m_z1);
            A_c = D1_diff_dim * m_K_c + J_cc_drift;
        }
        // Weighted stiffness blocks treating c as coefficient (migration term)
        Eigen::VectorXd w_cphi = (D1_mig_dim * m_z1) * c;
        Eigen::SparseMatrix<double> K_c_phi = assembleWeightedStiffness(w_cphi);
        Eigen::VectorXd w_phiphi = (D1_mig_dim + D2_mig_dim) * c;
        Eigen::SparseMatrix<double> K_phi_phi = assembleWeightedStiffness(w_phiphi);

        Eigen::SparseMatrix<double> J11 = (1.0 / dt_dim) * m_M_c + A_c;
        Eigen::SparseMatrix<double> J13 = K_c_phi;
        Eigen::SparseMatrix<double> J31 = -(D1_diff_dim - D2_diff_dim) * m_K_c;
        Eigen::SparseMatrix<double> J33 = K_phi_phi;

        // Residuals
        Eigen::VectorXd R_c   = (1.0 / dt_dim) * (m_M_c * (c  - c_prev)) + A_c * c + K_c_phi * phi;
        Eigen::VectorXd R_phi = K_phi_phi * phi + (-(D1_diff_dim - D2_diff_dim)) * (m_K_c * c);

        // Assemble big Jacobian via triplets (order: [c, phi])
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(J11.nonZeros() + J13.nonZeros() + J31.nonZeros() + J33.nonZeros());
        auto appendBlock = [&](const Eigen::SparseMatrix<double>& A, int r0, int c0) {
            for (int k = 0; k < A.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
                    trips.emplace_back(r0 + it.row(), c0 + it.col(), it.value());
                }
            }
        };
        appendBlock(J11, 0, 0);
        appendBlock(J13, 0, N);
        appendBlock(J31, N, 0);
        appendBlock(J33, N, N);

        // Stack residuals
        Eigen::VectorXd R(2 * N);
        R.segment(0, N) = R_c;
        R.segment(N, N) = R_phi;

        // Apply a Dirichlet anchor for phi at node 0: phi(0) = 0 (to fix gauge)
        const int anchorRow = N + 0;
        std::vector<Eigen::Triplet<double>> tripsFiltered;
        tripsFiltered.reserve(trips.size());
        for (const auto& t : trips) {
            if (t.row() != anchorRow) tripsFiltered.push_back(t);
        }
        tripsFiltered.emplace_back(anchorRow, anchorRow, 1.0);
        R(anchorRow) = phi(0) - 0.0;

        Eigen::SparseMatrix<double> J(2 * N, 2 * N);
        J.setFromTriplets(tripsFiltered.begin(), tripsFiltered.end());
        J.makeCompressed();

        // Solve J * delta = -R
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.analyzePattern(J);
        solver.factorize(J);
        if (solver.info() != Eigen::Success) { c_next = c_prev; phi_next = phi_prev; return; }
        Eigen::VectorXd delta = solver.solve(-R);

        // Update (simple under-relaxation for phi if desired)
        const double alpha = 1.0;
        const double alpha_phi = 1.0;
        c   += alpha     * delta.segment(0, N);
        phi += alpha_phi * delta.segment(N, N);

        // Convergence check on residual norm
        double nrm = R.norm();
        if (it == 0) initial_residual_norm = (nrm > 0 ? nrm : 1.0);
        if (nrm < (initial_residual_norm * rtol) + atol) break;
    }

    c_next = c;
    phi_next = phi;
}

void NPENSimulation::step2_many(const Eigen::VectorXd& c0,
                                const Eigen::VectorXd& phi0,
                                const Eigen::VectorXi& electrode_indices,
                                const Eigen::VectorXd& applied_voltages,
                                int steps,
                                Eigen::MatrixXd& c_hist,
                                Eigen::MatrixXd& phi_hist,
                                double rtol, double atol, int max_iter, double k_reaction) {
    const int N = static_cast<int>(m_mesh->numNodes());
    if (steps <= 0) {
        c_hist.resize(N, 0);
        phi_hist.resize(N, 0);
        return;
    }
    c_hist.resize(N, steps);
    phi_hist.resize(N, steps);
    Eigen::VectorXd c_prev = c0;
    Eigen::VectorXd phi_prev = phi0;
    Eigen::VectorXd c_next(N), phi_next(N);
    for (int s = 0; s < steps; ++s) {
        step2(c_prev, phi_prev, electrode_indices, applied_voltages,
              c_next, phi_next, rtol, atol, max_iter, k_reaction);
        c_hist.col(s) = c_next;
        phi_hist.col(s) = phi_next;
        c_prev.swap(c_next);
        phi_prev.swap(phi_next);
    }
}

void NPENSimulation::step2(const Eigen::VectorXd& c_prev,
                           const Eigen::VectorXd& phi_prev,
                           const Eigen::VectorXi& electrode_indices, 
                           const Eigen::VectorXd& applied_voltages,
                           Eigen::VectorXd& c_next,
                           Eigen::VectorXd& phi_next,
                           double rtol, double atol, int max_iter, double k_reaction) {
    const int N = static_cast<int>(m_mesh->numNodes());

    // Initialize from previous state
    Eigen::VectorXd c = c_prev;
    Eigen::VectorXd phi = phi_prev;

    // Dimensionless parameters
    const double D_c = std::max({m_D1_diff, m_D2_diff, m_D1_mig, m_D2_mig, m_D3});
    const double D1_diff_dim = m_D1_diff / D_c;
    const double D1_mig_dim  = m_D1_mig  / D_c;
    const double D2_diff_dim = m_D2_diff / D_c;
    const double D2_mig_dim  = m_D2_mig  / D_c;
    const double D3_dim = m_D3 / D_c;
    const double dt_dim = m_dt * D_c / (m_L_c * m_L_c);

    // First-order reaction coefficient (dimensionless): surface Robin n·J + k c = 0
    // Weak form contribution: add (-k_dimless) * M_boundary to J11 and residual
    const double k_dimless = k_reaction * m_L_c / D_c;
    const double k_reac_diag = -k_dimless;

    double initial_residual_norm = -1.0;
    for (int it = 0; it < max_iter; ++it) {
        // Jacobian blocks for c-equation
        Eigen::SparseMatrix<double> A_c;
        if (m_useSG) {
            A_c = assembleSGOperator(phi, D1_diff_dim, D1_mig_dim, m_z1);
        } else {
            // fallback: Galerkin diffusion + linearized drift
            Eigen::SparseMatrix<double> J_cc_drift = assembleConvectionMatrix(phi, D1_mig_dim * m_z1);
            A_c = D1_diff_dim * m_K_c + J_cc_drift;
        }
        // Coupling c-phi (retain existing linearization for phi blocks)
        Eigen::VectorXd w_cphi = (D1_mig_dim * m_z1) * c;
        Eigen::SparseMatrix<double> K_c_phi = assembleWeightedStiffness(w_cphi);
        Eigen::VectorXd w_phiphi = (D1_mig_dim + D2_mig_dim) * c;
        Eigen::SparseMatrix<double> K_phi_phi = assembleWeightedStiffness(w_phiphi);

        Eigen::SparseMatrix<double> J11 = (1.0 / dt_dim) * m_M_c + A_c;
        Eigen::SparseMatrix<double> J13 = K_c_phi;
        Eigen::SparseMatrix<double> J31 = -(D1_diff_dim - D2_diff_dim) * m_K_c;
        Eigen::SparseMatrix<double> J33 = K_phi_phi;

        // Residuals
        Eigen::VectorXd R_c   = (1.0 / dt_dim) * (m_M_c * (c  - c_prev)) + A_c * c + K_c_phi * phi;
        Eigen::VectorXd R_phi = K_phi_phi * phi + (-(D1_diff_dim - D2_diff_dim)) * (m_K_c * c);

        // Build phi Dirichlet set from configured electrode faces if present; else use passed-in indices
        std::vector<std::pair<int,double>> phi_dirichlet; // (node, voltage_dim)
        std::vector<Eigen::SparseMatrix<double>> M_gamma_list; // boundary mass per electrode
        if (!m_electrodeFaceSets.empty()) {
            // Collect unique nodes on each face set and their voltages
            const Eigen::MatrixXi& bfaces = m_mesh->getBoundaryFaces();
            std::vector<char> seen(N, 0);
            for (size_t s = 0; s < m_electrodeFaceSets.size(); ++s) {
                double V = (s < m_electrodeVoltages.size() ? m_electrodeVoltages[s] : 0.0);
                if (std::isnan(V)) continue;
                double Vdim = V / m_phi_c;
                // Ensure boundary mass for this surface exists
                if (s >= m_electrodeMboundary.size()) m_electrodeMboundary.resize(s+1);
                if (m_electrodeMboundary[s].rows() == 0) {
                    m_mesh->assembleBoundaryMassMatrix(m_electrodeMboundary[s], m_electrodeFaceSets[s], 1.0);
                }
                M_gamma_list.push_back(m_electrodeMboundary[s]);
                for (int fid : m_electrodeFaceSets[s]) {
                    if (fid < 0 || fid >= bfaces.rows()) continue;
                    int i = bfaces(fid,0), j = bfaces(fid,1), k = bfaces(fid,2);
                    if (!seen[i]) { phi_dirichlet.emplace_back(i, Vdim); seen[i] = 1; }
                    if (!seen[j]) { phi_dirichlet.emplace_back(j, Vdim); seen[j] = 1; }
                    if (!seen[k]) { phi_dirichlet.emplace_back(k, Vdim); seen[k] = 1; }
                }
            }
        } else {
            phi_dirichlet.reserve(electrode_indices.size());
            for (int i = 0; i < electrode_indices.size(); ++i) {
                int node = electrode_indices(i);
                if (node < 0 || node >= N) continue;
                if (i >= applied_voltages.size()) continue;
                double V = applied_voltages(i);
                if (std::isnan(V)) continue;
                phi_dirichlet.emplace_back(node, V / m_phi_c);
            }
        }

        // Apply Robin boundary on c using boundary mass matrices for configured electrode faces
        if (!M_gamma_list.empty()) {
            for (size_t s = 0; s < M_gamma_list.size(); ++s) {
                if (s >= m_electrodeK.size()) continue;
                double k_s_dim = m_electrodeK[s] * m_L_c / D_c;
                double coeff = -k_s_dim;
                const auto& Mgamma = M_gamma_list[s];
                // R_c += coeff * Mgamma * c ; J11 += coeff * Mgamma
                R_c += coeff * (Mgamma * c);
                J11 += coeff * Mgamma;
            }
        } else {
            // Backward compatibility: if caller passed k_reaction and node indices, apply as nodal Robin (lumped boundary)
            if (electrode_indices.size() > 0 && std::abs(k_reaction) > 0) {
                for (int i = 0; i < electrode_indices.size(); ++i) {
                    int node = electrode_indices(i);
                    if (node < 0 || node >= N) continue;
                    R_c(node) += k_reac_diag * c(node);
                    J11.coeffRef(node, node) += k_reac_diag; // safe for compressed? We'll add via triplets below instead if needed
                }
            }
        }

        // Assemble global Jacobian
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(J11.nonZeros() + J13.nonZeros() + J31.nonZeros() + J33.nonZeros() + static_cast<int>(phi_dirichlet.size()));
        auto appendBlock = [&](const Eigen::SparseMatrix<double>& A, int r0, int c0) {
            for (int k = 0; k < A.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
                    trips.emplace_back(r0 + it.row(), c0 + it.col(), it.value());
                }
            }
        };
        appendBlock(J11, 0, 0);
        appendBlock(J13, 0, N);
        appendBlock(J31, N, 0);
        appendBlock(J33, N, N);
        // Note: reaction contributions (Robin) already added directly to J11 above when using faces; nothing to add here.

        // Stack residuals
        Eigen::VectorXd R(2 * N);
        R.segment(0, N) = R_c;
        R.segment(N, N) = R_phi;

        // Strong Dirichlet for phi at electrodes
        std::vector<Eigen::Triplet<double>> tripsFiltered;
        tripsFiltered.reserve(trips.size());
        std::vector<int> enforced_rows; enforced_rows.reserve(phi_dirichlet.size());
        for (const auto& nd : phi_dirichlet) enforced_rows.push_back(N + nd.first);
        for (const auto& t : trips) {
            bool drop = false;
            for (int r : enforced_rows) { if (t.row() == r) { drop = true; break; } }
            if (!drop) tripsFiltered.push_back(t);
        }
        for (const auto& nd : phi_dirichlet) {
            int row = N + nd.first;
            tripsFiltered.emplace_back(row, row, 1.0);
            R(row) = phi(nd.first) - nd.second;
        }
        // If no electrodes given, anchor gauge at phi(0)
        if (phi_dirichlet.empty() && N > 0) {
            const int anchorRow = N + 0;
            std::vector<Eigen::Triplet<double>> tmp; tmp.reserve(tripsFiltered.size());
            for (const auto& t : tripsFiltered) if (t.row() != anchorRow) tmp.push_back(t);
            tmp.emplace_back(anchorRow, anchorRow, 1.0);
            tripsFiltered.swap(tmp);
            R(anchorRow) = phi(0);
        }

        Eigen::SparseMatrix<double> J(2 * N, 2 * N);
        J.setFromTriplets(tripsFiltered.begin(), tripsFiltered.end());
        J.makeCompressed();

        // Solve J * delta = -R
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.analyzePattern(J);
        solver.factorize(J);
        if (solver.info() != Eigen::Success) { c_next = c_prev; phi_next = phi_prev; return; }
        Eigen::VectorXd delta = solver.solve(-R);

        // Update
        const double alpha = 1.0, alpha_phi = 1.0;
        c   += alpha     * delta.segment(0, N);
        phi += alpha_phi * delta.segment(N, N);

        // Convergence
        double nrm = R.norm();
        if (it == 0) initial_residual_norm = (nrm > 0 ? nrm : 1.0);
        if (nrm < (initial_residual_norm * rtol) + atol) break;
    }
    // After iterations, write results
    c_next = c;
    phi_next = phi;
}
