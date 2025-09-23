#include "FluxCalculator.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>

FluxCalculator::FluxCalculator(std::shared_ptr<TetrahedralMesh> mesh, double D1, double z1)
    : m_mesh(mesh), m_D1(D1), m_z1(z1) {
    // Validate inputs
    if (!mesh) {
        throw std::invalid_argument("FluxCalculator: mesh cannot be null");
    }
    
    std::cout << "FluxCalculator initialized with D1=" << m_D1 
              << ", z1=" << m_z1 
              << ", mesh nodes=" << m_mesh->numNodes() 
              << ", elements=" << m_mesh->numElements() << std::endl;
}

void FluxCalculator::computeFlux(const Eigen::VectorXd& c, const Eigen::VectorXd& phi, 
                                 Eigen::MatrixXd& flux_vectors) const {
    const size_t numNodes = m_mesh->numNodes();
    
    // Validate input dimensions
    if (c.size() != static_cast<int>(numNodes) || phi.size() != static_cast<int>(numNodes)) {
        throw std::invalid_argument("FluxCalculator: field dimensions must match mesh nodes");
    }
    
    // Resize output matrix
    flux_vectors.resize(numNodes, 3);
    
    // Compute gradients
    Eigen::MatrixXd grad_c, grad_phi;
    computeGradient(c, grad_c);
    computeGradient(phi, grad_phi);
    
    // Compute flux: J = -D1 ∇c - z1 D1 c ∇phi
    for (size_t i = 0; i < numNodes; ++i) {
        Eigen::Vector3d grad_c_i = grad_c.row(i);
        Eigen::Vector3d grad_phi_i = grad_phi.row(i);
        double c_i = c(i);
        
        // J_i = -D1 * grad_c_i - z1 * D1 * c_i * grad_phi_i
        flux_vectors.row(i) = -m_D1 * grad_c_i - m_z1 * m_D1 * c_i * grad_phi_i;
    }
}

void FluxCalculator::computeFluxHistory(const Eigen::MatrixXd& c_history, 
                                       const Eigen::MatrixXd& phi_history,
                                       std::vector<Eigen::MatrixXd>& flux_tensor) const {
    const size_t numTimeSteps = c_history.rows();
    const size_t numNodes = c_history.cols();
    
    // Validate input dimensions
    if (phi_history.rows() != static_cast<int>(numTimeSteps) || 
        phi_history.cols() != static_cast<int>(numNodes)) {
        throw std::invalid_argument("FluxCalculator: c_history and phi_history dimensions must match");
    }
    
    if (numNodes != m_mesh->numNodes()) {
        throw std::invalid_argument("FluxCalculator: history dimensions must match mesh nodes");
    }
    
    // Resize output tensor
    flux_tensor.clear();
    flux_tensor.reserve(numTimeSteps);
    
    std::cout << "Computing flux for " << numTimeSteps << " time steps..." << std::endl;
    
    // Compute flux for each time step
    for (size_t t = 0; t < numTimeSteps; ++t) {
        Eigen::VectorXd c_t = c_history.row(t);
        Eigen::VectorXd phi_t = phi_history.row(t);
        
        Eigen::MatrixXd flux_t;
        computeFlux(c_t, phi_t, flux_t);
        
        flux_tensor.push_back(flux_t);
        
        // Progress output
        if ((t + 1) % 100 == 0 || t == numTimeSteps - 1) {
            std::cout << "  Processed " << (t + 1) << "/" << numTimeSteps << " time steps" << std::endl;
        }
    }
    
    std::cout << "Flux history computation completed." << std::endl;
}

void FluxCalculator::computeGradient(const Eigen::VectorXd& field, Eigen::MatrixXd& gradient) const {
    const size_t numNodes = m_mesh->numNodes();
    const size_t numElements = m_mesh->numElements();
    
    // Validate input dimensions
    if (field.size() != static_cast<int>(numNodes)) {
        throw std::invalid_argument("FluxCalculator: field dimension must match mesh nodes");
    }
    
    // Compute element-wise gradients (constant per element)
    std::vector<Eigen::Vector3d> element_gradients;
    element_gradients.reserve(numElements);
    
    for (size_t e = 0; e < numElements; ++e) {
        Eigen::Vector3d grad_e = computeElementGradient(field, e);
        element_gradients.push_back(grad_e);
    }
    
    // Average gradients to nodes using volume weighting
    averageGradientsToNodes(element_gradients, gradient);
}

Eigen::Vector3d FluxCalculator::computeElementGradient(const Eigen::VectorXd& field, size_t element_idx) const {
    // Get element data
    const ElementData& elemData = m_mesh->getElementData(element_idx);
    const Eigen::MatrixXi& elements = m_mesh->getElements();
    
    // Get field values at element nodes
    Eigen::Vector4d field_local;
    for (int i = 0; i < 4; ++i) {
        int node_idx = elements(element_idx, i);
        field_local(i) = field(node_idx);
    }
    
    // Compute gradient using precomputed basis function gradients
    // grad(field) = sum_i field_i * grad(N_i)
    Eigen::Vector3d gradient = elemData.grads.transpose() * field_local;
    
    return gradient;
}

void FluxCalculator::averageGradientsToNodes(const std::vector<Eigen::Vector3d>& element_gradients,
                                            Eigen::MatrixXd& node_gradients) const {
    const size_t numNodes = m_mesh->numNodes();
    const size_t numElements = m_mesh->numElements();
    const Eigen::MatrixXi& elements = m_mesh->getElements();
    
    // Initialize output
    node_gradients.resize(numNodes, 3);
    node_gradients.setZero();
    
    // Track volume weights for averaging
    Eigen::VectorXd volume_weights(numNodes);
    volume_weights.setZero();
    
    // Accumulate volume-weighted gradients from elements to nodes
    for (size_t e = 0; e < numElements; ++e) {
        const ElementData& elemData = m_mesh->getElementData(e);
        const double volume = elemData.volume;
        const Eigen::Vector3d& grad_e = element_gradients[e];
        
        // Distribute element gradient to its nodes
        for (int i = 0; i < 4; ++i) {
            int node_idx = elements(e, i);
            
            // Accumulate volume-weighted gradient
            node_gradients.row(node_idx) += volume * grad_e.transpose();
            volume_weights(node_idx) += volume;
        }
    }
    
    // Normalize by total volume weights to get average
    for (size_t n = 0; n < numNodes; ++n) {
        if (volume_weights(n) > 0) {
            node_gradients.row(n) /= volume_weights(n);
        }
    }
}
