#include "TetrahedralMesh.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <cmath>

TetrahedralMesh::TetrahedralMesh(const Eigen::MatrixXd& nodes, const Eigen::MatrixXi& elements)
    : m_nodes(nodes), m_elements(elements) {
    precomputeElementData();
}

void TetrahedralMesh::precomputeElementData() {
    size_t numElements = m_elements.rows();
    m_element_data.resize(numElements);
    
    for (size_t i = 0; i < numElements; ++i) {
        // Get the four nodes of the tetrahedron
        Eigen::Vector3d p1 = m_nodes.row(m_elements(i, 0));
        Eigen::Vector3d p2 = m_nodes.row(m_elements(i, 1));
        Eigen::Vector3d p3 = m_nodes.row(m_elements(i, 2));
        Eigen::Vector3d p4 = m_nodes.row(m_elements(i, 3));
        
        // Compute matrix for gradient calculation
        Eigen::Matrix3d mat;
        mat.col(0) = p1 - p4;
        mat.col(1) = p2 - p4;
        mat.col(2) = p3 - p4;
        
        // Compute volume
        double volume = std::abs(mat.determinant()) / 6.0;
        
        // Compute inverse matrix for gradient calculation
        Eigen::Matrix3d invMat = mat.inverse();
        
        // Compute gradients
        Eigen::Matrix<double, 4, 3> grads;
        grads.row(0) = invMat.row(0);
        grads.row(1) = invMat.row(1);
        grads.row(2) = invMat.row(2);
        grads.row(3) = -grads.row(0) - grads.row(1) - grads.row(2);
        
        // Store element data
        m_element_data[i].volume = volume;
        m_element_data[i].grads = grads;
    }
}

void TetrahedralMesh::assembleMassMatrix(Eigen::SparseMatrix<double>& matrix, double scalar) const {
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(m_elements.rows() * 16); // Estimate number of nonzeros
    
    for (size_t i = 0; i < m_elements.rows(); ++i) {
        const ElementData& elemData = m_element_data[i];
        double factor = scalar * elemData.volume / 20.0; // Mass matrix coefficient
        
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                int nodeJ = m_elements(i, j);
                int nodeK = m_elements(i, k);
                triplets.emplace_back(nodeJ, nodeK, factor);
            }
        }
    }
    
    matrix.setFromTriplets(triplets.begin(), triplets.end());
}

void TetrahedralMesh::assembleStiffnessMatrix(Eigen::SparseMatrix<double>& matrix, double scalar) const {
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(m_elements.rows() * 16); // Estimate number of nonzeros
    
    for (size_t i = 0; i < m_elements.rows(); ++i) {
        const ElementData& elemData = m_element_data[i];
        const Eigen::Matrix<double, 4, 3>& grads = elemData.grads;
        double factor = scalar * elemData.volume;
        
        // Compute local stiffness matrix: grad(i) * grad(j)
        Eigen::Matrix4d localStiffness = Eigen::Matrix4d::Zero();
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                localStiffness(j, k) = factor * grads.row(j).dot(grads.row(k));
            }
        }
        
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                int nodeJ = m_elements(i, j);
                int nodeK = m_elements(i, k);
                triplets.emplace_back(nodeJ, nodeK, localStiffness(j, k));
            }
        }
    }
    
    matrix.setFromTriplets(triplets.begin(), triplets.end());
}
