#include "TetrahedralMesh.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <cmath>
 #include <unordered_map>

TetrahedralMesh::TetrahedralMesh(const Eigen::MatrixXd& nodes, const Eigen::MatrixXi& elements)
    : m_nodes(nodes), m_elements(elements) {
    precomputeElementData();
    computeBoundaryFaces();
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

// Identify boundary faces (triangles) as those shared by exactly one tetrahedron
void TetrahedralMesh::computeBoundaryFaces() {
    struct FaceKey { int a,b,c; };
    struct KeyHash { std::size_t operator()(const FaceKey& k) const noexcept {
        std::size_t h = static_cast<std::size_t>(k.a);
        h = h * 1315423911u + static_cast<std::size_t>(k.b);
        h = h * 2654435761u + static_cast<std::size_t>(k.c);
        return h;
    }};
    struct KeyEq { bool operator()(const FaceKey& x, const FaceKey& y) const noexcept {
        return x.a==y.a && x.b==y.b && x.c==y.c;
    }};

    std::unordered_map<FaceKey, int, KeyHash, KeyEq> counts;
    std::unordered_map<FaceKey, Eigen::Vector3i, KeyHash, KeyEq> repr;

    auto add_face = [&](int i0, int i1, int i2){
        int a=i0,b=i1,c=i2;
        if (a>b) std::swap(a,b);
        if (b>c) std::swap(b,c);
        if (a>b) std::swap(a,b);
        FaceKey key{a,b,c};
        counts[key] += 1;
        if (!repr.count(key)) repr[key] = Eigen::Vector3i(i0,i1,i2);
    };

    for (int e = 0; e < m_elements.rows(); ++e) {
        int n0 = m_elements(e,0), n1 = m_elements(e,1), n2 = m_elements(e,2), n3 = m_elements(e,3);
        add_face(n0,n1,n2);
        add_face(n0,n1,n3);
        add_face(n0,n2,n3);
        add_face(n1,n2,n3);
    }

    std::vector<Eigen::Vector3i> faces;
    faces.reserve(counts.size());
    for (const auto& kv : counts) {
        if (kv.second == 1) {
            auto it = repr.find(kv.first);
            if (it != repr.end()) faces.push_back(it->second);
        }
    }
    m_boundaryFaces.resize(static_cast<int>(faces.size()), 3);
    for (int i=0; i<static_cast<int>(faces.size()); ++i) m_boundaryFaces.row(i) = faces[i];
}

// Assemble boundary (surface) mass matrix over selected boundary faces
void TetrahedralMesh::assembleBoundaryMassMatrix(Eigen::SparseMatrix<double>& matrix,
                                                 const std::vector<int>& face_ids,
                                                 double scalar) const {
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(face_ids.size()*9);
    auto tri_area = [&](const Eigen::Vector3d& A, const Eigen::Vector3d& B, const Eigen::Vector3d& C){
        return 0.5 * ((B-A).cross(C-A)).norm();
    };
    for (int fid : face_ids) {
        if (fid < 0 || fid >= m_boundaryFaces.rows()) continue;
        int i = m_boundaryFaces(fid,0);
        int j = m_boundaryFaces(fid,1);
        int k = m_boundaryFaces(fid,2);
        Eigen::Vector3d A = m_nodes.row(i);
        Eigen::Vector3d B = m_nodes.row(j);
        Eigen::Vector3d C = m_nodes.row(k);
        double Atri = tri_area(A,B,C);
        double w_ii = scalar * (Atri / 6.0);   // (A/12)*2
        double w_ij = scalar * (Atri / 12.0);  // (A/12)*1
        trips.emplace_back(i,i,w_ii);
        trips.emplace_back(j,j,w_ii);
        trips.emplace_back(k,k,w_ii);
        trips.emplace_back(i,j,w_ij);
        trips.emplace_back(i,k,w_ij);
        trips.emplace_back(j,i,w_ij);
        trips.emplace_back(j,k,w_ij);
        trips.emplace_back(k,i,w_ij);
        trips.emplace_back(k,j,w_ij);
    }
    matrix.resize(m_nodes.rows(), m_nodes.rows());
    matrix.setFromTriplets(trips.begin(), trips.end());
    matrix.makeCompressed();
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
