#ifndef TETRAHEDRAL_MESH_HPP
#define TETRAHEDRAL_MESH_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

// Structure to hold precomputed element data
struct ElementData {
    double volume;
    Eigen::Matrix<double, 4, 3> grads; // Gradients for the 4 basis functions
};

class TetrahedralMesh {
public:
    // Constructor
    TetrahedralMesh(const Eigen::MatrixXd& nodes, const Eigen::MatrixXi& elements);
    
    // Getters
    const Eigen::MatrixXd& getNodes() const { return m_nodes; }
    const Eigen::MatrixXi& getElements() const { return m_elements; }
    size_t numNodes() const { return m_nodes.rows(); }
    size_t numElements() const { return m_elements.rows(); }
    const ElementData& getElementData(size_t elemIdx) const { return m_element_data[elemIdx]; }
    // Boundary faces (triangles) on the outer surface; each row has 3 node indices
    const Eigen::MatrixXi& getBoundaryFaces() const { return m_boundaryFaces; }
    
    // Matrix assembly methods
    void assembleMassMatrix(Eigen::SparseMatrix<double>& matrix, double scalar = 1.0) const;
    void assembleStiffnessMatrix(Eigen::SparseMatrix<double>& matrix, double scalar = 1.0) const;
    // Assemble a boundary (surface) mass matrix over a set of boundary faces
    // face_ids are indices into getBoundaryFaces() rows
    void assembleBoundaryMassMatrix(Eigen::SparseMatrix<double>& matrix,
                                    const std::vector<int>& face_ids,
                                    double scalar = 1.0) const;
    
private:
    Eigen::MatrixXd m_nodes;      // Node coordinates (nx3)
    Eigen::MatrixXi m_elements;   // Element connectivity (ex4)
    std::vector<ElementData> m_element_data;  // Precomputed element data
    Eigen::MatrixXi m_boundaryFaces; // (nf x 3) boundary triangles (outer surface)
    
    // Precompute element data (volume and gradients)
    void precomputeElementData();
    // Discover boundary faces from element connectivity
    void computeBoundaryFaces();
};

#endif // TETRAHEDRAL_MESH_HPP
