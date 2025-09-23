#ifndef FLUX_CALCULATOR_HPP
#define FLUX_CALCULATOR_HPP

#include "TetrahedralMesh.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

/**
 * @brief FluxCalculator computes flux vectors J = -D1 ∇c - z1 D1 c ∇phi
 * 
 * This class follows the same design patterns as NPENSimulation and works
 * with the TetrahedralMesh infrastructure to compute flux vectors efficiently.
 * The flux is computed at mesh nodes using FEM gradient reconstruction.
 */
class FluxCalculator {
public:
    /**
     * @brief Constructor
     * @param mesh Shared pointer to tetrahedral mesh
     * @param D1 Diffusion coefficient (dimensionless)
     * @param z1 Ion valence (dimensionless)
     */
    FluxCalculator(std::shared_ptr<TetrahedralMesh> mesh, double D1, double z1);
    
    /**
     * @brief Compute flux vectors at mesh nodes
     * @param c Concentration field at nodes
     * @param phi Potential field at nodes
     * @param flux_vectors Output matrix (numNodes x 3) containing flux vectors
     */
    void computeFlux(const Eigen::VectorXd& c, const Eigen::VectorXd& phi, 
                     Eigen::MatrixXd& flux_vectors) const;
    
    /**
     * @brief Compute flux vectors for multiple time steps
     * @param c_history Matrix (numTimeSteps x numNodes) of concentration history
     * @param phi_history Matrix (numTimeSteps x numNodes) of potential history
     * @param flux_tensor Output tensor (numTimeSteps x numNodes x 3) containing flux vectors
     */
    void computeFluxHistory(const Eigen::MatrixXd& c_history, 
                           const Eigen::MatrixXd& phi_history,
                           std::vector<Eigen::MatrixXd>& flux_tensor) const;
    
    /**
     * @brief Compute gradient of a scalar field at mesh nodes
     * @param field Scalar field values at nodes
     * @param gradient Output matrix (numNodes x 3) containing gradient vectors
     */
    void computeGradient(const Eigen::VectorXd& field, Eigen::MatrixXd& gradient) const;
    
    /**
     * @brief Get mesh reference
     */
    const TetrahedralMesh& getMesh() const { return *m_mesh; }
    
    /**
     * @brief Get parameters
     */
    double getD1() const { return m_D1; }
    double getZ1() const { return m_z1; }

private:
    // Mesh reference
    std::shared_ptr<TetrahedralMesh> m_mesh;
    
    // Physical parameters
    double m_D1;  // Diffusion coefficient (dimensionless)
    double m_z1;  // Ion valence (dimensionless)
    
    /**
     * @brief Compute element-wise gradient for a scalar field
     * @param field Scalar field values at nodes
     * @param element_idx Element index
     * @return 3D gradient vector (constant within the element)
     */
    Eigen::Vector3d computeElementGradient(const Eigen::VectorXd& field, size_t element_idx) const;
    
    /**
     * @brief Average element gradients to nodes using volume weighting
     * @param element_gradients Vector of gradients for each element
     * @param node_gradients Output matrix (numNodes x 3) of averaged gradients
     */
    void averageGradientsToNodes(const std::vector<Eigen::Vector3d>& element_gradients,
                                Eigen::MatrixXd& node_gradients) const;
};

#endif // FLUX_CALCULATOR_HPP
