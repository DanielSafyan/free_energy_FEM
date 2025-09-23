#ifndef CURRENT_CALCULATOR_HPP
#define CURRENT_CALCULATOR_HPP

#include "TetrahedralMesh.hpp"
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief CurrentCalculator computes a scalar electric current field per node and
 *        (optionally) writes a time-series dataset to HDF5.
 *
 * Physics follows the Python reference calculate_current() used in NPEN runs:
 *   1) Build ionic current density J on each element using FEM gradients
 *      and NPEN single-salt model (both ions use c):
 *         grad_c  = sum_i c_i * grad(N_i)
 *         grad_phi= sum_i phi_i * grad(N_i)
 *         c_avg   = mean(c_i)
 *         J_elem  = F * [ z1 * (K_GRAD_C1*grad_c + K_MIG1*c_avg*grad_phi)
 *                        + z2 * (K_GRAD_C2*grad_c + K_MIG2*c_avg*grad_phi) ]
 *      with constants (dimensionally consistent to Python implementation):
 *         K_GRAD_Ck = -Dk * c0
 *         K_MIGk    = -(zk * F * Dk / (R*T)) * phi_c
 *
 *   2) Convert vector J_elem to a scalar current at each mesh node by integrating
 *      J·n over the three faces incident to that node within each adjacent tetra.
 *      The contribution per face is (J_elem · unit_normal) * area, and the sum
 *      over the three faces is divided by 3 (as in the Python reference).
 *
 *   3) The resulting nodal scalar quantity has units of current [A] per node
 *      (matching the electrode-node measurement semantics in calculate_current),
 *      and can be stored as a time-series dataset of shape (T, N) in HDF5 under
 *      e.g. states/current.
 *
 * Vectorization strategy (vs. the Python loop):
 *   - Gradients (grad_c, grad_phi) are computed element-wise in batch using the
 *     precomputed basis function gradients (ElementData.grads) and matrix ops.
 *   - J_elem is computed for all elements at once using Eigen arrays/matrices.
 *   - Only the final accumulation of element contributions to incident nodes
 *     uses sparse-style loops (one pass over precomputed node-face geometry).
 */
class CurrentCalculator {
public:
    /**
     * @brief Constructor
     * @param mesh   Shared tetrahedral mesh
     * @param R      Gas constant
     * @param T      Temperature
     * @param F      Faraday constant
     * @param D1     Diffusion coeff (species 1)
     * @param D2     Diffusion coeff (species 2)
     * @param z1     Valence (species 1)
     * @param z2     Valence (species 2)
     * @param c0     Concentration scale to convert dimensionless c -> phys
     * @param phi_c  Potential scale to convert dimensionless phi -> volts
     */
    CurrentCalculator(std::shared_ptr<TetrahedralMesh> mesh,
                      double R, double T, double F,
                      double D1, double D2,
                      double z1, double z2,
                      double c0, double phi_c);

    /**
     * @brief Compute scalar current per node for a single time step.
     * @param c     (N) nodal concentration (dimensionless)
     * @param phi   (N) nodal potential (dimensionless)
     * @param out_current (N) output scalar current per node [A]
     */
    void computeCurrentScalar(const Eigen::VectorXd& c,
                              const Eigen::VectorXd& phi,
                              Eigen::VectorXd& out_current) const;

    /**
     * @brief Compute scalar current time-series for multiple time steps.
     * @param c_history    (T x N)
     * @param phi_history  (T x N)
     * @param out_history  (T x N) scalar current per node [A]
     */
    void computeCurrentHistory(const Eigen::MatrixXd& c_history,
                               const Eigen::MatrixXd& phi_history,
                               Eigen::MatrixXd& out_history) const;

    /**
     * @brief Write (or overwrite/append) current history into an HDF5 file.
     *        The dataset will be created at datasetPath (default: states/current)
     *        with shape (T, N). This API is a no-op if the library was built
     *        without HDF5; the implementation will throw with a clear message.
     *
     * @param h5Filename   Path to HDF5 file
     * @param current      (T x N) current history to write
     * @param datasetPath  HDF5 dataset path (default: states/current)
     * @param overwrite    If true and dataset exists, it will be replaced
     */
    void writeCurrentToH5(const std::string& h5Filename,
                          const Eigen::MatrixXd& current,
                          const std::string& datasetPath = "states/current",
                          bool overwrite = false) const;

    /** @brief Get mesh reference */
    const TetrahedralMesh& getMesh() const { return *m_mesh; }

    // Accessors for constants
    double getR() const { return m_R; }
    double getT() const { return m_T; }
    double getF() const { return m_F; }
    double getD1() const { return m_D1; }
    double getD2() const { return m_D2; }
    double getZ1() const { return m_z1; }
    double getZ2() const { return m_z2; }
    double getC0() const { return m_c0; }
    double getPhiC() const { return m_phi_c; }

private:
    // Mesh
    std::shared_ptr<TetrahedralMesh> m_mesh;

    // Physical parameters (match python calculate_current)
    double m_R, m_T, m_F;
    double m_D1, m_D2;
    double m_z1, m_z2;
    double m_c0, m_phi_c;

    // Derived constant factors
    double m_K_GRAD_C1, m_K_GRAD_C2; // -Dk * c0
    double m_K_MIG1, m_K_MIG2;       // -(zk * F * Dk / (R*T)) * phi_c

    // Per-node incident face geometry cache
    struct FaceGeom {
        Eigen::Vector3d unit_normal; // outward from node within element
        double area;                  // face area
    };

    // For each node: list of (element index, vector of incident faces (3))
    using NodeFaces = std::vector<std::pair<size_t, std::vector<FaceGeom>>>;

    mutable std::vector<NodeFaces> m_node_faces_cache; // sized to numNodes, lazily built

    // Lazy builders
    void buildNodeToFacesCache() const;

    // Vectorized element computations
    // Compute element-wise grad(field) for all elements -> (E x 3)
    void computeElementGradientsBatch(const Eigen::VectorXd& field,
                                      Eigen::MatrixXd& grads_all) const;

    // Compute per-element average c (physical) using c0 scale -> (E)
    void computeElementAvgCPhysBatch(const Eigen::VectorXd& c,
                                     Eigen::VectorXd& c_avg_phys) const;

    // Compute J for all elements (E x 3) using precomputed grad_c, grad_phi, c_avg_phys
    void computeElementCurrentsBatch(const Eigen::MatrixXd& grad_c,
                                     const Eigen::MatrixXd& grad_phi,
                                     const Eigen::VectorXd& c_avg_phys,
                                     Eigen::MatrixXd& J_elem) const;

    // Accumulate element currents to nodal scalar using precomputed node-face geometry
    void accumulateToNodes(const Eigen::MatrixXd& J_elem,
                           Eigen::VectorXd& out_current) const;
};

#endif // CURRENT_CALCULATOR_HPP
