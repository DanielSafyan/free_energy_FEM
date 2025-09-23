#include "CurrentCalculator.hpp"
#include <stdexcept>

#ifdef HAVE_HDF5
  #include <H5Cpp.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector3d;

namespace {
inline bool normalize_safe(const Vector3d& v, Vector3d& out_unit, double& out_norm) {
    out_norm = v.norm();
    if (out_norm < 1e-18) return false;
    out_unit = v / out_norm;
    return true;
}
}

CurrentCalculator::CurrentCalculator(std::shared_ptr<TetrahedralMesh> mesh,
                                     double R, double T, double F,
                                     double D1, double D2,
                                     double z1, double z2,
                                     double c0, double phi_c)
    : m_mesh(std::move(mesh)), m_R(R), m_T(T), m_F(F),
      m_D1(D1), m_D2(D2), m_z1(z1), m_z2(z2), m_c0(c0), m_phi_c(phi_c) {
    if (!m_mesh) throw std::invalid_argument("CurrentCalculator: mesh cannot be null");
    m_K_GRAD_C1 = -m_D1 * m_c0;
    m_K_GRAD_C2 = -m_D2 * m_c0;
    m_K_MIG1    = -(m_z1 * m_F * m_D1 / (m_R * m_T)) * m_phi_c;
    m_K_MIG2    = -(m_z2 * m_F * m_D2 / (m_R * m_T)) * m_phi_c;
}

void CurrentCalculator::buildNodeToFacesCache() const {
    if (!m_node_faces_cache.empty()) return;
    const auto& nodes = m_mesh->getNodes();
    const auto& elements = m_mesh->getElements();
    const size_t N = m_mesh->numNodes();
    const size_t E = m_mesh->numElements();
    m_node_faces_cache.clear();
    m_node_faces_cache.resize(N);
    for (size_t e = 0; e < E; ++e) {
        int en[4] = {elements(e,0), elements(e,1), elements(e,2), elements(e,3)};
        for (int li = 0; li < 4; ++li) {
            int i = en[li];
            int others[3]; int oi = 0; for (int j=0;j<4;++j) if (j!=li) others[oi++]=j;
            int face_triplets[3][3] = {{others[0],others[1],others[2]}, {others[0],others[2],others[1]}, {others[1],others[2],others[0]}};
            std::vector<FaceGeom> face_list; face_list.reserve(3);
            Vector3d p_i = nodes.row(i);
            for (int f = 0; f < 3; ++f) {
                int a = en[face_triplets[f][0]];
                int b = en[face_triplets[f][1]];
                int opp = en[face_triplets[f][2]];
                Vector3d p_a = nodes.row(a), p_b = nodes.row(b), p_opp = nodes.row(opp);
                Vector3d v1 = p_a - p_i, v2 = p_b - p_i;
                Vector3d normal = v1.cross(v2);
                if (normal.dot(p_opp - p_i) > 0.0) normal = -normal;
                Vector3d unit; double normn; if (!normalize_safe(normal, unit, normn)) continue;
                double area = 0.5 * normn;
                face_list.push_back(FaceGeom{unit, area});
            }
            if (!face_list.empty()) m_node_faces_cache[i].emplace_back(e, std::move(face_list));
        }
    }
}

void CurrentCalculator::computeElementGradientsBatch(const VectorXd& field, MatrixXd& grads_all) const {
    const size_t E = m_mesh->numElements();
    grads_all.resize(E, 3);
    const auto& elements = m_mesh->getElements();
    for (size_t e = 0; e < E; ++e) {
        Eigen::Vector4d f_loc; for (int k=0;k<4;++k) f_loc(k) = field(elements(e,k));
        const auto& ed = m_mesh->getElementData(e);
        Vector3d grad = ed.grads.transpose() * f_loc;
        grads_all.row(e) = grad.transpose();
    }
}

void CurrentCalculator::computeElementAvgCPhysBatch(const VectorXd& c, VectorXd& c_avg_phys) const {
    const size_t E = m_mesh->numElements();
    c_avg_phys.resize(E);
    const auto& elements = m_mesh->getElements();
    for (size_t e = 0; e < E; ++e) {
        double sumc = 0.0; for (int k=0;k<4;++k) sumc += c(elements(e,k));
        c_avg_phys(e) = 0.25 * sumc * m_c0;
    }
}

void CurrentCalculator::computeElementCurrentsBatch(const MatrixXd& grad_c,
                                                    const MatrixXd& grad_phi,
                                                    const VectorXd& c_avg_phys,
                                                    MatrixXd& J_elem) const {
    const size_t E = m_mesh->numElements();
    J_elem.resize(E, 3);
    for (size_t e = 0; e < E; ++e) {
        Vector3d gc = grad_c.row(e), gp = grad_phi.row(e);
        double cavg = c_avg_phys(e);
        Vector3d term1 = m_K_GRAD_C1 * gc + (m_K_MIG1 * cavg) * gp;
        Vector3d term2 = m_K_GRAD_C2 * gc + (m_K_MIG2 * cavg) * gp;
        Vector3d J = m_F * (m_z1 * term1 + m_z2 * term2);
        J_elem.row(e) = J.transpose();
    }
}

void CurrentCalculator::accumulateToNodes(const MatrixXd& J_elem, VectorXd& out_current) const {
    const size_t N = m_mesh->numNodes();
    out_current.setZero(N);
    buildNodeToFacesCache();
    for (size_t i = 0; i < N; ++i) {
        double total = 0.0;
        for (const auto& ef : m_node_faces_cache[i]) {
            size_t e = ef.first; const auto& faces = ef.second; Vector3d J = J_elem.row(e);
            for (const auto& fg : faces) total += (J.dot(fg.unit_normal) * fg.area) / 3.0;
        }
        out_current(i) = total;
    }
}

void CurrentCalculator::computeCurrentScalar(const VectorXd& c, const VectorXd& phi, VectorXd& out_current) const {
    if (c.size() != (int)m_mesh->numNodes() || phi.size() != (int)m_mesh->numNodes())
        throw std::invalid_argument("CurrentCalculator::computeCurrentScalar: field sizes must match mesh nodes");
    MatrixXd gc, gp; computeElementGradientsBatch(c, gc); computeElementGradientsBatch(phi, gp);
    VectorXd cavg; computeElementAvgCPhysBatch(c, cavg);
    MatrixXd J; computeElementCurrentsBatch(gc, gp, cavg, J);
    out_current.resize(m_mesh->numNodes());
    accumulateToNodes(J, out_current);
}

void CurrentCalculator::computeCurrentHistory(const MatrixXd& c_history, const MatrixXd& phi_history, MatrixXd& out_history) const {
    if (c_history.rows() != phi_history.rows() || c_history.cols() != phi_history.cols())
        throw std::invalid_argument("CurrentCalculator::computeCurrentHistory: history shapes must match");
    if (c_history.cols() != (int)m_mesh->numNodes())
        throw std::invalid_argument("CurrentCalculator::computeCurrentHistory: node count mismatch");
    const size_t T = (size_t)c_history.rows(); const size_t N = m_mesh->numNodes();
    out_history.resize(T, N);
    buildNodeToFacesCache();
    MatrixXd gc, gp, J; VectorXd cavg, cur;
    for (size_t t = 0; t < T; ++t) {
        VectorXd c_t = c_history.row((int)t); VectorXd phi_t = phi_history.row((int)t);
        computeElementGradientsBatch(c_t, gc);
        computeElementGradientsBatch(phi_t, gp);
        computeElementAvgCPhysBatch(c_t, cavg);
        computeElementCurrentsBatch(gc, gp, cavg, J);
        cur.resize(N); accumulateToNodes(J, cur);
        out_history.row((int)t) = cur.transpose();
    }
}

void CurrentCalculator::writeCurrentToH5(const std::string& h5Filename, const MatrixXd& current, const std::string& datasetPath, bool overwrite) const {
#ifndef HAVE_HDF5
    (void)h5Filename; (void)current; (void)datasetPath; (void)overwrite;
    throw std::runtime_error("HDF5 not enabled. Rebuild with HDF5 to use writeCurrentToH5().");
#else
    H5::H5File file(h5Filename, H5F_ACC_RDWR);
    hsize_t dims[2] = { (hsize_t)current.rows(), (hsize_t)current.cols() };
    if (H5Lexists(file.getId(), datasetPath.c_str(), H5P_DEFAULT) > 0) {
        if (!overwrite) throw std::runtime_error("Dataset already exists: " + datasetPath);
        file.unlink(datasetPath);
    }
    H5::DataSpace space(2, dims);
    H5::DSetCreatPropList plist; // no compression by default
    H5::DataSet ds = file.createDataSet(datasetPath, H5::PredType::NATIVE_DOUBLE, space, plist);
    ds.write(current.data(), H5::PredType::NATIVE_DOUBLE);
    // minimal metadata
    ds.createAttribute("description", H5::StrType(0, H5T_VARIABLE), H5::DataSpace()).write(H5::StrType(0, H5T_VARIABLE), std::string("Nodal scalar electric current [A]"));
#endif
}
