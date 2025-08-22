#ifndef HDF5_WRITER_HPP
#define HDF5_WRITER_HPP

#include <Eigen/Dense>
#include <string>
#include <map>
#include <vector>

// Forward declaration for HDF5 types
namespace H5 {
    class H5File;
}

class HDF5Writer {
public:
    // Constructor creates/opens the HDF5 file
    HDF5Writer(const std::string& filename);
    
    // Destructor closes the file
    ~HDF5Writer();
    
    // Write mesh data
    void writeNodes(const Eigen::MatrixXd& nodes);
    void writeElements(const Eigen::MatrixXi& elements);
    
    // Write simulation data at a specific timestep
    void writeC(const Eigen::VectorXd& c, int timestep);
    void writeC3(const Eigen::VectorXd& c3, int timestep);
    void writePhi(const Eigen::VectorXd& phi, int timestep);
    
    // Write constants
    void writeConstants(const std::map<std::string, double>& constants);
    
    // Write attributes
    void writeAttributes(const std::map<std::string, int>& attributes);
    
private:
    std::unique_ptr<H5::H5File> m_file;
    
    // Helper to write datasets
    void writeMatrix(const std::string& datasetName, const Eigen::MatrixXd& matrix);
    void writeVector(const std::string& datasetName, const Eigen::VectorXd& vector);
};

#endif // HDF5_WRITER_HPP
