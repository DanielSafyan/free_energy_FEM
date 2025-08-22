#ifndef HDF5_READER_HPP
#define HDF5_READER_HPP

#include <Eigen/Dense>
#include <string>
#include <map>

// Forward declaration for HDF5 types
namespace H5 {
    class H5File;
    class DataSet;
}

class HDF5Reader {
public:
    // Constructor opens the HDF5 file
    HDF5Reader(const std::string& filename);
    
    // Destructor closes the file
    ~HDF5Reader();
    
    // Read mesh data
    Eigen::MatrixXd readNodes();
    Eigen::MatrixXi readElements();
    
    // Read simulation data at a specific timestep
    Eigen::VectorXd readC(int timestep);
    Eigen::VectorXd readC3(int timestep);
    Eigen::VectorXd readPhi(int timestep);
    
    // Read all simulation data
    std::vector<Eigen::VectorXd> readAllC();
    std::vector<Eigen::VectorXd> readAllC3();
    std::vector<Eigen::VectorXd> readAllPhi();
    
    // Read constants
    std::map<std::string, double> readConstants();
    
    // Read attributes
    std::map<std::string, int> readAttributes();
    
    // Get number of timesteps
    int getTimestepCount();
    
private:
    std::unique_ptr<H5::H5File> m_file;
    int m_timestepCount;
    
    // Helper to read datasets
    Eigen::MatrixXd readMatrix(const std::string& datasetName);
    Eigen::VectorXd readVector(const std::string& datasetName, int timestep = -1);
    std::vector<Eigen::VectorXd> readVectorSeries(const std::string& datasetName);
};

#endif // HDF5_READER_HPP
