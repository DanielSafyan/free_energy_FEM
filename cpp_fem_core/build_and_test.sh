#!/bin/bash

# Build and test script for C++ FEM core

echo "Building C++ FEM core..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
make

# Run test
echo "Running test..."
./test_simulation

echo "Build and test completed!"

echo "If pybind11 was found, the Python module fem_core_py should be available in the build directory."
ls -la fem_core_py*.so 2>/dev/null || echo "Python module not built (pybind11 not found)"
