# Building and Using the C++ FEM Core

This document provides instructions for building and using the C++ FEM core implementation.

## Prerequisites

To build the C++ FEM core, you need:

1. C++ compiler with C++17 support (GCC 7+, Clang 5+, or MSVC 2017+)
2. CMake 3.10 or higher
3. Eigen3 library
4. Optional: pybind11 for Python bindings
5. Optional: HDF5 C++ library

## Installing Dependencies

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install cmake libeigen3-dev pybind11-dev libhdf5-dev
```

### macOS with Homebrew

```bash
brew install cmake eigen pybind11 hdf5
```

### Using Conda Environment

We provide a conda environment file that includes all necessary dependencies:

```bash
conda env create -f environment.yml
conda activate fem_cpp_env
```

## Building the C++ Core

1. Navigate to the `cpp_fem_core` directory:

```bash
cd cpp_fem_core
```

2. Create a build directory and navigate to it:

```bash
mkdir build
cd build
```

3. Configure the project with CMake:

```bash
cmake ..
```

4. Build the project:

```bash
make
```

This will build:
- `libfem_core.a`: The static library containing the FEM core implementation
- `test_simulation`: A test executable
- `fem_core_py.so` (if pybind11 is found): Python bindings module

## Running Tests

After building, you can run the test executable:

```bash
./test_simulation
```

## Using Python Bindings

If pybind11 was found during the build process, you can use the C++ FEM core from Python:

```python
import sys
sys.path.append('build')  # Add the build directory to Python path

import fem_core_py as fem_cpp
import numpy as np

# Create a mesh
nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
elements = np.array([[0, 1, 2, 3]], dtype=np.int32)

mesh = fem_cpp.TetrahedralMesh(nodes, elements)
simulation = fem_cpp.NPENSimulation(mesh, 0.01, 1e-9, 1e-9, 1e-9, 1, -1, 8.854e-12, 8.314, 298, 1e-3, 1.0)

# Run simulation steps
# ... (see test_python_bindings.py for a complete example)
```

## Performance Benefits

The C++ implementation provides significant performance improvements over the Python version:

1. **Compiled vs Interpreted**: C++ code is compiled to machine code, while Python is interpreted
2. **Memory Efficiency**: C++ has no object overhead per array element
3. **Optimized Libraries**: Eigen provides highly optimized linear algebra operations
4. **Direct Memory Access**: No Python object creation overhead in tight loops

In our benchmarks, we expect to see 5-15x performance improvements, especially for larger meshes.

## Extending the Implementation

The C++ FEM core is designed to be modular and extensible:

1. **Adding New Physics**: Extend the `NPENSimulation` class to implement additional physics
2. **New Element Types**: Extend the `TetrahedralMesh` class to support other element types
3. **Different Solvers**: Replace the Eigen solvers with other sparse linear algebra libraries
4. **Parallelization**: Add OpenMP or MPI support for distributed computing

## Troubleshooting

### Missing Dependencies

If CMake cannot find Eigen3, you may need to specify its location:

```bash
cmake -DEigen3_DIR=/path/to/eigen3 ..
```

### Python Bindings Not Building

If the Python bindings are not being built, ensure pybind11 is installed and visible to CMake:

```bash
# Check if pybind11 is installed
python -c "import pybind11; print(pybind11.get_cmake_dir())"

# If needed, specify the path to pybind11
pip install pybind11
cmake -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") ..
```
