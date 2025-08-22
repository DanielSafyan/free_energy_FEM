# C++ FEM Core Implementation

This directory contains a C++ implementation of the core FEM simulation components for improved performance.

## Components

1. **FEM Mesh** - Tetrahedral mesh handling with precomputed element data
2. **NPEN Solver** - Nernst-Planck Electroneutral simulation solver
3. **HDF5 I/O** - Reading and writing simulation data

## Dependencies

- Eigen3 (Linear algebra)
- HDF5 (Data storage)
- CMake (Build system)

## Build Instructions

```bash
mkdir build
cd build
cmake ..
make
```
