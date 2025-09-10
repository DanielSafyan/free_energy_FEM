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

## Windows (Anaconda PowerShell + Visual Studio) Build Guide

This project ships with a convenient PowerShell build script and a CMake configuration that targets Visual Studio on Windows. The steps below assume you are using the Anaconda/Miniconda PowerShell prompt.

### Prerequisites (Windows)
- Visual Studio 2022 (Community/Pro/Enterprise) with the "Desktop development with C++" workload installed
- Anaconda or Miniconda
- CMake 3.19+ (installed via conda environment or system)
- Optional: vcpkg if you want HDF5 support

### 1) Create and activate the conda environment
```powershell
conda env create -f environment.yml
conda activate sim_env
```

The environment includes Python, CMake, Eigen, pybind11, and HDF5. On Windows, CMake may still require you to point to the Eigen headers explicitly (see step 3).

### 2) Allow running the local build script (current session only)
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Unblock-File .\build_and_test.ps1
```

### 3) Configure generator/architecture and optional dependencies
```powershell
# Use Visual Studio 2022 and x64
$env:GENERATOR = 'Visual Studio 17 2022'
$env:ARCH      = 'x64'
$env:CONFIG    = 'Release'   # or Debug

# If CMake cannot find Eigen automatically via conda, point it to the headers
$env:EIGEN3_INCLUDE_DIR = "$Env:CONDA_PREFIX\Library\include\eigen3"

# Optional: enable HDF5 via vcpkg (otherwise it is disabled by default)
# $env:VCPKG_ROOT    = 'C:\vcpkg'
# $env:DISABLE_HDF5  = 'OFF'   # turn ON (default) to skip HDF5 detection
```

### 4) Build, install, and test (recommended)
```powershell
.\build_and_test.ps1
```

The script will:
- Configure CMake for Visual Studio with the active Python
- Build the C++ targets (including `test_simulation`)
- Install the Python module `fem_core_py` into your conda environment’s site-packages
- Run a short Python smoke test

### 5) Manual CMake commands (alternative to the script)
```powershell
# Configure
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DPython3_EXECUTABLE="$Env:CONDA_PREFIX\python.exe" `
  -DCMAKE_DISABLE_FIND_PACKAGE_HDF5=ON `
  -DEIGEN3_INCLUDE_DIR="$Env:CONDA_PREFIX\Library\include\eigen3"

# If using vcpkg + HDF5 instead, enable detection and pass the toolchain
# cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
#   -DPython3_EXECUTABLE="$Env:CONDA_PREFIX\python.exe" `
#   -DCMAKE_TOOLCHAIN_FILE="$Env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
#   -DCMAKE_DISABLE_FIND_PACKAGE_HDF5=OFF `
#   -DEIGEN3_INCLUDE_DIR="$Env:CONDA_PREFIX\Library\include\eigen3"

# Build (Visual Studio/MSBuild)
cmake --build build --config Release -j

# Run the C++ test executable
.\build\Release\test_simulation.exe

# Install Python module into the active environment
cmake --install build --config Release

# Verify Python import
python -c "import fem_core_py as m; print('OK:', m)"
```

### 6) Building from the Visual Studio IDE (optional)
- After a successful CMake configure, open `build/FEMSimulationCore.sln` in Visual Studio
- Select configuration `Release` and platform `x64`
- Build the `ALL_BUILD` target, then optionally run `test_simulation`
- To install the Python module, either run `cmake --install build --config Release` from PowerShell or add an INSTALL step in VS

### Troubleshooting (Windows)
- pybind11 not found: ensure it’s installed in the active conda env (`python -c "import pybind11, sys; print(pybind11.get_cmake_dir())"`), and if needed pass `-Dpybind11_DIR=<that path>` on the CMake configure line.
- Eigen not found: set `EIGEN3_INCLUDE_DIR` to the conda path shown above.
- MSBuild not found: run from a "Developer PowerShell for VS 2022" or initialize the environment:
  ```powershell
  cmd /c "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64
  ```
- HDF5 detection: keep it disabled (default) or use vcpkg (`VCPKG_ROOT`) and set `DISABLE_HDF5=OFF`.
- Python import finds the wrong binary: ensure there is no stale `fem_core_py*.pyd` in your project root that could shadow the newly installed one in site-packages.

## Build Instructions (Linux/macOS)

```bash
mkdir build
cd build
cmake ..
make
```
