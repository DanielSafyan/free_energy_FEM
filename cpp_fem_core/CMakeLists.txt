cmake_minimum_required(VERSION 3.10)
project(FEMSimulationCore)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable position independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Explicitly set Python executable and libraries for pybind11
set(PYBIND11_PYTHON_VERSION 3.13)
set(PYBIND11_FINDPYTHON ON)

# Find required packages
find_package(Eigen3 REQUIRED)

# Try to find HDF5 but make it optional
find_package(HDF5 COMPONENTS CXX)

# Try to find pybind11 but make it optional
find_package(pybind11 CONFIG)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIR})
if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
endif()
include_directories(src)

# Add library
add_library(fem_core
    src/TetrahedralMesh.cpp
    src/NPENSimulation.cpp
)

if(HDF5_FOUND)
    target_link_libraries(fem_core ${HDF5_LIBRARIES})
    target_compile_definitions(fem_core PRIVATE HAVE_HDF5)
endif()

# Add executable for testing
add_executable(test_simulation test/test_simulation.cpp)
target_link_libraries(test_simulation fem_core)
if(HDF5_FOUND)
    target_link_libraries(test_simulation ${HDF5_LIBRARIES})
endif()

# Add Python binding module if pybind11 is found
if(pybind11_FOUND)
    pybind11_add_module(fem_core_py src/pybind_module.cpp)
    target_link_libraries(fem_core_py PRIVATE fem_core)
    if(HDF5_FOUND)
        target_link_libraries(fem_core_py PRIVATE ${HDF5_LIBRARIES})
    endif()
endif()
