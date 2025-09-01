cmake_minimum_required(VERSION 3.19)
project(FEMSimulationCore)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable position independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Let pybind11 find the active Python interpreter/libraries dynamically
set(PYBIND11_FINDPYTHON ON)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths().get('platlib') or sysconfig.get_paths()['purelib'])"
    OUTPUT_VARIABLE PY_SITE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Cache Python info so external scripts can retrieve it reliably
set(PY_SITE "${PY_SITE}" CACHE PATH "Python site-packages for install")
if (DEFINED Python3_EXECUTABLE)
  set(PYTHON3_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "Python3 executable used by CMake")
endif()
if (DEFINED Python3_VERSION)
  set(PYTHON3_VERSION "${Python3_VERSION}" CACHE STRING "Python3 version used by CMake")
endif()

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

    # Install Python module into the active interpreter's site-packages
    install(TARGETS fem_core_py
        LIBRARY DESTINATION ${PY_SITE}
        RUNTIME DESTINATION ${PY_SITE}
        ARCHIVE DESTINATION ${PY_SITE}
    )
    if(HDF5_FOUND)
        target_link_libraries(fem_core_py PRIVATE ${HDF5_LIBRARIES})
    endif()
endif()
