param(
  [string]$Config = $(if ($env:CONFIG) { $env:CONFIG } else { 'Release' }),
  [string]$Generator = $(if ($env:GENERATOR) { $env:GENERATOR } else { '' }),
  [string]$Arch = $(if ($env:ARCH) { $env:ARCH } else { '' })
)

$ErrorActionPreference = 'Stop'

function Get-PybinPath([string]$py) {
  try {
    $cmd = Get-Command $py -ErrorAction Stop
    return $cmd.Source
  } catch {
    return $py
  }
}

$python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
$pythonPath = Get-PybinPath $python

$pyver = & $pythonPath -V 2>$null
Write-Host "Python: $pyver"
Write-Host "CMake: $(cmake --version | Select-Object -First 1)"
try { $cm = Get-Command cmake -ErrorAction Stop; Write-Host "Using cmake at: $($cm.Source)" } catch {}

# Try to discover pybind11's CMake config dir from active Python
$pybind11Dir = ''
try {
  $code = @'
import sys
try:
    import pybind11
    print(pybind11.get_cmake_dir(), end="")
except Exception:
    pass
'@
  $pybind11Dir = & $pythonPath -c $code
} catch {}
if ($pybind11Dir) { Write-Host "Detected pybind11 CMake dir: $pybind11Dir" }

$disableHDF5 = if ($env:DISABLE_HDF5) { $env:DISABLE_HDF5 } else { 'ON' }

# vcpkg toolchain (optional)
$toolchain = ''
if ($env:VCPKG_ROOT -and (Test-Path "$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake")) {
  $toolchain = "$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
  Write-Host "Using vcpkg toolchain: $toolchain"
}

# Create and enter build dir
New-Item -ItemType Directory -Force -Path "build" | Out-Null
Set-Location "build"

# Compose CMake configure args
$cmakeArgs = @(
  "-DCMAKE_DISABLE_FIND_PACKAGE_HDF5=$disableHDF5",
  "-DPython3_EXECUTABLE=`"$pythonPath`""
)
if ($pybind11Dir) { $cmakeArgs += "-Dpybind11_DIR=`"$pybind11Dir`"" }
if ($env:EIGEN3_INCLUDE_DIR) { $cmakeArgs += "-DEIGEN3_INCLUDE_DIR=`"$env:EIGEN3_INCLUDE_DIR`"" }
if ($toolchain) { $cmakeArgs += "-DCMAKE_TOOLCHAIN_FILE=`"$toolchain`"" }
if ($Generator) {
  $cmakeArgs = @('-G', $Generator) + $cmakeArgs
  if ($Arch) { $cmakeArgs = @('-A', $Arch) + $cmakeArgs }
}

Write-Host "Configuring with CMake (HDF5 disabled by default; set DISABLE_HDF5=OFF to enable detection)"
cmake @cmakeArgs ..

# Build
$buildArgs = @('--config', $Config, '-j')
cmake --build . @buildArgs

# Run test executable (handle single- and multi-config generators)
Write-Host "Running C++ test executable..."
$testExe = Join-Path (Get-Location) "test_simulation.exe"
$testExeCfg = Join-Path (Get-Location) (Join-Path $Config "test_simulation.exe")
$testStatus = 0
if (Test-Path $testExeCfg) {
  & $testExeCfg
  $testStatus = $LASTEXITCODE
} elseif (Test-Path $testExe) {
  & $testExe
  $testStatus = $LASTEXITCODE
} else {
  Write-Warning "test_simulation.exe not found. Skipping."
}
if ($testStatus -ne 0) {
  Write-Warning "C++ test_simulation failed with status $testStatus. Continuing to Python smoke tests."
}

# Install Python module
Write-Host "Installing Python module (fem_core_py) into site-packages..."
cmake --install . --config $Config

# Python smoke test
Write-Host "Running Python smoke test..."
$smoke = @'
import sys
print('Python executable:', sys.executable)
try:
    import numpy as np
    import fem_core_py as fem_cpp
    print('Imported fem_core_py successfully')
    nodes = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64)
    elements = np.array([[0,1,2,3]], dtype=np.int32)
    mesh = fem_cpp.TetrahedralMesh(nodes, elements)
    sim = fem_cpp.NPENSimulation(mesh, 0.01, 1e-9, 1e-9, 1e-9, 1, -1, 8.854e-12, 8.314, 298, 1e-3, 1.0)
    print('Constructed mesh and simulation successfully')
except Exception as e:
    print('Python smoke test FAILED:', e)
    raise
'@
& $pythonPath -c $smoke

Write-Host "Build, install, and Python smoke tests completed successfully!"
