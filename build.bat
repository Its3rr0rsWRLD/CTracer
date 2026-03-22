@echo off
setlocal enabledelayedexpansion

where cmake >nul 2>&1
if !ERRORLEVEL! NEQ 0 goto :no_cmake

where nvcc >nul 2>&1
if !ERRORLEVEL! NEQ 0 goto :no_nvcc

set "CUDA_DIR="
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" set "CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if "!CUDA_DIR!"=="" if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" set "CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
if "!CUDA_DIR!"=="" if defined CUDA_PATH set "CUDA_DIR=!CUDA_PATH!"
if "!CUDA_DIR!"=="" goto :no_cuda
echo Using CUDA: !CUDA_DIR!

set "NINJA_EXE="
where ninja >nul 2>&1 && set "NINJA_EXE=ninja"
if not "!NINJA_EXE!"=="" goto :ninja_found
for /f "delims=" %%P in ('dir /s /b "C:\Program Files\Microsoft Visual Studio\*ninja.exe" 2^>nul') do set "NINJA_EXE=%%P"
:ninja_found
if "!NINJA_EXE!"=="" goto :no_ninja
echo Using Ninja: !NINJA_EXE!

where cl >nul 2>&1
if !ERRORLEVEL! EQU 0 goto :msvc_ready

echo Setting up MSVC environment...
set "VCVARS="
for /f "delims=" %%V in ('dir /s /b "C:\Program Files\Microsoft Visual Studio\*vcvarsall.bat" 2^>nul') do set "VCVARS=%%V"
if "!VCVARS!"=="" goto :no_vcvars

set "SAVE_CUDA=!CUDA_DIR!"
set "SAVE_NINJA=!NINJA_EXE!"
set "SAVE_VCVARS=!VCVARS!"
endlocal & set "CUDA_DIR=%SAVE_CUDA%" & set "NINJA_EXE=%SAVE_NINJA%" & set "VCVARS=%SAVE_VCVARS%"
call "%VCVARS%" x64
if %ERRORLEVEL% NEQ 0 goto :no_msvc
setlocal enabledelayedexpansion

:msvc_ready

if not exist build\CMakeCache.txt goto :cache_ok
findstr /c:"CMAKE_GENERATOR:INTERNAL=Ninja" build\CMakeCache.txt >nul 2>&1
if !ERRORLEVEL! NEQ 0 goto :clean_cache
findstr /c:"CMAKE_CUDA_COMPILER_ID" build\CMakeCache.txt >nul 2>&1
if !ERRORLEVEL! NEQ 0 goto :clean_cache
goto :cache_ok

:clean_cache
echo Clearing stale build cache...
rmdir /s /q build

:cache_ok
if not exist build mkdir build
cd build

echo.
echo [1/2] Configuring...
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="!CUDA_DIR!\bin\nvcc.exe" -DCUDAToolkit_ROOT="!CUDA_DIR!" -DCMAKE_MAKE_PROGRAM="!NINJA_EXE!" -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
if !ERRORLEVEL! NEQ 0 goto :configure_fail

echo.
echo [2/2] Building...
cmake --build . --config Release
if !ERRORLEVEL! NEQ 0 goto :build_fail

cd ..
echo.
echo Build succeeded. Run: build\ctracer.exe --port 8000
exit /b 0

:no_cmake
echo ERROR: cmake not found in PATH.
exit /b 1

:no_nvcc
echo ERROR: nvcc not found in PATH. Install CUDA Toolkit and add its bin/ to PATH.
exit /b 1

:no_cuda
echo ERROR: Could not find CUDA Toolkit installation.
exit /b 1

:no_ninja
echo ERROR: ninja not found.
exit /b 1

:no_vcvars
echo ERROR: vcvarsall.bat not found. Install Visual Studio C++ workload.
exit /b 1

:no_msvc
echo ERROR: Failed to set up MSVC environment.
exit /b 1

:configure_fail
cd ..
echo ERROR: CMake configure failed.
exit /b 1

:build_fail
cd ..
echo ERROR: Build failed.
exit /b 1
