@echo off
title VRChat Auto Fish - Install Dependencies

echo ============================================
echo   VRChat Auto Fish - Dependency Installer
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ and check "Add to PATH"
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] Python detected:
python --version
echo.

:: Check PyTorch
echo [2/3] Checking GPU...
python -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}')" 2>nul
if errorlevel 1 (
    echo   PyTorch not installed, installing now...
    call :install_torch
) else (
    echo   PyTorch already installed
)

echo.
echo [3/3] Installing other dependencies...
pip install -r requirements.txt
echo.

echo ============================================
echo   Done! Run start.bat to launch the app.
echo ============================================
pause
exit /b 0

:install_torch
echo.
echo Checking for NVIDIA GPU (CUDA support)...
setlocal EnableExtensions EnableDelayedExpansion
set "GPU_NAME="
set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu"
set "TORCH_LABEL=CPU"
set "LEGACY_NVIDIA_GPU=0"

for /f "usebackq delims=" %%G in (`nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul`) do (
    if not defined GPU_NAME set "GPU_NAME=%%G"
)

if not defined GPU_NAME (
    echo   No NVIDIA GPU detected, installing CPU version of PyTorch
) else (
    echo   NVIDIA GPU detected: !GPU_NAME!
    call :is_legacy_nvidia_gpu "!GPU_NAME!" LEGACY_NVIDIA_GPU
    if "!LEGACY_NVIDIA_GPU!"=="1" (
        echo   Legacy NVIDIA GPU detected, installing CUDA 11.8 version of PyTorch
        echo   Some older NVIDIA GPUs cannot use the default CUDA 12.8 build.
        set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118"
        set "TORCH_LABEL=CUDA 11.8"
    ) else (
        echo   Installing CUDA 12.8 version of PyTorch (GPU acceleration)
        set "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"
        set "TORCH_LABEL=CUDA 12.8"
    )
)

pip install torch torchvision --index-url !TORCH_INDEX_URL!
set "INSTALL_ERR=!ERRORLEVEL!"

if "!INSTALL_ERR!"=="0" if "!LEGACY_NVIDIA_GPU!"=="1" (
    echo.
    echo   [WARNING] 您的设备型号过于老旧，使用时可能会导致异常。
    echo   [WARNING] Detected GPU: !GPU_NAME!
    echo   [WARNING] Installed PyTorch build: !TORCH_LABEL!
)

endlocal & exit /b %INSTALL_ERR%

:is_legacy_nvidia_gpu
set "%~2=0"
echo %~1 | findstr /i /c:"GTX 6" /c:"GTX 7" /c:"GTX 8" /c:"GTX 9" /c:"GTX 10" /c:"GT 7" /c:"GT 8" /c:"GT 9" /c:"GT 10" /c:"Quadro K" /c:"Quadro M" /c:"Quadro P" /c:"Tesla K" /c:"Tesla M" /c:"Tesla P" /c:"Tesla V" /c:"TITAN X" /c:"TITAN XP" /c:"TITAN V" >nul && set "%~2=1"
exit /b 0
