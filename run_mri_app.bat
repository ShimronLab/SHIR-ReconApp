@echo off
title SHIR — System for Hybrid Image Reconstruction
echo Starting SHIR - System for Hybrid Image Reconstruction (WSL + conda mri_toolkit)...
pushd "%~dp0AnalyzeMRIDataApp"
wsl bash -ic "conda activate mri_toolkit && python app.py"
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: App exited with code %ERRORLEVEL%
    pause
)
popd
