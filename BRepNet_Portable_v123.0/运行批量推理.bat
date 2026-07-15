@echo off
chcp 65001 >nul 2>&1
title BRepNet Batch Inference

echo ========================================
echo    BRepNet Batch Inference Tool
echo ========================================
echo.

REM Use batch_lib for this tool
set PATH=%~dp0batch_lib;%PATH%
cd /d "%~dp0"

echo Starting batch inference...
echo.
bin\brepnet.exe %*

echo.
echo ========================================
echo    Processing completed!
echo    Output: cpp_logits\
echo            cpp_results\
echo ========================================
echo.
pause
