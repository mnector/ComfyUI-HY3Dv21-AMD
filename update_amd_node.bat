@echo off
setlocal enabledelayedexpansion
title "ComfyUI-HY3Dv21-AMD - Upstream Sync and AMD Shielding"

cd /d "%~dp0"

:: 1. Read private .env.local if present
if exist "%~dp0.env.local" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.env.local") do (
        set "KEY=%%A"
        set "VAL=%%B"
        if "!KEY!"=="COMFYUI_PYTHON" set "COMFYUI_PYTHON=!VAL!"
    )
)

:: 2. Auto-detect ComfyUI Python environment by priority
if defined COMFYUI_PYTHON if exist "%COMFYUI_PYTHON%" (
    set "PYTHON_EXE=%COMFYUI_PYTHON%"
    goto :run
)

if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
    set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
    goto :run
)

if exist "%~dp0..\..\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\..\venv\Scripts\python.exe"
    goto :run
)

if exist "%~dp0..\..\..\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\..\..\venv\Scripts\python.exe"
    goto :run
)

if exist "%~dp0..\..\python_embeded\python.exe" (
    set "PYTHON_EXE=%~dp0..\..\python_embeded\python.exe"
    goto :run
)

if exist "%~dp0..\..\..\python_embeded\python.exe" (
    set "PYTHON_EXE=%~dp0..\..\..\python_embeded\python.exe"
    goto :run
)

set "PYTHON_EXE=python.exe"

:run
echo =======================================================================
echo  ComfyUI-HY3Dv21-AMD: Auto-Update from Upstream with AMD Shielding
echo =======================================================================
echo [*] Working directory : %~dp0
echo [*] Detected Python   : %PYTHON_EXE%
echo.

"%PYTHON_EXE%" "%~dp0update_from_upstream.py"

if errorlevel 1 (
    echo.
    echo =======================================================================
    echo  [!] Update failed or rolled back to protect your installation.
    echo =======================================================================
) else (
    echo.
    echo =======================================================================
    echo  [OK] Update process finished successfully.
    echo =======================================================================
)

echo.
pause
